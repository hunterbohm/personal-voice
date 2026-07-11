"""Guarded local style rendering for frontier-authored canonical drafts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from personal_voice.local_only import (
    inside_git_worktree,
    write_private_json,
    write_private_json_exclusive,
)

_MEDIA = {"text", "dm", "email", "public"}
_EVALUATION_STATUSES = {"rendered", "rejected", "unavailable"}
_MAX_REPLY_WORDS = 256
_MIN_BASE_HELDOUT_RENDERS = 50
_WORD = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)
_PLACEHOLDER = re.compile(r"\[\[PV_LITERAL_\d+\]\]")
_EMPTY_THINK_PREFIX = re.compile(r"\A<think>[\t\r\n ]*</think>[\t\r\n ]*", re.IGNORECASE)
_RENDER_OPEN = "PV_RENDERED_START"
_RENDER_CLOSE = "PV_RENDERED_END"
_PRIVACY_LITERAL = "pvliteraltoken"
# Bump for any prompt, guard, decoder, or promotion-evaluation contract change.
RENDERER_CONTRACT_VERSION = 1
_RENDER_ENVELOPE = re.compile(
    rf"\A{re.escape(_RENDER_OPEN)}\n(?P<draft>[\s\S]*?)\n{re.escape(_RENDER_CLOSE)}\Z"
)
_RENDER_USER_PROMPT = re.compile(
    r"\AMedium: (?P<medium>text|dm|email|public)\nCanonical draft:\n(?P<draft>[\s\S]+)\Z"
)
_DIRECT_PLACEHOLDER = re.compile(r"<\|(?:url|email_address|phone)\|>")
_URL = re.compile(r"(?:https?://|www\.)[^\s<>]+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\w)\+?\d(?:[ .()/-]*\d){7,14}(?!\w)")
_MENTION = re.compile(r"(?<!\w)@[A-Za-z0-9_]{1,30}")
_CURRENCY = re.compile(
    r"(?:[$€£]\s?\d[\d,]*(?:\.\d+)?|\b(?:USD|EUR|GBP)\s+\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"(?<!\w)\d+(?:[.,:/\-]\d+)*(?!\w)")
_QUOTED = re.compile(r'"[^"\n]+"|“[^”\n]+”')
_BUILTIN_LITERAL_PATTERNS = (
    _DIRECT_PLACEHOLDER,
    _URL,
    _EMAIL,
    _PHONE,
    _MENTION,
    _CURRENCY,
    _NUMBER,
    _QUOTED,
)

RENDER_SYSTEM_PROMPT = (
    "Rewrite the complete canonical draft in the writer's learned voice. "
    "Preserve its meaning and every literal. Do not add facts, promises, questions, "
    "permissions, deadlines, or commitments. Return exactly PV_RENDERED_START, a newline, "
    "the rewritten draft, a newline, and PV_RENDERED_END, with nothing else."
)


class RendererBackend(Protocol):
    """Backend boundary used by the renderer guard."""

    version: str

    def generate(self, medium: str, slotized_draft: str) -> str: ...


class MlxSubprocessBackend:
    """Run the pinned local MLX model without putting private drafts in argv."""

    def __init__(
        self,
        *,
        model_path: Path,
        adapter_path: Path | None,
        version: str,
        artifact_hash: str,
        candidate_hash: str,
        executable: str,
        timeout_seconds: int = 120,
    ) -> None:
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.version = version
        self.artifact_hash = artifact_hash
        self.candidate_hash = candidate_hash
        self.executable = executable
        self.timeout_seconds = timeout_seconds

    def generate(self, medium: str, slotized_draft: str) -> str:
        command = [
            self.executable,
            "--model",
            str(self.model_path),
            "--system-prompt",
            RENDER_SYSTEM_PROMPT,
            "--prompt",
            "-",
            "--max-tokens",
            str(_max_render_tokens(slotized_draft)),
            "--temp",
            "0",
            "--verbose",
            "false",
        ]
        if self.adapter_path is not None:
            command[3:3] = ["--adapter-path", str(self.adapter_path)]
        environment = os.environ.copy()
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        result = subprocess.run(
            command,
            input=render_user_prompt(medium, slotized_draft),
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
            env=environment,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError("local renderer failed")
        return result.stdout.strip()


class MlxInProcessBackend:
    """Load one pinned MLX base-plus-adapter for batch evaluation."""

    def __init__(
        self,
        *,
        model_path: Path,
        adapter_path: Path | None,
        version: str,
        artifact_hash: str,
        candidate_hash: str,
    ) -> None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import mlx_lm
        from mlx_lm.sample_utils import make_sampler

        self._generate = mlx_lm.generate
        self._model, self._tokenizer = mlx_lm.load(
            str(model_path),
            adapter_path=str(adapter_path) if adapter_path is not None else None,
        )
        self._sampler = make_sampler(temp=0)
        self.artifact_hash = artifact_hash
        self.candidate_hash = candidate_hash
        self.version = version

    def generate(self, medium: str, slotized_draft: str) -> str:
        messages = [
            {"role": "system", "content": RENDER_SYSTEM_PROMPT},
            {"role": "user", "content": render_user_prompt(medium, slotized_draft)},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        return str(
            self._generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=_max_render_tokens(slotized_draft),
                sampler=self._sampler,
                verbose=False,
            )
        ).strip()


def load_mlx_backend(
    manifest_path: str | Path,
    *,
    use_adapter: bool = True,
) -> MlxSubprocessBackend:
    """Load a private versioned renderer manifest."""

    manifest = Path(manifest_path).expanduser()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if manifest.stat().st_mode & 0o077:
        raise PermissionError("renderer manifest must be private (0600)")
    if not renderer_doctor(manifest)["ok"]:
        raise ValueError("renderer failed integrity checks")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if type(payload.get("schema_version")) is not int or payload.get("schema_version") != 2:
        raise ValueError("unsupported renderer manifest")
    model_path = Path(str(payload.get("model_path") or "")).expanduser()
    adapter_path = Path(str(payload.get("adapter_path") or "")).expanduser()
    if not model_path.is_dir() or (use_adapter and not adapter_path.is_dir()):
        raise FileNotFoundError("renderer model or adapter is missing")
    executable = _mlx_generate_executable()
    if not executable:
        raise FileNotFoundError("mlx_lm.generate is unavailable")
    return MlxSubprocessBackend(
        model_path=model_path,
        adapter_path=adapter_path if use_adapter else None,
        version=str(payload.get("version") or "unknown"),
        artifact_hash=str(payload.get("adapter_sha256") or "unknown"),
        candidate_hash=str(payload.get("candidate_sha256") or "unknown"),
        executable=executable,
    )


def load_mlx_inprocess_backend(
    manifest_path: str | Path,
    *,
    use_adapter: bool = True,
) -> MlxInProcessBackend:
    """Load a candidate once for private aggregate evaluation commands."""

    manifest = Path(manifest_path).expanduser()
    if not manifest.is_file() or manifest.stat().st_mode & 0o077:
        raise FileNotFoundError("private renderer manifest is missing")
    if not renderer_doctor(manifest)["ok"]:
        raise ValueError("renderer failed integrity checks")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    model_path = Path(str(payload["model_path"])).expanduser()
    adapter_path = Path(str(payload["adapter_path"])).expanduser()
    return MlxInProcessBackend(
        model_path=model_path,
        adapter_path=adapter_path if use_adapter else None,
        version=str(payload["version"]),
        artifact_hash=str(payload["adapter_sha256"]),
        candidate_hash=str(payload["candidate_sha256"]),
    )


def install_renderer_manifest(
    renderer_root: str | Path,
    *,
    model_path: str | Path,
    adapter_path: str | Path,
    version: str,
    base_model: str,
    base_revision: str,
) -> Path:
    """Stage a versioned private base-plus-adapter renderer for evaluation."""

    root = Path(renderer_root).expanduser()
    model = Path(model_path).expanduser().resolve()
    adapter = Path(adapter_path).expanduser().resolve()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", version):
        raise ValueError("renderer version contains unsupported characters")
    if not model.is_dir() or not adapter.is_dir():
        raise FileNotFoundError("renderer model or adapter is missing")
    if inside_git_worktree(root) or inside_git_worktree(adapter):
        raise ValueError("renderer artifacts must live outside a Git worktree")
    artifacts_valid, training_succeeded = _adapter_artifact_status(adapter)
    if not artifacts_valid or not training_succeeded:
        raise ValueError("renderer adapter is incomplete or unsuccessful")
    training_metadata = _adapter_training_metadata(adapter)
    if not _model_is_4bit(model) or model.name != base_revision:
        raise ValueError("renderer base model is not the pinned 4-bit revision")
    if not _adapter_matches_model(adapter, model):
        raise ValueError("renderer adapter was trained against a different base model")
    _make_private_directory(root)
    versions = root / "versions"
    _make_private_directory(versions)
    _make_private_tree(adapter)
    payload: dict[str, object] = {
        "adapter_path": str(adapter),
        "adapter_sha256": _directory_sha256(adapter),
        "base_model": base_model,
        "base_revision": base_revision,
        "model_path": str(model),
        "model_fingerprint": _model_fingerprint(model),
        "renderer_contract_version": RENDERER_CONTRACT_VERSION,
        "schema_version": 2,
        "training_dataset_records": int(training_metadata["dataset_records"]),
        "training_dataset_sha256": str(training_metadata["dataset_sha256"]),
        "version": version,
    }
    payload["candidate_sha256"] = _candidate_manifest_sha256(payload)
    version_manifest = versions / f"{version}.json"
    try:
        write_private_json_exclusive(version_manifest, payload)
    except FileExistsError as error:
        raise FileExistsError("renderer version is immutable and already installed") from error
    return version_manifest


def activate_renderer_manifest(
    renderer_root: str | Path,
    version: str,
    *,
    blind_pack: str | Path | None = None,
    privacy_report: str | Path | None = None,
    runtime_report: str | Path | None = None,
    data_path: str | Path | None = None,
) -> Path:
    """Promote a candidate only after every candidate-bound gate passes."""

    root = Path(renderer_root).expanduser()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", version):
        raise ValueError("renderer version contains unsupported characters")
    version_manifest = root / "versions" / f"{version}.json"
    if not version_manifest.is_file() or version_manifest.stat().st_mode & 0o077:
        raise FileNotFoundError("private renderer version manifest is missing")
    manifest_bytes = version_manifest.read_bytes()
    payload = json.loads(manifest_bytes)
    if not isinstance(payload, dict) or payload.get("version") != version:
        raise ValueError("renderer version manifest is invalid")
    from personal_voice.blind_eval import blind_eval_stats
    from personal_voice.privacy_probe import (
        privacy_report_eligible,
        require_candidate_dataset_binding,
    )
    from personal_voice.renderer_eval import evaluate_renderer_dataset
    from personal_voice.renderer_semantic import evaluate_semantic_gates
    from personal_voice.runtime_gate import runtime_report_eligible

    binding = (
        version,
        str(payload.get("adapter_sha256") or ""),
        str(payload.get("candidate_sha256") or ""),
    )
    data = Path(data_path).expanduser() if data_path is not None else root / "data/mlx"
    with _private_manifest_snapshot(root, manifest_bytes) as manifest_snapshot:
        if not renderer_doctor(manifest_snapshot)["ok"]:
            raise ValueError("renderer failed integrity checks")
        if blind_pack is None:
            raise PermissionError("renderer activation requires blind evaluation evidence")
        try:
            require_candidate_dataset_binding(manifest_snapshot, data)
        except ValueError as error:
            raise PermissionError(
                "renderer activation requires the candidate-bound training dataset"
            ) from error

        blind = blind_eval_stats(
            blind_pack,
            model_version=version,
            candidate_sha256=binding[2],
            renderer_contract_version=RENDERER_CONTRACT_VERSION,
        )
        if not blind["activation_eligible"]:
            raise PermissionError("renderer candidate has not passed blind evaluation")
        privacy_evidence = (
            Path(privacy_report).expanduser()
            if privacy_report is not None
            else root / "evaluation/privacy" / f"{version}.json"
        )
        if not privacy_report_eligible(
            privacy_evidence,
            model_version=version,
            adapter_sha256=binding[1],
            candidate_sha256=binding[2],
            dataset_sha256=str(payload.get("training_dataset_sha256") or ""),
            dataset_records=int(payload.get("training_dataset_records") or 0),
        ):
            raise PermissionError("renderer candidate has not passed the privacy probe")
        runtime_evidence = (
            Path(runtime_report).expanduser()
            if runtime_report is not None
            else root / "evaluation/runtime" / f"{version}.json"
        )
        if not runtime_report_eligible(
            runtime_evidence,
            model_version=version,
            adapter_sha256=binding[1],
            candidate_sha256=binding[2],
        ):
            raise PermissionError("renderer candidate has not passed the local runtime benchmark")
        tuned_backend = load_mlx_inprocess_backend(manifest_snapshot)
        _require_backend_binding(tuned_backend, binding)
        if not evaluate_semantic_gates(tuned_backend)["ok"]:
            raise PermissionError("renderer candidate has not passed semantic gates")
        tuned_evaluation = evaluate_renderer_dataset(data, tuned_backend)
        base_backend = load_mlx_inprocess_backend(manifest_snapshot, use_adapter=False)
        _require_backend_binding(base_backend, binding)
        base_evaluation = evaluate_renderer_dataset(data, base_backend)
        try:
            require_candidate_dataset_binding(manifest_snapshot, data)
        except ValueError as error:
            raise PermissionError(
                "renderer training dataset changed during held-out evaluation"
            ) from error
        if not _style_evaluation_eligible(tuned_evaluation, base_evaluation):
            raise PermissionError("renderer fine-tune has not beaten the 4-bit base model")
        if not renderer_doctor(manifest_snapshot)["ok"] or not _manifest_unchanged(
            version_manifest, manifest_bytes
        ):
            raise PermissionError("renderer manifest or artifacts changed during activation")
        current = root / "current.json"
        write_private_json(current, payload)
        return current


def _require_backend_binding(backend: object, binding: tuple[str, str, str]) -> None:
    actual = (
        str(getattr(backend, "version", "")),
        str(getattr(backend, "artifact_hash", "")),
        str(getattr(backend, "candidate_hash", "")),
    )
    if actual != binding:
        raise PermissionError("renderer backend changed during activation")


def _manifest_unchanged(path: Path, expected: bytes) -> bool:
    try:
        return bool(
            path.is_file()
            and not path.is_symlink()
            and not path.stat().st_mode & 0o077
            and path.read_bytes() == expected
        )
    except OSError:
        return False


@contextmanager
def _private_manifest_snapshot(root: Path, payload: bytes):
    descriptor, name = tempfile.mkstemp(prefix=".activation-manifest-", suffix=".json", dir=root)
    snapshot = Path(name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
        descriptor = -1
        yield snapshot
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        snapshot.unlink(missing_ok=True)


def _style_evaluation_eligible(tuned: dict[str, object], base: dict[str, object]) -> bool:
    tuned_total = int(tuned.get("total") or 0)
    base_total = int(base.get("total") or 0)
    tuned_statuses = tuned.get("statuses")
    base_statuses = base.get("statuses")
    tuned_mediums = tuned.get("mediums")
    base_mediums = base.get("mediums")
    if (
        not _exact_count_map(tuned_statuses, _EVALUATION_STATUSES, tuned_total)
        or not _exact_count_map(base_statuses, _EVALUATION_STATUSES, base_total)
        or not _exact_count_map(tuned_mediums, _MEDIA, tuned_total)
        or not _exact_count_map(base_mediums, _MEDIA, base_total)
    ):
        return False
    tuned_rendered = int(tuned_statuses["rendered"])
    base_rendered = int(base_statuses["rendered"])
    tuned_unavailable = int(tuned_statuses["unavailable"])
    base_unavailable = int(base_statuses["unavailable"])
    return bool(
        tuned_total >= 100
        and tuned_total == base_total
        and tuned_mediums == base_mediums
        and tuned_rendered / tuned_total >= 0.8
        and base_rendered >= _MIN_BASE_HELDOUT_RENDERS
        and tuned_rendered > base_rendered
        and tuned_unavailable == 0
        and base_unavailable == 0
        and float(tuned.get("mean_similarity_delta") or 0.0) > 0.0
        and float(tuned.get("mean_similarity_delta") or 0.0)
        > float(base.get("mean_similarity_delta") or 0.0)
        and int(tuned.get("closer_to_target") or 0) > int(base.get("closer_to_target") or 0)
    )


def _exact_count_map(value: object, keys: set[str], total: int) -> bool:
    return bool(
        isinstance(value, dict)
        and set(value) == keys
        and all(type(count) is int and count >= 0 for count in value.values())
        and sum(value.values()) == total
    )


def renderer_doctor(manifest_path: str | Path) -> dict[str, object]:
    """Return aggregate-only renderer readiness checks."""

    manifest = Path(manifest_path).expanduser()
    manifest_private = manifest.is_file() and not bool(manifest.stat().st_mode & 0o077)
    payload: dict[str, object] = {}
    if manifest_private:
        try:
            loaded = json.loads(manifest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            loaded = {}
        if isinstance(loaded, dict):
            payload = loaded
    model_value = str(payload.get("model_path") or "").strip()
    adapter_value = str(payload.get("adapter_path") or "").strip()
    model = Path(model_value).expanduser()
    adapter = Path(adapter_value).expanduser()
    model_exists = bool(model_value) and model.is_dir()
    adapter_exists = bool(adapter_value) and adapter.is_dir()
    adapter_private = adapter_exists and _tree_is_private(adapter)
    adapter_artifacts_valid, training_succeeded = (
        _adapter_artifact_status(adapter) if adapter_exists else (False, False)
    )
    training_metadata = _adapter_training_metadata(adapter) if adapter_exists else {}
    training_dataset_match = bool(
        training_succeeded
        and payload.get("training_dataset_sha256") == training_metadata.get("dataset_sha256")
        and payload.get("training_dataset_records") == training_metadata.get("dataset_records")
    )
    adapter_base_match = bool(
        adapter_artifacts_valid and model_exists and _adapter_matches_model(adapter, model)
    )
    model_4bit = model_exists and _model_is_4bit(model)
    model_revision_match = bool(
        model_exists
        and str(payload.get("base_revision") or "")
        and model.name == str(payload.get("base_revision"))
    )
    expected_model_fingerprint = str(payload.get("model_fingerprint") or "")
    try:
        actual_model_fingerprint = _model_fingerprint(model) if model_exists else ""
    except (OSError, ValueError):
        actual_model_fingerprint = ""
    model_fingerprint_match = bool(
        model_exists
        and expected_model_fingerprint
        and actual_model_fingerprint == expected_model_fingerprint
    )
    expected_hash = str(payload.get("adapter_sha256") or "")
    adapter_hash_match = bool(
        adapter_private and expected_hash and _directory_sha256(adapter) == expected_hash
    )
    expected_candidate_hash = str(payload.get("candidate_sha256") or "")
    candidate_hash_match = bool(
        re.fullmatch(r"[0-9a-f]{64}", expected_candidate_hash)
        and _candidate_manifest_sha256(payload) == expected_candidate_hash
    )
    executable_available = bool(_mlx_generate_executable())
    schema_version = payload.get("schema_version")
    schema_version_valid = type(schema_version) is int and schema_version == 2
    contract_version = payload.get("renderer_contract_version", 1)
    contract_version_valid = (
        type(contract_version) is int and contract_version == RENDERER_CONTRACT_VERSION
    )
    result: dict[str, object] = {
        "adapter_base_match": adapter_base_match,
        "adapter_artifacts_valid": adapter_artifacts_valid,
        "adapter_hash_match": adapter_hash_match,
        "adapter_private": adapter_private,
        "candidate_hash_match": candidate_hash_match,
        "contract_version": contract_version,
        "contract_version_match": contract_version_valid,
        "executable_available": executable_available,
        "manifest_private": manifest_private,
        "model_4bit": model_4bit,
        "model_exists": model_exists,
        "model_fingerprint_match": model_fingerprint_match,
        "model_revision_match": model_revision_match,
        "ok": False,
        "schema_version": schema_version,
        "training_succeeded": training_succeeded,
        "training_dataset_match": training_dataset_match,
        "version": str(payload.get("version") or "unknown"),
    }
    result["ok"] = bool(
        schema_version_valid
        and contract_version_valid
        and result["model_exists"]
        and adapter_base_match
        and manifest_private
        and adapter_private
        and adapter_artifacts_valid
        and adapter_hash_match
        and candidate_hash_match
        and executable_available
        and model_4bit
        and model_fingerprint_match
        and model_revision_match
        and training_succeeded
        and training_dataset_match
    )
    return result


def render_user_prompt(medium: str, draft: str) -> str:
    """Build the exact user prompt shared by training and inference."""

    return f"Medium: {medium}\nCanonical draft:\n{draft}"


def parse_renderer_user_prompt(value: str) -> str:
    """Extract the private canonical body from an exact renderer user prompt."""

    match = _RENDER_USER_PROMPT.fullmatch(str(value))
    if match is None:
        raise ValueError("renderer user prompt is invalid")
    return match.group("draft").strip()


def format_renderer_target(draft: str) -> str:
    """Wrap one training target in the strict inference envelope."""

    return f"{_RENDER_OPEN}\n{draft}\n{_RENDER_CLOSE}"


def parse_renderer_target(value: str) -> str:
    """Extract a draft body from the exact training and inference envelope."""

    envelope = _RENDER_ENVELOPE.fullmatch(str(value).strip())
    if envelope is None:
        raise ValueError("renderer target has an invalid envelope")
    draft = envelope.group("draft").strip()
    if not draft:
        raise ValueError("renderer target is empty")
    return draft


def parse_renderer_output(value: str) -> str:
    """Extract the body from the only runtime output shape the guard accepts."""

    candidate = _EMPTY_THINK_PREFIX.sub("", str(value).strip(), count=1).strip()
    if "<think>" in candidate.casefold() or "</think>" in candidate.casefold():
        raise ValueError("renderer output contains nonempty thinking")
    return parse_renderer_target(candidate)


def renderer_literal_skeleton(value: str) -> str:
    """Normalize slotized and restored literals for aggregate privacy comparisons."""

    source = _PLACEHOLDER.sub(f" {_PRIVACY_LITERAL} ", str(value))
    pieces: list[str] = []
    cursor = 0
    for start, end, _literal in _selected_literal_spans(source, ()):
        pieces.append(source[cursor:start])
        pieces.append(f" {_PRIVACY_LITERAL} ")
        cursor = end
    pieces.append(source[cursor:])
    return "".join(pieces)


def literal_counter(text: str) -> Counter[str]:
    """Return exact immutable literals found in text."""

    return Counter(value for _, _, value in _selected_literal_spans(str(text), ()))


def literal_sequence(text: str) -> tuple[str, ...]:
    """Return immutable literals in semantic clause order."""

    return tuple(value for _, _, value in _selected_literal_spans(str(text), ()))


def slotize_training_pair(canonical: str, target: str) -> tuple[str, str]:
    """Replace identical literals with matching placeholders in a training pair."""

    if _PLACEHOLDER.search(canonical) or _PLACEHOLDER.search(target):
        raise ValueError("training pair contains a reserved placeholder")
    if literal_counter(canonical) != literal_counter(target):
        raise ValueError("training pair literals do not match")
    if literal_sequence(canonical) != literal_sequence(target):
        raise ValueError("training pair literal order does not match")
    slotized_target, slots = _slotize(target, ())
    slotized_canonical, canonical_slots = _slotize(canonical, ())
    target_placeholder = {literal: placeholder for placeholder, literal in slots.items()}
    translations = {
        placeholder: target_placeholder[literal] for placeholder, literal in canonical_slots.items()
    }
    slotized_canonical = _PLACEHOLDER.sub(
        lambda match: translations[match.group()], slotized_canonical
    )
    if Counter(_PLACEHOLDER.findall(slotized_canonical)) != Counter(
        _PLACEHOLDER.findall(slotized_target)
    ):
        raise ValueError("training pair literal occurrences do not align")
    return slotized_canonical, slotized_target


def render_draft(
    draft: str,
    medium: str,
    backend: RendererBackend,
    *,
    preserve_literals: Iterable[str] = (),
) -> dict[str, str]:
    """Rewrite a complete draft while deterministically preserving literals."""

    original = str(draft or "")
    canonical = original.strip()
    destination = str(medium or "").strip().casefold()
    if not canonical:
        raise ValueError("draft is required")
    if destination not in _MEDIA:
        raise ValueError("medium must be text, dm, email, or public")

    version = str(getattr(backend, "version", "unknown"))
    if len(_WORD.findall(canonical)) > _MAX_REPLY_WORDS:
        return _fallback(original, version, "rejected", "long_form_unsupported")
    if _RENDER_OPEN in canonical or _RENDER_CLOSE in canonical:
        return _fallback(original, version, "rejected", "reserved_control_token")
    if _PLACEHOLDER.search(canonical):
        return _fallback(original, version, "rejected", "reserved_placeholder")
    try:
        slotized, slots = _slotize(canonical, preserve_literals)
    except ValueError:
        return _fallback(original, version, "rejected", "preserve_literal_missing")
    try:
        candidate = str(backend.generate(destination, slotized)).strip()
    except Exception:
        return _fallback(original, version, "unavailable", "backend_unavailable")

    candidate = _EMPTY_THINK_PREFIX.sub("", candidate, count=1).strip()
    if "<think>" in candidate.casefold() or "</think>" in candidate.casefold():
        return _fallback(original, version, "rejected", "thinking_output")
    try:
        candidate = parse_renderer_target(candidate)
    except ValueError:
        return _fallback(original, version, "rejected", "malformed_output")
    error = _candidate_error(slotized, candidate)
    if error:
        return _fallback(original, version, "rejected", error)
    restored = candidate
    for placeholder, literal in slots.items():
        restored = restored.replace(placeholder, literal)
    if not _acceptable_length(canonical, restored):
        return _fallback(original, version, "rejected", "length_ratio")
    return {
        "model_version": version,
        "reason_code": "",
        "status": "rendered",
        "text": restored,
    }


def _slotize(text: str, preserve_literals: Iterable[str]) -> tuple[str, dict[str, str]]:
    selected = _selected_literal_spans(text, preserve_literals)
    literal_placeholders: dict[str, str] = {}
    slots: dict[str, str] = {}
    pieces: list[str] = []
    cursor = 0
    for start, end, literal in selected:
        pieces.append(text[cursor:start])
        placeholder = literal_placeholders.get(literal)
        if placeholder is None:
            placeholder = f"[[PV_LITERAL_{len(literal_placeholders)}]]"
            literal_placeholders[literal] = placeholder
            slots[placeholder] = literal
        pieces.append(placeholder)
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces), slots


def _selected_literal_spans(
    text: str, preserve_literals: Iterable[str]
) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, int, str]] = []
    for literal in preserve_literals:
        value = str(literal or "")
        if not value:
            continue
        matches = list(re.finditer(re.escape(value), text))
        if not matches:
            raise ValueError("a preserve literal is not present in the draft")
        spans.extend((match.start(), match.end(), 0, match.group()) for match in matches)
    for pattern_index, pattern in enumerate(_BUILTIN_LITERAL_PATTERNS, start=1):
        spans.extend(
            (match.start(), match.end(), pattern_index, match.group())
            for match in pattern.finditer(text)
        )
    selected: list[tuple[int, int, str]] = []
    occupied_until = -1
    for start, end, priority, value in sorted(
        spans, key=lambda item: (item[0], item[2], -(item[1] - item[0]))
    ):
        del priority
        if start < occupied_until:
            continue
        selected.append((start, end, value))
        occupied_until = end
    return selected


def _candidate_error(slotized: str, candidate: str) -> str:
    if not candidate:
        return "empty"
    lowered = candidate.casefold()
    if "<think>" in lowered or "</think>" in lowered:
        return "thinking_output"
    expected_placeholders = _PLACEHOLDER.findall(slotized)
    candidate_placeholders = _PLACEHOLDER.findall(candidate)
    if Counter(expected_placeholders) != Counter(candidate_placeholders):
        return "literal_mismatch"
    if expected_placeholders != candidate_placeholders:
        return "literal_order"
    without_placeholders = _PLACEHOLDER.sub("", candidate)
    if any(pattern.search(without_placeholders) for pattern in _BUILTIN_LITERAL_PATTERNS):
        return "unexpected_literal"
    return ""


def _acceptable_length(draft: str, rendered: str) -> bool:
    draft_words = len(_WORD.findall(draft))
    rendered_words = len(_WORD.findall(rendered))
    if draft_words < 5:
        return rendered_words <= max(12, draft_words * 3)
    return 0.5 <= rendered_words / draft_words <= 2.0


def _max_render_tokens(draft: str) -> int:
    return min(512, max(32, len(_WORD.findall(draft)) * 3 + 16))


def _fallback(draft: str, version: str, status: str, reason: str) -> dict[str, str]:
    return {
        "model_version": version,
        "reason_code": reason,
        "status": status,
        "text": draft,
    }


def _make_private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)


def _make_private_tree(path: Path) -> None:
    if path.is_symlink() or any(child.is_symlink() for child in path.rglob("*")):
        raise ValueError("renderer adapter tree cannot contain symlinks")
    path.chmod(0o700)
    for child in path.rglob("*"):
        child.chmod(0o700 if child.is_dir() else 0o600)


def _tree_is_private(path: Path) -> bool:
    if path.is_symlink() or path.stat().st_mode & 0o077:
        return False
    return all(
        not child.is_symlink() and not bool(child.stat().st_mode & 0o077)
        for child in path.rglob("*")
    )


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files: list[Path] = []
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            if not item.exists() or not item.is_file():
                raise ValueError("renderer artifact tree contains a non-regular artifact")
            files.append(item)
        elif item.is_dir():
            continue
        elif item.is_file():
            files.append(item)
        else:
            raise ValueError("renderer artifact tree contains a non-regular artifact")
    digest.update(b"personal-voice-directory-sha256-v2\0")
    digest.update(len(files).to_bytes(8, "big"))
    for child in files:
        relative = str(child.relative_to(path)).encode()
        file_digest = hashlib.sha256()
        size = 0
        with child.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                file_digest.update(chunk)
                size += len(chunk)
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(size.to_bytes(8, "big"))
        digest.update(file_digest.digest())
    return digest.hexdigest()


def _candidate_manifest_sha256(payload: dict[str, object]) -> str:
    fields = [
        "adapter_sha256",
        "base_model",
        "base_revision",
        "model_fingerprint",
        "schema_version",
        "training_dataset_records",
        "training_dataset_sha256",
        "version",
    ]
    if "renderer_contract_version" in payload:
        fields.append("renderer_contract_version")
    stable = {field: payload.get(field) for field in fields}
    encoded = json.dumps(stable, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _adapter_artifact_status(path: Path) -> tuple[bool, bool]:
    if not path.is_dir():
        return False, False
    artifacts_valid = all(
        (path / name).is_file()
        for name in ("adapter_config.json", "adapters.safetensors", "training-run.json")
    )
    if not artifacts_valid:
        return False, False
    report = _adapter_training_metadata(path)
    dataset_sha256 = str(report.get("dataset_sha256") or "")
    dataset_records = report.get("dataset_records")
    training_succeeded = bool(
        type(report.get("exit_code")) is int
        and report.get("exit_code") == 0
        and report.get("dataset_unchanged") is True
        and re.fullmatch(r"[0-9a-f]{64}", dataset_sha256)
        and type(dataset_records) is int
        and dataset_records > 0
    )
    return True, training_succeeded


def _adapter_training_metadata(path: Path) -> dict[str, object]:
    try:
        report = json.loads((path / "training-run.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return report if isinstance(report, dict) else {}


def _adapter_matches_model(adapter: Path, model: Path) -> bool:
    try:
        payload = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(payload, dict):
        return False
    configured_model = str(payload.get("model") or "")
    configured_path = Path(configured_model).expanduser()
    return bool(
        payload.get("fine_tune_type") in {"lora", "dora"}
        and payload.get("mask_prompt") is True
        and configured_model
        and (configured_path.resolve() == model.resolve() or configured_path.name == model.name)
    )


def _model_is_4bit(model: Path) -> bool:
    try:
        payload = json.loads((model / "config.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(payload, dict):
        return False
    quantization = payload.get("quantization") or payload.get("quantization_config") or {}
    return isinstance(quantization, dict) and quantization.get("bits") == 4


def _model_fingerprint(model: Path) -> str:
    weights = list(model.glob("*.safetensors"))
    if not weights or any(not path.is_file() for path in weights):
        raise ValueError("renderer base model has an invalid weight artifact")
    return _directory_sha256(model)


def _mlx_generate_executable() -> str | None:
    configured = os.environ.get("PERSONAL_VOICE_MLX_GENERATE")
    if configured:
        candidate = Path(configured).expanduser()
        return str(candidate) if candidate.is_file() and os.access(candidate, os.X_OK) else None
    sibling = Path(sys.executable).resolve().parent / "mlx_lm.generate"
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    return shutil.which("mlx_lm.generate")
