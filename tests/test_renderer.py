import json
import os
from pathlib import Path

import pytest

from personal_voice.renderer import (
    _candidate_manifest_sha256,
    _mlx_generate_executable,
    activate_renderer_manifest,
    format_renderer_target,
    install_renderer_manifest,
    load_mlx_backend,
    parse_renderer_target,
    parse_renderer_user_prompt,
    render_draft,
    render_user_prompt,
    renderer_doctor,
    renderer_literal_skeleton,
    slotize_training_pair,
)


class FakeBackend:
    version = "synthetic-adapter-v1"

    def __init__(self, transform, *, raw: bool = False):
        self.transform = transform
        self.raw = raw

    def generate(self, medium: str, slotized_draft: str) -> str:
        result = self.transform(medium, slotized_draft)
        return result if self.raw else format_renderer_target(result)


def test_mlx_executable_discovery_keeps_the_virtualenv_bin_directory(
    tmp_path: Path, monkeypatch
) -> None:
    real_bin = tmp_path / "runtime/bin"
    real_bin.mkdir(parents=True)
    real_python = real_bin / "python3"
    real_python.write_text("#!/bin/sh\nexit 0\n")
    real_python.chmod(0o700)

    venv_bin = tmp_path / "venv/bin"
    venv_bin.mkdir(parents=True)
    linked_python = venv_bin / "python"
    linked_python.symlink_to(real_python)
    sibling = venv_bin / "mlx_lm.generate"
    sibling.write_text("#!/bin/sh\nexit 0\n")
    sibling.chmod(0o700)

    monkeypatch.delenv("PERSONAL_VOICE_MLX_GENERATE", raising=False)
    monkeypatch.setattr("personal_voice.renderer.sys.executable", str(linked_python))
    monkeypatch.setattr("personal_voice.renderer.shutil.which", lambda _name: None)

    assert _mlx_generate_executable() == str(sibling)


def test_renderer_target_envelope_round_trips_and_rejects_wrappers() -> None:
    target = "yeah that works for me"

    assert parse_renderer_target(format_renderer_target(target)) == target
    with pytest.raises(ValueError, match="invalid envelope"):
        parse_renderer_target(f"Sure, here it is: {format_renderer_target(target)}")


def test_renderer_user_prompt_parser_extracts_only_the_private_canonical_body() -> None:
    canonical = "private alpha beta"

    assert parse_renderer_user_prompt(render_user_prompt("text", canonical)) == canonical
    with pytest.raises(ValueError, match="invalid"):
        parse_renderer_user_prompt(f"Canonical draft:\n{canonical}")


def test_renderer_literal_skeleton_matches_training_slots_to_restored_literals() -> None:
    slotized = "I can meet at [[PV_LITERAL_0]] about [[PV_LITERAL_1]]."
    restored = "I can meet at 3:30 about $250."

    assert (
        renderer_literal_skeleton(slotized).split() == renderer_literal_skeleton(restored).split()
    )


def _write_synthetic_model(path: Path) -> None:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({"model_type": "synthetic", "quantization": {"bits": 4}})
    )
    (path / "model.safetensors").write_bytes(b"synthetic-model")


def _write_successful_adapter(
    path: Path,
    model: Path,
    weights: bytes = b"synthetic-adapter",
) -> None:
    path.mkdir()
    (path / "adapters.safetensors").write_bytes(weights)
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "fine_tune_type": "lora",
                "mask_prompt": True,
                "model": str(model.resolve()),
                "rank": 8,
            }
        )
    )
    (path / "training-run.json").write_text(
        json.dumps(
            {
                "dataset_records": 100,
                "dataset_sha256": "d" * 64,
                "dataset_unchanged": True,
                "exit_code": 0,
            }
        )
    )


def test_render_draft_restores_immutable_literals_after_style_rewrite() -> None:
    draft = "I can review Project Atlas at 3:30 on Friday."
    backend = FakeBackend(
        lambda medium, text: f"yeah for {medium} — " + text.replace("I can", "I can definitely")
    )

    result = render_draft(
        draft,
        "text",
        backend,
        preserve_literals=["Project Atlas", "Friday"],
    )

    assert result == {
        "model_version": "synthetic-adapter-v1",
        "reason_code": "",
        "status": "rendered",
        "text": "yeah for text — I can definitely review Project Atlas at 3:30 on Friday.",
    }


def test_render_draft_rejects_missing_literals_and_returns_original() -> None:
    draft = "I can meet at 3:30 on Friday."
    backend = FakeBackend(lambda _medium, text: text.replace("[[PV_LITERAL_0]]", ""))

    result = render_draft(draft, "dm", backend, preserve_literals=["Friday"])

    assert result == {
        "model_version": "synthetic-adapter-v1",
        "reason_code": "literal_mismatch",
        "status": "rejected",
        "text": draft,
    }


def test_render_draft_rejects_reordered_literals_and_returns_original() -> None:
    draft = "I can meet at 3:30 on Friday."

    def swap(_medium: str, text: str) -> str:
        return (
            text.replace("[[PV_LITERAL_0]]", "[[PV_SWAP]]")
            .replace("[[PV_LITERAL_1]]", "[[PV_LITERAL_0]]")
            .replace("[[PV_SWAP]]", "[[PV_LITERAL_1]]")
        )

    result = render_draft(draft, "dm", FakeBackend(swap), preserve_literals=["Friday"])

    assert result == {
        "model_version": "synthetic-adapter-v1",
        "reason_code": "literal_order",
        "status": "rejected",
        "text": draft,
    }


def test_render_draft_rejects_reserved_placeholder_input() -> None:
    draft = "Please keep [[PV_LITERAL_0]] exactly as written."

    result = render_draft(draft, "text", FakeBackend(lambda _medium, text: text))

    assert result == {
        "model_version": "synthetic-adapter-v1",
        "reason_code": "reserved_placeholder",
        "status": "rejected",
        "text": draft,
    }


def test_render_draft_falls_back_when_preserve_literal_is_absent() -> None:
    draft = "That works for me."

    result = render_draft(
        draft,
        "text",
        FakeBackend(lambda _medium, text: text),
        preserve_literals=["missing name"],
    )

    assert result == {
        "model_version": "synthetic-adapter-v1",
        "reason_code": "preserve_literal_missing",
        "status": "rejected",
        "text": draft,
    }


def test_render_draft_rejects_long_form_outside_reply_training_scope() -> None:
    draft = " ".join(["word"] * 257)

    result = render_draft(draft, "public", FakeBackend(lambda _medium, text: text))

    assert result == {
        "model_version": "synthetic-adapter-v1",
        "reason_code": "long_form_unsupported",
        "status": "rejected",
        "text": draft,
    }


def test_render_draft_falls_back_when_backend_is_unavailable() -> None:
    draft = "\nThat works for me.\n"

    def unavailable(_medium: str, _text: str) -> str:
        raise RuntimeError("synthetic backend failure")

    result = render_draft(draft, "email", FakeBackend(unavailable))

    assert result == {
        "model_version": "synthetic-adapter-v1",
        "reason_code": "backend_unavailable",
        "status": "unavailable",
        "text": draft,
    }


def test_render_draft_strips_only_an_empty_thinking_wrapper() -> None:
    draft = "That works for me."
    empty_wrapper = FakeBackend(
        lambda _medium, text: (
            "<think>\n\n</think>\n\n" + format_renderer_target(text.replace("That", "yeah that"))
        ),
        raw=True,
    )
    nonempty_wrapper = FakeBackend(
        lambda _medium, text: "<think>reasoning</think>\n\n" + format_renderer_target(text),
        raw=True,
    )

    accepted = render_draft(draft, "text", empty_wrapper)
    rejected = render_draft(draft, "text", nonempty_wrapper)

    assert accepted["status"] == "rendered"
    assert accepted["text"] == "yeah that works for me."
    assert rejected["status"] == "rejected"
    assert rejected["reason_code"] == "thinking_output"
    assert rejected["text"] == draft


def test_render_draft_rejects_model_meta_wrappers() -> None:
    draft = "That works for me."
    backend = FakeBackend(
        lambda _medium, text: f"Sure — here is the rewritten draft: {text}",
        raw=True,
    )

    result = render_draft(draft, "text", backend)

    assert result == {
        "model_version": "synthetic-adapter-v1",
        "reason_code": "malformed_output",
        "status": "rejected",
        "text": draft,
    }


def test_install_renderer_manifest_is_private_versioned_and_verifiable(
    tmp_path: Path, monkeypatch
) -> None:
    private = tmp_path / "renderer-private"
    private.mkdir(mode=0o700)
    model = private / "base-model"
    adapter = private / "adapter-v1"
    _write_synthetic_model(model)
    _write_successful_adapter(adapter, model)
    executable = tmp_path / "mlx_lm.generate"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    monkeypatch.setenv("PERSONAL_VOICE_MLX_GENERATE", str(executable))

    manifest = install_renderer_manifest(
        private,
        model_path=model,
        adapter_path=adapter,
        version="synthetic-v1",
        base_model="synthetic/base",
        base_revision=model.name,
    )

    assert manifest == private / "versions/synthetic-v1.json"
    assert manifest.stat().st_mode & 0o077 == 0
    assert (private / "versions/synthetic-v1.json").stat().st_mode & 0o077 == 0
    assert not (private / "current.json").exists()
    assert all(path.stat().st_mode & 0o077 == 0 for path in adapter.rglob("*"))
    assert renderer_doctor(manifest) == {
        "adapter_base_match": True,
        "adapter_artifacts_valid": True,
        "adapter_hash_match": True,
        "adapter_private": True,
        "candidate_hash_match": True,
        "contract_version": 1,
        "contract_version_match": True,
        "executable_available": True,
        "manifest_private": True,
        "model_4bit": True,
        "model_exists": True,
        "model_fingerprint_match": True,
        "model_revision_match": True,
        "ok": True,
        "schema_version": 2,
        "training_dataset_match": True,
        "training_succeeded": True,
        "version": "synthetic-v1",
    }


def test_load_fails_closed_after_base_model_tampering(tmp_path: Path, monkeypatch) -> None:
    private = tmp_path / "renderer-private"
    private.mkdir(mode=0o700)
    model = private / "model-revision"
    _write_synthetic_model(model)
    adapter = private / "adapter"
    _write_successful_adapter(adapter, model)
    executable = tmp_path / "mlx_lm.generate"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    monkeypatch.setenv("PERSONAL_VOICE_MLX_GENERATE", str(executable))
    manifest = install_renderer_manifest(
        private,
        model_path=model,
        adapter_path=adapter,
        version="candidate-v1",
        base_model="synthetic/base",
        base_revision=model.name,
    )
    (model / "config.json").write_text(
        json.dumps({"model_type": "synthetic", "quantization": {"bits": 8}})
    )

    with pytest.raises(ValueError, match="failed integrity checks"):
        load_mlx_backend(manifest)
    doctor = renderer_doctor(manifest)
    assert doctor["model_4bit"] is False
    assert doctor["model_fingerprint_match"] is False


def test_load_fails_closed_after_same_size_middle_weight_tampering(
    tmp_path: Path, monkeypatch
) -> None:
    private = tmp_path / "renderer-private"
    private.mkdir(mode=0o700)
    model = private / "model-revision"
    _write_synthetic_model(model)
    weights = model / "model.safetensors"
    weights.write_bytes(b"a" * (256 * 1024))
    adapter = private / "adapter"
    _write_successful_adapter(adapter, model)
    executable = tmp_path / "mlx_lm.generate"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    monkeypatch.setenv("PERSONAL_VOICE_MLX_GENERATE", str(executable))
    manifest = install_renderer_manifest(
        private,
        model_path=model,
        adapter_path=adapter,
        version="candidate-v1",
        base_model="synthetic/base",
        base_revision=model.name,
    )

    with weights.open("r+b") as handle:
        handle.seek(128 * 1024)
        handle.write(b"tampered")

    doctor = renderer_doctor(manifest)
    assert doctor["model_fingerprint_match"] is False
    assert doctor["ok"] is False
    with pytest.raises(ValueError, match="failed integrity checks"):
        load_mlx_backend(manifest)


def test_load_fails_closed_after_tokenizer_asset_tampering(tmp_path: Path, monkeypatch) -> None:
    private = tmp_path / "renderer-private"
    private.mkdir(mode=0o700)
    model = private / "model-revision"
    _write_synthetic_model(model)
    tokenizer_asset = model / "special_tokens_map.json"
    tokenizer_asset.write_text(json.dumps({"eos_token": "synthetic-one"}))
    adapter = private / "adapter"
    _write_successful_adapter(adapter, model)
    executable = tmp_path / "mlx_lm.generate"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    monkeypatch.setenv("PERSONAL_VOICE_MLX_GENERATE", str(executable))
    manifest = install_renderer_manifest(
        private,
        model_path=model,
        adapter_path=adapter,
        version="candidate-v1",
        base_model="synthetic/base",
        base_revision=model.name,
    )

    tokenizer_asset.write_text(json.dumps({"eos_token": "synthetic-two"}))

    doctor = renderer_doctor(manifest)
    assert doctor["model_fingerprint_match"] is False
    assert doctor["ok"] is False


def test_model_fingerprint_frames_file_boundaries_unambiguously(
    tmp_path: Path, monkeypatch
) -> None:
    private = tmp_path / "renderer-private"
    private.mkdir(mode=0o700)
    model = private / "model-revision"
    _write_synthetic_model(model)
    first = model / "z-a"
    first.write_bytes(b"xz-b\0y")
    adapter = private / "adapter"
    _write_successful_adapter(adapter, model)
    executable = tmp_path / "mlx_lm.generate"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    monkeypatch.setenv("PERSONAL_VOICE_MLX_GENERATE", str(executable))
    manifest = install_renderer_manifest(
        private,
        model_path=model,
        adapter_path=adapter,
        version="candidate-v1",
        base_model="synthetic/base",
        base_revision=model.name,
    )

    first.write_bytes(b"x")
    (model / "z-b").write_bytes(b"y")

    doctor = renderer_doctor(manifest)
    assert doctor["model_fingerprint_match"] is False
    assert doctor["ok"] is False


@pytest.mark.parametrize(
    "artifact_kind",
    ("broken_weight_symlink", "weight_directory", "fifo", "broken_extra_symlink"),
)
def test_install_rejects_non_regular_model_artifacts(
    tmp_path: Path,
    artifact_kind: str,
) -> None:
    private = tmp_path / "renderer-private"
    private.mkdir(mode=0o700)
    model = private / "model-revision"
    if artifact_kind in {"fifo", "broken_extra_symlink"}:
        _write_synthetic_model(model)
        if artifact_kind == "fifo":
            os.mkfifo(model / "unexpected.pipe")
        else:
            (model / "tokenizer.asset").symlink_to(model / "missing-tokenizer.asset")
    else:
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps({"model_type": "synthetic", "quantization": {"bits": 4}})
        )
        weights = model / "model.safetensors"
        if artifact_kind == "broken_weight_symlink":
            weights.symlink_to(model / "missing-weights")
        else:
            weights.mkdir()
    adapter = private / "adapter"
    _write_successful_adapter(adapter, model)

    with pytest.raises(ValueError, match="artifact"):
        install_renderer_manifest(
            private,
            model_path=model,
            adapter_path=adapter,
            version=f"candidate-{artifact_kind}",
            base_model="synthetic/base",
            base_revision=model.name,
        )


def test_install_accepts_hugging_face_snapshot_file_symlinks(tmp_path: Path, monkeypatch) -> None:
    private = tmp_path / "renderer-private"
    private.mkdir(mode=0o700)
    blobs = private / "blobs"
    blobs.mkdir()
    weight_blob = blobs / "weight-blob"
    weight_blob.write_bytes(b"synthetic-model")
    model = private / "model-revision"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "synthetic", "quantization": {"bits": 4}})
    )
    (model / "model.safetensors").symlink_to(weight_blob)
    adapter = private / "adapter"
    _write_successful_adapter(adapter, model)
    executable = tmp_path / "mlx_lm.generate"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    monkeypatch.setenv("PERSONAL_VOICE_MLX_GENERATE", str(executable))

    manifest = install_renderer_manifest(
        private,
        model_path=model,
        adapter_path=adapter,
        version="candidate-v1",
        base_model="synthetic/base",
        base_revision=model.name,
    )

    assert renderer_doctor(manifest)["ok"] is True


def test_load_and_activation_fail_closed_after_adapter_tampering(
    tmp_path: Path, monkeypatch
) -> None:
    private = tmp_path / "renderer-private"
    private.mkdir(mode=0o700)
    model = private / "base-model"
    _write_synthetic_model(model)
    executable = tmp_path / "mlx_lm.generate"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    monkeypatch.setenv("PERSONAL_VOICE_MLX_GENERATE", str(executable))

    first_adapter = private / "adapter-v1"
    _write_successful_adapter(first_adapter, model, b"first")
    staged_first = install_renderer_manifest(
        private,
        model_path=model,
        adapter_path=first_adapter,
        version="synthetic-v1",
        base_model="synthetic/base",
        base_revision=model.name,
    )
    assert json.loads(staged_first.read_text())["version"] == "synthetic-v1"

    second_adapter = private / "adapter-v2"
    _write_successful_adapter(second_adapter, model, b"second")
    weights = second_adapter / "adapters.safetensors"
    staged = install_renderer_manifest(
        private,
        model_path=model,
        adapter_path=second_adapter,
        version="synthetic-v2",
        base_model="synthetic/base",
        base_revision=model.name,
    )
    weights.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="failed integrity checks"):
        load_mlx_backend(staged)
    with pytest.raises(ValueError, match="failed integrity checks"):
        activate_renderer_manifest(private, "synthetic-v2")
    assert not (private / "current.json").exists()


def test_install_rejects_private_adapter_inside_git_or_incomplete_adapter(
    tmp_path: Path,
) -> None:
    private = tmp_path / "renderer-private"
    private.mkdir(mode=0o700)
    model = private / "model"
    _write_synthetic_model(model)
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / ".git").mkdir()
    git_adapter = repository / "adapter"
    _write_successful_adapter(git_adapter, model)

    with pytest.raises(ValueError, match="outside a Git worktree"):
        install_renderer_manifest(
            private,
            model_path=model,
            adapter_path=git_adapter,
            version="synthetic-git",
            base_model="synthetic/base",
            base_revision=model.name,
        )

    incomplete = private / "incomplete-adapter"
    incomplete.mkdir()
    with pytest.raises(ValueError, match="incomplete or unsuccessful"):
        install_renderer_manifest(
            private,
            model_path=model,
            adapter_path=incomplete,
            version="synthetic-incomplete",
            base_model="synthetic/base",
            base_revision=model.name,
        )

    boolean_exit = private / "boolean-exit-adapter"
    _write_successful_adapter(boolean_exit, model)
    training_report = boolean_exit / "training-run.json"
    report = json.loads(training_report.read_text())
    report["exit_code"] = False
    training_report.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="incomplete or unsuccessful"):
        install_renderer_manifest(
            private,
            model_path=model,
            adapter_path=boolean_exit,
            version="synthetic-boolean-exit",
            base_model="synthetic/base",
            base_revision=model.name,
        )

    changed_data = private / "changed-data-adapter"
    _write_successful_adapter(changed_data, model)
    training_report = changed_data / "training-run.json"
    report = json.loads(training_report.read_text())
    report["dataset_unchanged"] = False
    training_report.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="incomplete or unsuccessful"):
        install_renderer_manifest(
            private,
            model_path=model,
            adapter_path=changed_data,
            version="synthetic-changed-data",
            base_model="synthetic/base",
            base_revision=model.name,
        )


def test_doctor_rejects_non_executable_configured_runtime(tmp_path: Path, monkeypatch) -> None:
    private = tmp_path / "renderer-private"
    private.mkdir(mode=0o700)
    model = private / "model"
    _write_synthetic_model(model)
    adapter = private / "adapter"
    _write_successful_adapter(adapter, model)
    executable = tmp_path / "not-executable"
    executable.write_text("nope")
    monkeypatch.setenv("PERSONAL_VOICE_MLX_GENERATE", str(executable))

    manifest = install_renderer_manifest(
        private,
        model_path=model,
        adapter_path=adapter,
        version="synthetic-v1",
        base_model="synthetic/base",
        base_revision=model.name,
    )

    assert renderer_doctor(manifest)["executable_available"] is False
    assert renderer_doctor(manifest)["ok"] is False


def test_renderer_doctor_does_not_treat_cwd_as_a_missing_model(tmp_path: Path) -> None:
    result = renderer_doctor(tmp_path / "missing.json")

    assert result["manifest_private"] is False
    assert result["model_exists"] is False
    assert result["adapter_private"] is False
    assert result["ok"] is False


def test_renderer_manifest_requires_an_integer_schema_version(tmp_path: Path, monkeypatch) -> None:
    private = tmp_path / "renderer-private"
    private.mkdir(mode=0o700)
    model = private / "model"
    _write_synthetic_model(model)
    adapter = private / "adapter"
    _write_successful_adapter(adapter, model)
    executable = tmp_path / "mlx_lm.generate"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    monkeypatch.setenv("PERSONAL_VOICE_MLX_GENERATE", str(executable))
    manifest = install_renderer_manifest(
        private,
        model_path=model,
        adapter_path=adapter,
        version="candidate-v1",
        base_model="synthetic/base",
        base_revision=model.name,
    )
    payload = json.loads(manifest.read_text())
    payload["schema_version"] = 2.0
    payload["candidate_sha256"] = _candidate_manifest_sha256(payload)
    manifest.write_text(json.dumps(payload))
    manifest.chmod(0o600)

    doctor = renderer_doctor(manifest)
    assert doctor["candidate_hash_match"] is True
    assert doctor["ok"] is False
    with pytest.raises(ValueError, match="integrity checks|unsupported renderer manifest"):
        load_mlx_backend(manifest)


def test_renderer_manifest_binds_the_execution_contract_with_v1_legacy_support(
    tmp_path: Path, monkeypatch
) -> None:
    private = tmp_path / "renderer-private"
    private.mkdir(mode=0o700)
    model = private / "model"
    _write_synthetic_model(model)
    adapter = private / "adapter"
    _write_successful_adapter(adapter, model)
    executable = tmp_path / "mlx_lm.generate"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    monkeypatch.setenv("PERSONAL_VOICE_MLX_GENERATE", str(executable))
    manifest = install_renderer_manifest(
        private,
        model_path=model,
        adapter_path=adapter,
        version="candidate-v1",
        base_model="synthetic/base",
        base_revision=model.name,
    )
    payload = json.loads(manifest.read_text())

    payload.pop("renderer_contract_version")
    payload["candidate_sha256"] = _candidate_manifest_sha256(payload)
    manifest.write_text(json.dumps(payload))
    manifest.chmod(0o600)
    legacy = renderer_doctor(manifest)
    assert legacy["contract_version"] == 1
    assert legacy["contract_version_match"] is True
    assert legacy["ok"] is True

    payload["renderer_contract_version"] = 2
    payload["candidate_sha256"] = _candidate_manifest_sha256(payload)
    manifest.write_text(json.dumps(payload))
    manifest.chmod(0o600)
    mismatched = renderer_doctor(manifest)
    assert mismatched["contract_version_match"] is False
    assert mismatched["ok"] is False


def test_activation_rejects_a_dataset_not_bound_to_the_candidate(
    tmp_path: Path, monkeypatch
) -> None:
    private = tmp_path / "renderer-private"
    private.mkdir(mode=0o700)
    model = private / "model"
    _write_synthetic_model(model)
    adapter = private / "adapter"
    _write_successful_adapter(adapter, model)
    executable = tmp_path / "mlx_lm.generate"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    monkeypatch.setenv("PERSONAL_VOICE_MLX_GENERATE", str(executable))
    install_renderer_manifest(
        private,
        model_path=model,
        adapter_path=adapter,
        version="candidate-v1",
        base_model="synthetic/base",
        base_revision=model.name,
    )
    data = private / "data/substituted"
    data.mkdir(mode=0o700, parents=True)
    for split, content in (("train", "{}\n"), ("valid", ""), ("test", "")):
        path = data / f"{split}.jsonl"
        path.write_text(content)
        path.chmod(0o600)

    with pytest.raises(PermissionError, match="training dataset"):
        activate_renderer_manifest(
            private,
            "candidate-v1",
            blind_pack=private / "evaluation/blind.sqlite",
            data_path=data,
        )


def test_activation_requires_candidate_bound_blind_and_semantic_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    private = tmp_path / "renderer-private"
    private.mkdir(mode=0o700)
    model = private / "model"
    _write_synthetic_model(model)
    adapter = private / "adapter"
    _write_successful_adapter(adapter, model)
    executable = tmp_path / "mlx_lm.generate"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    monkeypatch.setenv("PERSONAL_VOICE_MLX_GENERATE", str(executable))
    manifest = install_renderer_manifest(
        private,
        model_path=model,
        adapter_path=adapter,
        version="candidate-v1",
        base_model="synthetic/base",
        base_revision=model.name,
    )

    with pytest.raises(PermissionError, match="blind evaluation"):
        activate_renderer_manifest(private, "candidate-v1")

    monkeypatch.setattr(
        "personal_voice.privacy_probe.require_candidate_dataset_binding",
        lambda *_args, **_kwargs: ("bound", 100),
    )
    seen = {}

    def eligible(_pack, *, model_version, candidate_sha256, renderer_contract_version):
        assert renderer_contract_version == 1
        seen.update(version=model_version, candidate_sha256=candidate_sha256)
        return {"activation_eligible": True}

    monkeypatch.setattr("personal_voice.blind_eval.blind_eval_stats", eligible)
    with pytest.raises(PermissionError, match="privacy probe"):
        activate_renderer_manifest(
            private,
            "candidate-v1",
            blind_pack=private / "evaluation/blind.sqlite",
        )
    monkeypatch.setattr(
        "personal_voice.privacy_probe.privacy_report_eligible",
        lambda _report, **binding: bool(
            binding["model_version"] == "candidate-v1"
            and binding["adapter_sha256"]
            and binding["candidate_sha256"]
            and binding["dataset_sha256"]
            and binding["dataset_records"] == 100
        ),
    )
    with pytest.raises(PermissionError, match="runtime benchmark"):
        activate_renderer_manifest(
            private,
            "candidate-v1",
            blind_pack=private / "evaluation/blind.sqlite",
        )
    monkeypatch.setattr(
        "personal_voice.runtime_gate.runtime_report_eligible",
        lambda _report, **binding: bool(
            binding["model_version"] == "candidate-v1"
            and binding["adapter_sha256"]
            and binding["candidate_sha256"]
        ),
    )
    monkeypatch.setattr(
        "personal_voice.renderer_semantic.evaluate_semantic_gates",
        lambda _backend: {"ok": True},
    )

    def load_candidate(_manifest, *, use_adapter=True):
        backend = FakeBackend(lambda _medium, text: text)
        candidate = json.loads(Path(_manifest).read_text())
        backend.version = candidate["version"]
        backend.artifact_hash = candidate["adapter_sha256"]
        backend.candidate_hash = candidate["candidate_sha256"]
        backend.use_adapter = use_adapter
        return backend

    base_rendered = 49
    base_total = 200
    base_unavailable = 0
    base_rejected_adjustment = 0
    base_statuses_override = None
    base_closer = 100
    base_mean_delta = 0.1
    base_mediums = {"dm": 50, "email": 50, "public": 50, "text": 50}
    tuned_rendered = 180
    tuned_total = 200
    tuned_unavailable = 0
    tuned_statuses_override = None
    tuned_closer = 160
    tuned_mean_delta = 0.2
    tuned_mediums = {"dm": 50, "email": 50, "public": 50, "text": 50}

    def style_evaluation(_data, backend, **_kwargs):
        tuned = backend.use_adapter
        statuses = {
            "rendered": tuned_rendered if tuned else base_rendered,
            "rejected": (
                tuned_total - tuned_rendered - tuned_unavailable
                if tuned
                else base_total - base_rendered - base_unavailable + base_rejected_adjustment
            ),
            "unavailable": tuned_unavailable if tuned else base_unavailable,
        }
        override = tuned_statuses_override if tuned else base_statuses_override
        return {
            "closer_to_target": tuned_closer if tuned else base_closer,
            "mean_similarity_delta": tuned_mean_delta if tuned else base_mean_delta,
            "mediums": tuned_mediums if tuned else base_mediums,
            "statuses": override if override is not None else statuses,
            "total": tuned_total if tuned else base_total,
        }

    monkeypatch.setattr("personal_voice.renderer.load_mlx_inprocess_backend", load_candidate)
    monkeypatch.setattr(
        "personal_voice.renderer_eval.evaluate_renderer_dataset",
        style_evaluation,
    )
    with pytest.raises(PermissionError, match="beaten the 4-bit base model"):
        activate_renderer_manifest(
            private,
            "candidate-v1",
            blind_pack=private / "evaluation/blind.sqlite",
        )

    base_rendered = 50
    base_total = 199
    with pytest.raises(PermissionError, match="beaten the 4-bit base model"):
        activate_renderer_manifest(
            private,
            "candidate-v1",
            blind_pack=private / "evaluation/blind.sqlite",
        )

    base_total = 200
    base_mediums = {"dm": 49, "email": 51, "public": 50, "text": 50}
    with pytest.raises(PermissionError, match="beaten the 4-bit base model"):
        activate_renderer_manifest(
            private,
            "candidate-v1",
            blind_pack=private / "evaluation/blind.sqlite",
        )

    base_mediums = tuned_mediums
    tuned_total = 99
    base_total = 99
    tuned_rendered = 80
    tuned_closer = 80
    base_closer = 40
    base_mediums = {"dm": 25, "email": 24, "public": 25, "text": 25}
    tuned_mediums = base_mediums
    with pytest.raises(PermissionError, match="beaten the 4-bit base model"):
        activate_renderer_manifest(
            private,
            "candidate-v1",
            blind_pack=private / "evaluation/blind.sqlite",
        )

    tuned_total = 200
    base_total = 200
    tuned_rendered = 159
    tuned_closer = 160
    base_closer = 100
    base_mediums = {"dm": 50, "email": 50, "public": 50, "text": 50}
    tuned_mediums = base_mediums
    with pytest.raises(PermissionError, match="beaten the 4-bit base model"):
        activate_renderer_manifest(
            private,
            "candidate-v1",
            blind_pack=private / "evaluation/blind.sqlite",
        )

    tuned_rendered = 180
    base_unavailable = 1
    with pytest.raises(PermissionError, match="beaten the 4-bit base model"):
        activate_renderer_manifest(
            private,
            "candidate-v1",
            blind_pack=private / "evaluation/blind.sqlite",
        )

    base_unavailable = 0
    tuned_unavailable = 1
    with pytest.raises(PermissionError, match="beaten the 4-bit base model"):
        activate_renderer_manifest(
            private,
            "candidate-v1",
            blind_pack=private / "evaluation/blind.sqlite",
        )

    tuned_unavailable = 0
    base_rendered = 180
    with pytest.raises(PermissionError, match="beaten the 4-bit base model"):
        activate_renderer_manifest(
            private,
            "candidate-v1",
            blind_pack=private / "evaluation/blind.sqlite",
        )

    base_rendered = 50
    tuned_mean_delta = 0.0
    base_mean_delta = -0.1
    with pytest.raises(PermissionError, match="beaten the 4-bit base model"):
        activate_renderer_manifest(
            private,
            "candidate-v1",
            blind_pack=private / "evaluation/blind.sqlite",
        )

    tuned_mean_delta = 0.1
    base_mean_delta = 0.1
    with pytest.raises(PermissionError, match="beaten the 4-bit base model"):
        activate_renderer_manifest(
            private,
            "candidate-v1",
            blind_pack=private / "evaluation/blind.sqlite",
        )

    tuned_mean_delta = 0.2
    tuned_closer = 100
    with pytest.raises(PermissionError, match="beaten the 4-bit base model"):
        activate_renderer_manifest(
            private,
            "candidate-v1",
            blind_pack=private / "evaluation/blind.sqlite",
        )

    tuned_closer = 160
    base_rejected_adjustment = 1
    with pytest.raises(PermissionError, match="beaten the 4-bit base model"):
        activate_renderer_manifest(
            private,
            "candidate-v1",
            blind_pack=private / "evaluation/blind.sqlite",
        )

    base_rejected_adjustment = 0
    valid_base_statuses = {"rendered": 50, "rejected": 150, "unavailable": 0}
    for invalid in (
        {"rendered": 50, "rejected": 151, "unavailable": -1},
        {**valid_base_statuses, "unexpected": 0},
    ):
        base_statuses_override = invalid
        with pytest.raises(PermissionError, match="beaten the 4-bit base model"):
            activate_renderer_manifest(
                private,
                "candidate-v1",
                blind_pack=private / "evaluation/blind.sqlite",
            )
    base_statuses_override = None
    for invalid in (
        {"rendered": 180, "rejected": 20, "unavailable": False},
        {"rendered": 180, "rejected": 20},
        {"rendered": 180, "rejected": 21, "unavailable": 0},
    ):
        tuned_statuses_override = invalid
        with pytest.raises(PermissionError, match="beaten the 4-bit base model"):
            activate_renderer_manifest(
                private,
                "candidate-v1",
                blind_pack=private / "evaluation/blind.sqlite",
            )
    tuned_statuses_override = None

    valid_mediums = {"dm": 50, "email": 50, "public": 50, "text": 50}
    tuned_mediums = valid_mediums
    for invalid in (
        {"dm": 50, "email": 50, "public": 100},
        {"dm": 51, "email": 50, "public": 100, "text": -1},
        {"dm": 50, "email": 50, "public": 50, "text": 51},
    ):
        base_mediums = invalid
        with pytest.raises(PermissionError, match="beaten the 4-bit base model"):
            activate_renderer_manifest(
                private,
                "candidate-v1",
                blind_pack=private / "evaluation/blind.sqlite",
            )
    base_mediums = valid_mediums
    for invalid in (
        {"dm": 50, "email": 50, "public": 100, "text": False},
        {**valid_mediums, "unexpected": 0},
        {"dm": 50, "email": 50, "public": 50, "text": 51},
    ):
        tuned_mediums = invalid
        with pytest.raises(PermissionError, match="beaten the 4-bit base model"):
            activate_renderer_manifest(
                private,
                "candidate-v1",
                blind_pack=private / "evaluation/blind.sqlite",
            )
    tuned_mediums = valid_mediums
    current = activate_renderer_manifest(
        private,
        "candidate-v1",
        blind_pack=private / "evaluation/blind.sqlite",
    )

    payload = json.loads(manifest.read_text())
    assert seen == {
        "candidate_sha256": payload["candidate_sha256"],
        "version": "candidate-v1",
    }
    assert current == private / "current.json"
    assert json.loads(current.read_text())["version"] == "candidate-v1"

    current.unlink()
    original_manifest = manifest.read_bytes()
    swapped = False

    def swap_manifest_during_load(snapshot, *, use_adapter=True):
        nonlocal swapped
        backend = load_candidate(snapshot, use_adapter=use_adapter)
        if not swapped:
            altered = json.loads(original_manifest)
            altered["version"] = "candidate-v2"
            altered["candidate_sha256"] = _candidate_manifest_sha256(altered)
            manifest.write_text(json.dumps(altered))
            manifest.chmod(0o600)
            swapped = True
        return backend

    monkeypatch.setattr(
        "personal_voice.renderer.load_mlx_inprocess_backend",
        swap_manifest_during_load,
    )
    with pytest.raises(PermissionError, match="changed during activation"):
        activate_renderer_manifest(
            private,
            "candidate-v1",
            blind_pack=private / "evaluation/blind.sqlite",
        )
    assert not current.exists()
    manifest.write_bytes(original_manifest)
    manifest.chmod(0o600)


def test_slotize_training_pair_does_not_rewrite_digits_inside_placeholders() -> None:
    canonical = "The meeting is at 3:30 and the remaining balance is 0."
    target = "yeah the meeting is at 3:30 and the remaining balance is 0"

    slotized_canonical, slotized_target = slotize_training_pair(canonical, target)

    assert slotized_canonical == (
        "The meeting is at [[PV_LITERAL_0]] and the remaining balance is [[PV_LITERAL_1]]."
    )
    assert slotized_target == (
        "yeah the meeting is at [[PV_LITERAL_0]] and the remaining balance is [[PV_LITERAL_1]]"
    )


def test_slotize_training_pair_rejects_reordered_literal_associations() -> None:
    with pytest.raises(ValueError, match="literal order"):
        slotize_training_pair(
            "The first review is at 3:30 and the second is at 5:00.",
            "the second review is at 5:00 and the first is at 3:30",
        )
