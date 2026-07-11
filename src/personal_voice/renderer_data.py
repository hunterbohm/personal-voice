"""Prepare private, leak-resistant data for the local style renderer."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import urllib.request
from collections import defaultdict
from collections.abc import Callable, Mapping
from difflib import SequenceMatcher
from pathlib import Path

from personal_voice.local_only import (
    inside_git_worktree,
    open_loopback_http_request,
    require_loopback_http_endpoint,
)
from personal_voice.renderer import (
    RENDER_SYSTEM_PROMPT,
    format_renderer_target,
    literal_counter,
    literal_sequence,
    render_user_prompt,
    slotize_training_pair,
)
from personal_voice.training_data import ChatTokenizer, training_record_lengths

_MEDIA = ("text", "dm", "email", "public")
_SPLITS = ("train", "valid", "test")
_WORD = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)
_DIRECT_IDENTIFIER = re.compile(r"<\|(?:url|email_address|phone)\|>")
_MENTION = re.compile(r"(?<!\w)@[A-Za-z0-9_]{1,30}")
_NUMBER = re.compile(r"(?<!\w)\d+(?:[.,:/\-]\d+)*(?!\w)")
_SIGNATURE_TOKEN = re.compile(r"<(?:slot|mention|num)>|[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)
_MAX_REPLY_WORDS = 256
_MAX_SHORT_FRACTION = 0.30

Neutralizer = Callable[[str, str, int], str]
ProgressCallback = Callable[[dict[str, int]], None]


class OllamaNeutralizer:
    """Create canonical drafts with a local Ollama model."""

    def __init__(
        self,
        model: str = "qwen3.5:9b",
        endpoint: str = "http://127.0.0.1:11434/api/generate",
        timeout_seconds: int = 180,
    ) -> None:
        self.model = model
        self.endpoint = require_loopback_http_endpoint(endpoint, path="/api/generate")
        self.timeout_seconds = timeout_seconds

    def __call__(self, medium: str, target: str, attempt: int) -> str:
        return self.neutralize_batch([(medium, target, attempt)])[0]

    def neutralize_batch(self, items: list[tuple[str, str, int]]) -> list[str]:
        if not items:
            return []
        retry_note = (
            " A previous attempt failed a deterministic preservation check; preserve every "
            "literal exactly."
            if any(attempt > 1 for _, _, attempt in items)
            else ""
        )
        messages = [
            {"index": index, "medium": medium, "message": target}
            for index, (medium, target, _attempt) in enumerate(items)
        ]
        prompt = (
            "Rewrite each message in plain, neutral, standard wording for use as canonical "
            "style-transfer training data. Preserve every fact, name, number, date, time, "
            "amount, URL, handle, question, negation, uncertainty, sentiment, and commitment. "
            "Do not add, infer, omit, explain, or answer anything. Remove distinctive slang, "
            "filler, unusual casing, and personal punctuation. Return one draft for every "
            f"input index using the required JSON schema.{retry_note}\n\nItems:\n"
            + json.dumps(messages, ensure_ascii=False)
        )
        schema = {
            "type": "object",
            "properties": {
                "drafts": {
                    "type": "array",
                    "minItems": len(items),
                    "maxItems": len(items),
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer"},
                            "draft": {"type": "string"},
                        },
                        "required": ["index", "draft"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["drafts"],
            "additionalProperties": False,
        }
        total_words = sum(len(_WORD.findall(target)) for _, target, _ in items)
        body = json.dumps(
            {
                "format": schema,
                "keep_alive": "30m",
                "model": self.model,
                "options": {
                    "num_ctx": 4096,
                    "num_predict": min(2048, max(128, total_words * 4 + 64)),
                    "seed": 0,
                    "temperature": 0,
                },
                "prompt": prompt,
                "stream": False,
                "think": False,
            },
            ensure_ascii=False,
        ).encode()
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with open_loopback_http_request(
            request,
            path="/api/generate",
            timeout=self.timeout_seconds,
        ) as response:
            payload = json.load(response)
        result = json.loads(str(payload.get("response") or ""))
        drafts = result.get("drafts") if isinstance(result, dict) else None
        if not isinstance(drafts, list):
            raise ValueError("neutralizer returned an invalid response")
        indexed: dict[int, str] = {}
        for item in drafts:
            if not isinstance(item, dict) or not isinstance(item.get("draft"), str):
                raise ValueError("neutralizer returned an invalid response")
            try:
                index = int(item.get("index"))
            except (TypeError, ValueError) as error:
                raise ValueError("neutralizer returned an invalid response") from error
            indexed[index] = str(item["draft"]).strip()
        if set(indexed) != set(range(len(items))):
            raise ValueError("neutralizer returned an invalid response")
        return [indexed[index] for index in range(len(items))]


class MlxNeutralizer:
    """Create canonical drafts with one locally loaded MLX teacher model."""

    def __init__(self, model_path: str | Path) -> None:
        import importlib

        import mlx_lm
        from mlx_lm.sample_utils import make_sampler

        generate_module = importlib.import_module("mlx_lm.generate")
        self._batch_generate = generate_module.batch_generate
        self._sampler = make_sampler(temp=0)
        self._model, self._tokenizer = mlx_lm.load(str(Path(model_path).expanduser()))

    def neutralize_batch(self, items: list[tuple[str, str, int]]) -> list[str]:
        system = (
            "Rewrite the complete message in plain, neutral, standard wording for canonical "
            "style-transfer training data. Preserve every fact, name, number, date, time, "
            "amount, URL, handle, question, negation, uncertainty, sentiment, and commitment. "
            "Do not add, infer, omit, explain, or answer anything. Remove distinctive slang, "
            "filler, unusual casing, and personal punctuation. Return only the neutral rewrite."
        )
        prompts = []
        max_tokens = 64
        for medium, target, attempt in items:
            retry_note = (
                "\nA previous attempt failed a literal or structure check. Preserve every "
                "literal exactly."
                if attempt > 1
                else ""
            )
            user = f"Medium: {medium}\nMessage:\n{target}{retry_note}"
            prompts.append(
                self._tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    add_generation_prompt=True,
                )
            )
            max_tokens = max(
                max_tokens,
                min(768, max(64, len(_WORD.findall(target)) * 4 + 32)),
            )
        response = self._batch_generate(
            self._model,
            self._tokenizer,
            prompts,
            max_tokens=max_tokens,
            sampler=self._sampler,
            verbose=False,
        )
        return [str(text).strip() for text in response.texts]


def prepare_renderer_workspace(
    corpus_path: str | Path,
    workspace_path: str | Path,
    *,
    per_medium: Mapping[str, int],
    conversation_cap: int = 250,
    identity_fraction: float = 0.0,
    seed: str = "personal-renderer-v1",
) -> dict:
    """Build a private renderer workspace from the example corpus."""

    corpus = Path(corpus_path).expanduser()
    workspace = Path(workspace_path).expanduser()
    if not corpus.is_file():
        raise FileNotFoundError(corpus)
    if conversation_cap <= 0:
        raise ValueError("conversation_cap must be positive")
    if not 0.0 <= identity_fraction <= 1.0:
        raise ValueError("identity_fraction must be between zero and one")
    limits = {medium: int(per_medium.get(medium, 0)) for medium in _MEDIA}
    if any(limit < 0 for limit in limits.values()):
        raise ValueError("per-medium limits cannot be negative")

    candidates = _load_candidates(corpus)
    capped = _cap_conversations(candidates, conversation_cap, seed)
    split_rows = _assign_splits(capped, seed)
    leak_free = _purge_cross_split_replies(split_rows)
    deduped = _cap_duplicate_replies(leak_free, seed)
    selected = _select_balanced(deduped, limits, seed)
    _write_workspace(
        workspace,
        selected,
        seed,
        conversation_cap,
        limits,
        identity_fraction,
    )
    return renderer_workspace_stats(workspace)


def renderer_workspace_stats(workspace_path: str | Path) -> dict:
    """Return aggregate-only workspace counts and leakage checks."""

    workspace = Path(workspace_path).expanduser()
    connection = sqlite3.connect(f"file:{workspace}?mode=ro", uri=True)
    try:
        total = int(connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0])
        short = int(
            connection.execute("SELECT COUNT(*) FROM pairs WHERE word_count <= 5").fetchone()[0]
        )
        mediums: dict[str, dict[str, int]] = {}
        for medium in _MEDIA:
            counts = dict(
                connection.execute(
                    "SELECT split, COUNT(*) FROM pairs WHERE medium = ? GROUP BY split",
                    (medium,),
                ).fetchall()
            )
            medium_total = sum(int(value) for value in counts.values())
            if medium_total:
                mediums[medium] = {
                    **{split: int(counts.get(split, 0)) for split in _SPLITS},
                    "total": medium_total,
                }
        conversation_leaks = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT medium, conversation_id
                    FROM pairs
                    GROUP BY medium, conversation_id
                    HAVING COUNT(DISTINCT split) > 1
                )
                """
            ).fetchone()[0]
        )
        reply_leaks = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT reply_signature
                    FROM pairs
                    GROUP BY reply_signature
                    HAVING COUNT(DISTINCT split) > 1
                )
                """
            ).fetchone()[0]
        )
        neutralization = {
            status: int(count)
            for status, count in connection.execute(
                "SELECT neutral_status, COUNT(*) FROM pairs GROUP BY neutral_status"
            ).fetchall()
        }
        canonical_splits: dict[str, set[str]] = defaultdict(set)
        for split, neutral_text in connection.execute(
            "SELECT split, neutral_text FROM pairs WHERE neutral_status = 'ready'"
        ):
            canonical_splits[_reply_signature(str(neutral_text))].add(str(split))
        canonical_leaks = sum(len(splits) > 1 for splits in canonical_splits.values())
    finally:
        connection.close()
    return {
        "canonical_leaks": canonical_leaks,
        "conversation_leaks": conversation_leaks,
        "mediums": mediums,
        "neutralization": {
            status: neutralization.get(status, 0)
            for status in ("excluded", "failed", "pending", "ready")
        },
        "reply_leaks": reply_leaks,
        "short_fraction": round(short / total, 6) if total else 0.0,
        "total": total,
    }


def finalize_renderer_workspace(workspace_path: str | Path) -> dict:
    """Exclude canonical drafts that collapse across held-out split boundaries."""

    workspace = Path(workspace_path).expanduser()
    connection = sqlite3.connect(workspace)
    try:
        pending = int(
            connection.execute(
                "SELECT COUNT(*) FROM pairs WHERE neutral_status = 'pending'"
            ).fetchone()[0]
        )
        if pending:
            raise RuntimeError("renderer neutralization is incomplete")
        grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for record_id, split, neutral_text in connection.execute(
            """
            SELECT record_id, split, neutral_text
            FROM pairs WHERE neutral_status = 'ready'
            """
        ):
            grouped[_reply_signature(str(neutral_text))].append((str(record_id), str(split)))
        split_priority = {"train": 0, "valid": 1, "test": 2}
        excluded_ids: list[tuple[str]] = []
        for rows in grouped.values():
            splits = {split for _record_id, split in rows}
            if len(splits) <= 1:
                continue
            retained_split = min(splits, key=split_priority.__getitem__)
            excluded_ids.extend(
                (record_id,) for record_id, split in rows if split != retained_split
            )
        connection.executemany(
            """
            UPDATE pairs
            SET neutral_text = NULL, neutral_status = 'excluded',
                neutral_error = 'canonical_split_leak'
            WHERE record_id = ?
            """,
            excluded_ids,
        )
        connection.commit()
    finally:
        connection.close()
    return renderer_workspace_stats(workspace)


def neutralize_renderer_workspace(
    workspace_path: str | Path,
    neutralizer: Neutralizer,
    *,
    limit: int | None = None,
    batch_size: int = 8,
    max_attempts: int = 2,
    progress: ProgressCallback | None = None,
    progress_every: int = 100,
    retry_failed: bool = False,
) -> dict:
    """Fill pending canonical drafts without emitting private text."""

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")
    workspace = Path(workspace_path).expanduser()
    connection = sqlite3.connect(workspace)
    try:
        statuses = ("pending", "failed") if retry_failed else ("pending",)
        placeholders = ",".join("?" for _ in statuses)
        query = (
            "SELECT record_id, medium, target_text, neutral_attempts "
            f"FROM pairs WHERE neutral_status IN ({placeholders}) ORDER BY record_id"
        )
        parameters: list[object] = list(statuses)
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)
        rows = connection.execute(query, parameters).fetchall()
        processed = 0
        ready_count = 0
        failed_count = 0
        for offset in range(0, len(rows), batch_size):
            batch = [
                {
                    "attempts_used": 0,
                    "error_code": "backend_error",
                    "medium": str(medium),
                    "previous_attempts": int(previous_attempts),
                    "record_id": str(record_id),
                    "target": str(target),
                }
                for record_id, medium, target, previous_attempts in rows[
                    offset : offset + batch_size
                ]
            ]
            active = batch
            for attempt in range(1, max_attempts + 1):
                items = [(row["medium"], row["target"], attempt) for row in active]
                try:
                    outputs = _neutralize_batch(neutralizer, items)
                    if len(outputs) != len(active):
                        raise ValueError("neutralizer returned the wrong batch size")
                except Exception:
                    outputs = [None] * len(active)
                remaining = []
                for row, output in zip(active, outputs, strict=True):
                    row["attempts_used"] += 1
                    if output is None:
                        row["error_code"] = "backend_error"
                        remaining.append(row)
                        continue
                    neutral = str(output).strip()
                    row["error_code"] = _neutralization_error(row["target"], neutral)
                    if row["error_code"]:
                        remaining.append(row)
                        continue
                    connection.execute(
                        """
                        UPDATE pairs
                        SET neutral_text = ?, neutral_status = 'ready',
                            neutral_attempts = ?, neutral_error = NULL
                        WHERE record_id = ?
                        """,
                        (
                            neutral,
                            row["previous_attempts"] + row["attempts_used"],
                            row["record_id"],
                        ),
                    )
                    ready_count += 1
                active = remaining
                if not active:
                    break
            for row in active:
                connection.execute(
                    """
                    UPDATE pairs
                    SET neutral_text = NULL, neutral_status = 'failed',
                        neutral_attempts = ?, neutral_error = ?
                    WHERE record_id = ?
                    """,
                    (
                        row["previous_attempts"] + row["attempts_used"],
                        row["error_code"],
                        row["record_id"],
                    ),
                )
                failed_count += 1
            connection.commit()
            processed += len(batch)
            if progress is not None and processed % progress_every == 0:
                progress({"failed": failed_count, "processed": processed, "ready": ready_count})
        if progress is not None and processed % progress_every:
            progress({"failed": failed_count, "processed": processed, "ready": ready_count})
    finally:
        connection.close()
    return renderer_workspace_stats(workspace)


def _neutralize_batch(
    neutralizer: Neutralizer,
    items: list[tuple[str, str, int]],
) -> list[str]:
    batch_method = getattr(neutralizer, "neutralize_batch", None)
    if callable(batch_method):
        return [str(value) for value in batch_method(items)]
    return [str(neutralizer(medium, target, attempt)) for medium, target, attempt in items]


def export_renderer_dataset(
    workspace_path: str | Path,
    output_directory: str | Path,
    *,
    tokenizer: ChatTokenizer | None = None,
    max_seq_length: int = 512,
    min_supervised_tokens: int = 4,
    require_semantic_audit: bool = False,
) -> dict:
    """Export ready pairs in MLX-LM chat format without identifiers."""

    workspace = Path(workspace_path).expanduser()
    output = Path(output_directory).expanduser()
    finalize_renderer_workspace(workspace)
    if require_semantic_audit:
        from personal_voice.renderer_audit import require_complete_semantic_audit

        require_complete_semantic_audit(workspace)
    _ensure_private_directory(output)
    connection = sqlite3.connect(f"file:{workspace}?mode=ro", uri=True)
    counts: dict[str, int] = {}
    excluded_for_tokens: dict[str, int] = {}
    try:
        for split in _SPLITS:
            rows = connection.execute(
                """
                SELECT medium, neutral_text, target_text
                FROM pairs
                WHERE split = ? AND neutral_status = 'ready'
                ORDER BY record_id
                """,
                (split,),
            ).fetchall()
            destination = output / f"{split}.jsonl"
            if not rows:
                destination.unlink(missing_ok=True)
                counts[split] = 0
                excluded_for_tokens[split] = 0
                continue
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{split}.", suffix=".tmp", dir=output
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            temporary.chmod(0o600)
            exported = 0
            excluded = 0
            try:
                with temporary.open("w", encoding="utf-8") as handle:
                    for medium, neutral, target in rows:
                        slotized_neutral, slotized_target = slotize_training_pair(
                            str(neutral), str(target)
                        )
                        record = {
                            "messages": [
                                {"role": "system", "content": RENDER_SYSTEM_PROMPT},
                                {
                                    "role": "user",
                                    "content": render_user_prompt(str(medium), slotized_neutral),
                                },
                                {
                                    "role": "assistant",
                                    "content": format_renderer_target(slotized_target),
                                },
                            ]
                        }
                        if tokenizer is not None:
                            total_tokens, supervised_tokens = training_record_lengths(
                                record, tokenizer
                            )
                            if (
                                total_tokens > max_seq_length
                                or supervised_tokens < min_supervised_tokens
                            ):
                                excluded += 1
                                continue
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        exported += 1
                if exported:
                    os.replace(temporary, destination)
                    destination.chmod(0o600)
                else:
                    temporary.unlink(missing_ok=True)
                    destination.unlink(missing_ok=True)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
            counts[split] = exported
            excluded_for_tokens[split] = excluded
    finally:
        connection.close()
    return {
        "excluded_for_tokens": excluded_for_tokens,
        "splits": counts,
        "total": sum(counts.values()),
    }


def _load_candidates(corpus: Path) -> list[dict]:
    connection = sqlite3.connect(f"file:{corpus}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT example_id, medium, conversation_id, timestamp, reply_text
            FROM examples
            WHERE medium IN ('text', 'dm', 'email', 'public')
            """
        ).fetchall()
    finally:
        connection.close()

    candidates: list[dict] = []
    for example_id, medium, conversation_id, timestamp, reply_text in rows:
        text = str(reply_text or "").strip()
        word_count = len(_WORD.findall(text))
        if not text or word_count > _MAX_REPLY_WORDS:
            continue
        candidates.append(
            {
                "record_id": _digest("record", example_id),
                "medium": str(medium),
                "conversation_id": str(conversation_id),
                "timestamp": int(timestamp),
                "target_text": text,
                "reply_signature": _reply_signature(text),
                "word_count": word_count,
            }
        )
    return candidates


def _cap_conversations(rows: list[dict], cap: int, seed: str) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["medium"], row["conversation_id"])].append(row)
    selected: list[dict] = []
    for group in grouped.values():
        ranked = sorted(group, key=lambda row: _rank(seed, "conversation", row["record_id"]))
        selected.extend(ranked[:cap])
    return selected


def _assign_splits(rows: list[dict], seed: str) -> list[dict]:
    assigned: list[dict] = []
    for row in rows:
        bucket = int(_digest(seed, "split", row["medium"], row["conversation_id"])[:8], 16) % 100
        split = "train" if bucket < 90 else "valid" if bucket < 95 else "test"
        assigned.append({**row, "split": split})
    return assigned


def _purge_cross_split_replies(rows: list[dict]) -> list[dict]:
    splits_by_signature: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        splits_by_signature[row["reply_signature"]].add(row["split"])
    priority = {"train": 0, "valid": 1, "test": 2}
    allowed = {
        signature: min(splits, key=priority.__getitem__)
        for signature, splits in splits_by_signature.items()
    }
    return [row for row in rows if row["split"] == allowed[row["reply_signature"]]]


def _cap_duplicate_replies(rows: list[dict], seed: str) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["medium"], row["split"], row["reply_signature"])].append(row)
    selected: list[dict] = []
    for group in grouped.values():
        cap = 10 if group[0]["word_count"] <= 5 else 3
        ranked = sorted(group, key=lambda row: _rank(seed, "duplicate", row["record_id"]))
        selected.extend(ranked[:cap])
    return selected


def _select_balanced(rows: list[dict], limits: Mapping[str, int], seed: str) -> list[dict]:
    selected: list[dict] = []
    for medium in _MEDIA:
        medium_rows = [row for row in rows if row["medium"] == medium]
        limit = min(limits[medium], len(medium_rows))
        if not limit:
            continue
        evaluation_limit = max(1, round(limit * 0.05)) if limit >= 20 else 0
        split_limits = {
            "train": limit - evaluation_limit * 2,
            "valid": evaluation_limit,
            "test": evaluation_limit,
        }
        chosen: list[dict] = []
        for split in _SPLITS:
            pool = [row for row in medium_rows if row["split"] == split]
            chosen.extend(
                _select_with_short_cap(
                    pool,
                    min(split_limits[split], len(pool)),
                    seed,
                    f"{medium}:{split}",
                )
            )
        if len(chosen) < limit:
            chosen_ids = {row["record_id"] for row in chosen}
            remaining = [row for row in medium_rows if row["record_id"] not in chosen_ids]
            chosen.extend(
                _select_with_short_cap(
                    remaining,
                    limit - len(chosen),
                    seed,
                    f"{medium}:fill",
                    existing_short=sum(row["word_count"] <= 5 for row in chosen),
                    total_limit=limit,
                )
            )
        selected.extend(chosen)
    return selected


def _select_with_short_cap(
    rows: list[dict],
    limit: int,
    seed: str,
    namespace: str,
    *,
    existing_short: int = 0,
    total_limit: int | None = None,
) -> list[dict]:
    if limit <= 0:
        return []
    final_limit = total_limit if total_limit is not None else limit
    short_target = max(0, int(final_limit * _MAX_SHORT_FRACTION) - existing_short)
    short = sorted(
        (row for row in rows if row["word_count"] <= 5),
        key=lambda row: _rank(seed, namespace, row["record_id"]),
    )
    long = sorted(
        (row for row in rows if row["word_count"] > 5),
        key=lambda row: _rank(seed, namespace, row["record_id"]),
    )
    chosen = short[: min(limit, short_target)]
    remaining = limit - len(chosen)
    chosen.extend(long[:remaining])
    return chosen


def _write_workspace(
    workspace: Path,
    rows: list[dict],
    seed: str,
    conversation_cap: int,
    limits: Mapping[str, int],
    identity_fraction: float,
) -> None:
    _ensure_private_parent(workspace)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{workspace.name}.", suffix=".tmp", dir=workspace.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.chmod(0o600)
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(temporary)
        connection.executescript(
            """
            CREATE TABLE metadata (
                schema_version INTEGER NOT NULL,
                seed TEXT NOT NULL,
                conversation_cap INTEGER NOT NULL,
                limits_json TEXT NOT NULL,
                identity_fraction REAL NOT NULL
            );
            CREATE TABLE pairs (
                record_id TEXT PRIMARY KEY,
                medium TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                split TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                target_text TEXT NOT NULL,
                reply_signature TEXT NOT NULL,
                word_count INTEGER NOT NULL,
                neutral_text TEXT,
                neutral_status TEXT NOT NULL DEFAULT 'pending',
                neutral_attempts INTEGER NOT NULL DEFAULT 0,
                neutral_error TEXT
            );
            CREATE INDEX pairs_split_medium ON pairs(split, medium);
            """
        )
        connection.execute(
            "INSERT INTO metadata VALUES (1, ?, ?, ?, ?)",
            (
                seed,
                conversation_cap,
                json.dumps(dict(limits), sort_keys=True),
                identity_fraction,
            ),
        )
        connection.executemany(
            """
            INSERT INTO pairs(
                record_id, medium, conversation_id, split, timestamp,
                target_text, reply_signature, word_count, neutral_text, neutral_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row["record_id"],
                    row["medium"],
                    row["conversation_id"],
                    row["split"],
                    row["timestamp"],
                    row["target_text"],
                    row["reply_signature"],
                    row["word_count"],
                    (
                        row["target_text"]
                        if _identity_selected(row["record_id"], seed, identity_fraction)
                        else None
                    ),
                    (
                        "ready"
                        if _identity_selected(row["record_id"], seed, identity_fraction)
                        else "pending"
                    ),
                )
                for row in rows
            ],
        )
        connection.commit()
        connection.close()
        connection = None
        os.replace(temporary, workspace)
        workspace.chmod(0o600)
    except Exception:
        if connection is not None:
            connection.close()
        temporary.unlink(missing_ok=True)
        raise


def _ensure_private_parent(path: Path) -> None:
    if inside_git_worktree(path):
        raise ValueError("renderer data must live outside a Git worktree")
    parent_existed = path.parent.exists()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if parent_existed and path.parent.stat().st_mode & 0o077:
        raise PermissionError(f"renderer data requires a private directory (0700): {path.parent}")
    if not parent_existed:
        path.parent.chmod(0o700)


def _ensure_private_directory(path: Path) -> None:
    if inside_git_worktree(path):
        raise ValueError("renderer data must live outside a Git worktree")
    existed = path.exists()
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if existed and path.stat().st_mode & 0o077:
        raise PermissionError(f"renderer data requires a private directory (0700): {path}")
    if not existed:
        path.chmod(0o700)


def _reply_signature(text: str) -> str:
    value = " ".join(text.casefold().split())
    value = _DIRECT_IDENTIFIER.sub("<slot>", value)
    value = _MENTION.sub("<mention>", value)
    value = _NUMBER.sub("<num>", value)
    return " ".join(_SIGNATURE_TOKEN.findall(value))


def _neutralization_error(target: str, neutral: str) -> str:
    if not neutral:
        return "empty"
    if _protected_literals(target) != _protected_literals(neutral):
        return "literal_mismatch"
    if literal_sequence(target) != literal_sequence(neutral):
        return "literal_order"
    target_words = len(_WORD.findall(target))
    neutral_words = len(_WORD.findall(neutral))
    if target_words >= 5 and not 0.4 <= neutral_words / max(1, target_words) <= 2.5:
        return "length_ratio"
    if target_words > 8:
        similarity = SequenceMatcher(
            None,
            " ".join(target.casefold().split()),
            " ".join(neutral.casefold().split()),
        ).ratio()
        if similarity > 0.97:
            return "unchanged"
    return ""


def _protected_literals(text: str) -> object:
    return literal_counter(text)


def _digest(*parts: object) -> str:
    material = "\0".join(str(part) for part in parts).encode()
    return hashlib.sha256(material).hexdigest()


def _rank(*parts: object) -> str:
    return _digest(*parts)


def _identity_selected(record_id: str, seed: str, fraction: float) -> bool:
    if fraction <= 0:
        return False
    bucket = int(_digest(seed, "identity", record_id)[:8], 16) / 0xFFFFFFFF
    return bucket < fraction
