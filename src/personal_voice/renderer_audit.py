"""Local semantic auditing for neutral-draft training pairs."""

from __future__ import annotations

import json
import re
import sqlite3
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from personal_voice.local_only import open_loopback_http_request, require_loopback_http_endpoint

_CATEGORIES = {
    "addition",
    "commitment",
    "deadline",
    "equivalent",
    "fact",
    "negation",
    "omission",
    "other",
    "permission",
    "question",
    "uncertainty",
}
_SEMANTIC_PATTERNS = {
    "approval": re.compile(r"\b(?:approve|approved|approval|confirm|confirmed)\b", re.I),
    "future_commitment": re.compile(
        r"\b(?:will|shall|promise|guarantee|commit(?:ted)?)\b|"
        r"\b(?:i['’]?ll|we['’]?ll|going to)\b",
        re.I,
    ),
    "negation": re.compile(
        r"\b(?:no|not|never|cannot|without|don['’]?t|doesn['’]?t|didn['’]?t|"
        r"won['’]?t|can['’]?t|isn['’]?t|aren['’]?t|wasn['’]?t|weren['’]?t|"
        r"shouldn['’]?t|wouldn['’]?t|couldn['’]?t)\b",
        re.I,
    ),
    "obligation": re.compile(r"\b(?:must|should|need to|have to|required)\b", re.I),
    "question": re.compile(
        r"\?|\b(?:who|what|when|where|why|how|can you|could you|would you|do you|"
        r"did you|are you|is it|should we)\b",
        re.I,
    ),
    "uncertainty": re.compile(
        r"\b(?:maybe|might|perhaps|possibly|probably|unsure)\b|"
        r"\b(?:i think|i guess|not sure|i don['’]?t know|idk)\b",
        re.I,
    ),
}


def _judge_output_budget(item_count: int) -> int:
    """Reserve enough tokens for the largest schema-valid verdict payload."""

    longest_category = max(_CATEGORIES, key=len)
    worst_case = json.dumps({"verdicts": [longest_category] * item_count})
    return max(256, len(worst_case))


class SemanticJudge(Protocol):
    """Private local batch judge boundary."""

    def judge_batch(self, items: list[tuple[str, str]]) -> list[dict[str, object]]: ...


class OllamaSemanticJudge:
    """Use a local Ollama model to audit proposition equivalence."""

    def __init__(
        self,
        model: str = "qwen3.5:9b",
        endpoint: str = "http://127.0.0.1:11434/api/generate",
        timeout_seconds: int = 240,
    ) -> None:
        self.model = model
        self.endpoint = require_loopback_http_endpoint(endpoint, path="/api/generate")
        self.timeout_seconds = timeout_seconds

    def judge_batch(self, items: list[tuple[str, str]]) -> list[dict[str, object]]:
        payload_items = [
            {"canonical": canonical, "index": index, "target": target}
            for index, (target, canonical) in enumerate(items)
        ]
        prompt = (
            "Audit whether each canonical rewrite has exactly the same propositions as the "
            "target message. Ignore casing, punctuation, slang, warmth, and wording. Reject any "
            "added, removed, strengthened, or weakened fact, question, negation, uncertainty, "
            "permission, deadline, promise, or commitment. Return only the required JSON.\n\n"
            + json.dumps(payload_items, ensure_ascii=False)
        )
        schema = {
            "type": "object",
            "properties": {
                "verdicts": {
                    "type": "array",
                    "minItems": len(items),
                    "maxItems": len(items),
                    "items": {"enum": sorted(_CATEGORIES)},
                }
            },
            "required": ["verdicts"],
            "additionalProperties": False,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(
                {
                    "format": schema,
                    "keep_alive": "30m",
                    "model": self.model,
                    "options": {
                        "num_ctx": 8192,
                        "num_predict": _judge_output_budget(len(items)),
                        "seed": 0,
                        "temperature": 0,
                    },
                    "prompt": prompt,
                    "stream": False,
                    "think": False,
                },
                ensure_ascii=False,
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with open_loopback_http_request(
            request,
            path="/api/generate",
            timeout=self.timeout_seconds,
        ) as response:
            outer = json.load(response)
        decoded = json.loads(str(outer.get("response") or ""))
        verdicts = decoded.get("verdicts") if isinstance(decoded, dict) else None
        if not isinstance(verdicts, list) or len(verdicts) != len(items):
            raise ValueError("semantic judge returned an invalid response")
        results: list[dict[str, object]] = []
        for verdict in verdicts:
            category = str(verdict)
            if category not in _CATEGORIES:
                raise ValueError("semantic judge returned an invalid response")
            results.append({"category": category, "equivalent": category == "equivalent"})
        return results


def prepare_semantic_audit(
    workspace_path: str | Path,
) -> dict[str, object]:
    """Queue every non-identity pair for stronger local semantic judging."""
    workspace = Path(workspace_path).expanduser()
    connection = sqlite3.connect(workspace)
    try:
        _ensure_columns(connection)
        rows = connection.execute(
            """
            SELECT record_id, medium, target_text, neutral_text
            FROM pairs WHERE neutral_status = 'ready'
            """
        ).fetchall()
        risky: set[str] = set()
        identities: set[str] = set()
        ordinary: set[str] = set()
        for record_id, _medium, target, neutral in rows:
            record = str(record_id)
            if str(target) == str(neutral):
                identities.add(record)
            elif _semantic_shape(str(target)) != _semantic_shape(str(neutral)):
                risky.add(record)
            else:
                ordinary.add(record)
        connection.executemany(
            """
            UPDATE pairs SET semantic_status = 'passed', semantic_reason = NULL,
                semantic_audit_kind = 'identity' WHERE record_id = ?
            """,
            [(record_id,) for record_id in identities],
        )
        connection.executemany(
            """
            UPDATE pairs SET semantic_status = 'pending', semantic_reason = NULL,
                semantic_audit_kind = ? WHERE record_id = ?
            """,
            [("risk", record_id) for record_id in risky]
            + [("full", record_id) for record_id in ordinary],
        )
        connection.commit()
    finally:
        connection.close()
    return semantic_audit_stats(workspace)


def audit_renderer_workspace(
    workspace_path: str | Path,
    judge: SemanticJudge,
    *,
    batch_size: int = 8,
    max_attempts: int = 2,
    progress: Callable[[dict[str, int]], None] | None = None,
    progress_every: int = 100,
) -> dict[str, object]:
    """Judge queued pairs locally and exclude semantic failures."""

    if batch_size <= 0 or max_attempts <= 0 or progress_every <= 0:
        raise ValueError("semantic audit limits must be positive")
    workspace = Path(workspace_path).expanduser()
    connection = sqlite3.connect(workspace)
    try:
        _ensure_columns(connection)
        rows = connection.execute(
            """
            SELECT record_id, target_text, neutral_text
            FROM pairs
            WHERE neutral_status = 'ready' AND semantic_status = 'pending'
            ORDER BY record_id
            """
        ).fetchall()
        processed = 0
        passed = 0
        failed = 0
        backend_errors = 0
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset : offset + batch_size]
            results: list[dict[str, object]] | None = None
            for _attempt in range(max_attempts):
                try:
                    results = judge.judge_batch(
                        [(str(target), str(neutral)) for _record, target, neutral in batch]
                    )
                    if len(results) != len(batch):
                        raise ValueError("semantic judge returned the wrong batch size")
                except Exception:
                    results = None
                    continue
                break
            if results is None:
                backend_errors += len(batch)
                processed += len(batch)
                if progress is not None and (
                    processed % progress_every == 0 or processed == len(rows)
                ):
                    progress(
                        {
                            "backend_errors": backend_errors,
                            "failed": failed,
                            "passed": passed,
                            "processed": processed,
                        }
                    )
                continue
            for (record_id, _target, _neutral), result in zip(batch, results, strict=True):
                equivalent = bool(result.get("equivalent"))
                category = str(result.get("category") or "other")
                if equivalent:
                    connection.execute(
                        """
                        UPDATE pairs SET semantic_status = 'passed', semantic_reason = ?
                        WHERE record_id = ?
                        """,
                        (category, record_id),
                    )
                    passed += 1
                else:
                    connection.execute(
                        """
                        UPDATE pairs
                        SET neutral_text = NULL, neutral_status = 'excluded',
                            neutral_error = 'semantic_mismatch', semantic_status = 'failed',
                            semantic_reason = ?
                        WHERE record_id = ?
                        """,
                        (category if category in _CATEGORIES else "other", record_id),
                    )
                    failed += 1
            connection.commit()
            processed += len(batch)
            if progress is not None and (processed % progress_every == 0 or processed == len(rows)):
                progress(
                    {
                        "backend_errors": backend_errors,
                        "failed": failed,
                        "passed": passed,
                        "processed": processed,
                    }
                )
    finally:
        connection.close()
    return semantic_audit_stats(workspace)


def semantic_audit_stats(workspace_path: str | Path) -> dict[str, object]:
    """Return aggregate-only semantic audit counts."""

    workspace = Path(workspace_path).expanduser()
    connection = sqlite3.connect(f"file:{workspace}?mode=ro", uri=True)
    try:
        columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(pairs)").fetchall()}
        if "semantic_status" not in columns:
            return {"complete": False, "statuses": {}, "total": 0}
        statuses = {
            str(status): int(count)
            for status, count in connection.execute(
                "SELECT semantic_status, COUNT(*) FROM pairs GROUP BY semantic_status"
            ).fetchall()
            if status is not None
        }
        ready_unchecked = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM pairs
                WHERE neutral_status = 'ready' AND semantic_status IS NULL
                """
            ).fetchone()[0]
        )
        ready_not_passed = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM pairs
                WHERE neutral_status = 'ready'
                  AND COALESCE(semantic_status, '') != 'passed'
                """
            ).fetchone()[0]
        )
        legacy_shape_pass = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM pairs
                WHERE neutral_status = 'ready' AND semantic_status = 'shape_pass'
                """
            ).fetchone()[0]
        )
    finally:
        connection.close()
    return {
        "complete": ready_not_passed == 0,
        "legacy_shape_pass": legacy_shape_pass,
        "ready_not_passed": ready_not_passed,
        "statuses": dict(sorted(statuses.items())),
        "total": sum(statuses.values()) + ready_unchecked,
        "unchecked_ready": ready_unchecked,
    }


def require_complete_semantic_audit(workspace_path: str | Path) -> None:
    """Refuse production export until every ready pair passed local audit routing."""

    stats = semantic_audit_stats(workspace_path)
    if not stats["complete"] or not stats["total"]:
        raise RuntimeError("renderer semantic audit is incomplete")


def _semantic_shape(text: str) -> tuple[bool, ...]:
    return tuple(bool(pattern.search(text)) for pattern in _SEMANTIC_PATTERNS.values())


def _ensure_columns(connection: sqlite3.Connection) -> None:
    columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(pairs)").fetchall()}
    for name in ("semantic_status", "semantic_reason", "semantic_audit_kind"):
        if name not in columns:
            connection.execute(f"ALTER TABLE pairs ADD COLUMN {name} TEXT")
    connection.commit()
