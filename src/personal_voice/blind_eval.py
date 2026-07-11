"""Private blind A/B evaluation for retrieval-only versus tuned drafts."""

from __future__ import annotations

import hashlib
import math
import sqlite3
import time
from pathlib import Path

from personal_voice.local_only import inside_git_worktree
from personal_voice.renderer import RENDERER_CONTRACT_VERSION, RendererBackend, render_draft


def add_blind_case(
    pack_path: str | Path,
    baseline_draft: str,
    medium: str,
    backend: RendererBackend,
    *,
    baseline_source: str,
    preserve_literals: list[str] | None = None,
    evaluation_context: str = "",
) -> dict[str, object]:
    """Render and privately store one randomized baseline-versus-tuned case."""

    if baseline_source != "retrieval_augmented_draft":
        raise ValueError("blind baseline must attest retrieval_augmented_draft provenance")
    pack = Path(pack_path).expanduser()
    _ensure_pack(pack)
    baseline = str(baseline_draft or "").strip()
    result = render_draft(
        baseline,
        medium,
        backend,
        preserve_literals=preserve_literals or [],
    )
    if result["status"] != "rendered" or result["text"] == baseline:
        return {"added": False, "case_id": "", "status": str(result["status"])}
    candidate_hash = str(getattr(backend, "candidate_hash", "unknown"))
    case_id = _digest(
        "blind-case-v2",
        medium,
        baseline,
        candidate_hash,
        RENDERER_CONTRACT_VERSION,
        baseline_source,
    )
    tuned_is_a = int(_digest("candidate-order", case_id)[:8], 16) % 2 == 0
    candidate_a = str(result["text"]) if tuned_is_a else baseline
    candidate_b = baseline if tuned_is_a else str(result["text"])
    label_a = "tuned" if tuned_is_a else "baseline"
    label_b = "baseline" if tuned_is_a else "tuned"
    connection = sqlite3.connect(pack)
    try:
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO cases(
                case_id, medium, candidate_a, candidate_b, label_a, label_b,
                status, model_version, candidate_sha256, baseline_source,
                evaluation_context, renderer_contract_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?)
            """,
            (
                case_id,
                medium,
                candidate_a,
                candidate_b,
                label_a,
                label_b,
                str(result["model_version"]),
                candidate_hash,
                baseline_source,
                str(evaluation_context or ""),
                RENDERER_CONTRACT_VERSION,
                int(time.time()),
            ),
        )
        connection.commit()
        added = cursor.rowcount == 1
    finally:
        connection.close()
    pack.chmod(0o600)
    return {"added": added, "case_id": case_id, "status": "rendered"}


def next_blind_case(pack_path: str | Path) -> dict[str, str] | None:
    """Return the next unlabeled case without revealing candidate provenance."""

    pack = Path(pack_path).expanduser()
    _ensure_pack(pack)
    connection = sqlite3.connect(f"file:{pack}?mode=ro", uri=True)
    try:
        row = connection.execute(
            """
            SELECT case_id, medium, candidate_a, candidate_b, evaluation_context
            FROM cases WHERE status = 'pending' ORDER BY case_id LIMIT 1
            """
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        return None
    return {
        "candidate_a": str(row[2]),
        "candidate_b": str(row[3]),
        "case_id": str(row[0]),
        "evaluation_context": str(row[4]),
        "medium": str(row[1]),
    }


def record_blind_vote(
    pack_path: str | Path,
    case_id: str,
    choice: str,
    *,
    would_send: str,
) -> None:
    """Record one blind preference and sendability choice."""

    if choice not in {"a", "b", "tie"}:
        raise ValueError("choice must be a, b, or tie")
    if would_send not in {"a", "b", "both", "neither"}:
        raise ValueError("would_send must be a, b, both, or neither")
    pack = Path(pack_path).expanduser()
    _ensure_pack(pack)
    connection = sqlite3.connect(pack)
    try:
        cursor = connection.execute(
            """
            UPDATE cases
            SET status = 'completed', choice = ?, would_send = ?
            WHERE case_id = ? AND status = 'pending'
            """,
            (choice, would_send, case_id),
        )
        if cursor.rowcount != 1:
            raise ValueError("blind case is missing or already completed")
        connection.commit()
    finally:
        connection.close()


def blind_eval_stats(
    pack_path: str | Path,
    *,
    model_version: str | None = None,
    candidate_sha256: str | None = None,
    renderer_contract_version: int | None = None,
) -> dict[str, object]:
    """Return aggregate-only hidden-label preference counts."""

    pack = Path(pack_path).expanduser()
    _ensure_pack(pack)
    connection = sqlite3.connect(f"file:{pack}?mode=ro", uri=True)
    try:
        where: list[str] = ["baseline_source = ?"]
        parameters: list[str] = ["retrieval_augmented_draft"]
        if model_version is not None:
            where.append("model_version = ?")
            parameters.append(model_version)
        if candidate_sha256 is not None:
            where.append("candidate_sha256 = ?")
            parameters.append(candidate_sha256)
        if renderer_contract_version is not None:
            where.append("renderer_contract_version = ?")
            parameters.append(str(renderer_contract_version))
        clause = " WHERE " + " AND ".join(where) if where else ""
        rows = connection.execute(
            f"""
            SELECT status, choice, would_send, label_a, label_b, medium,
                   candidate_a, candidate_b
            FROM cases{clause}
            """,
            parameters,
        ).fetchall()
    finally:
        connection.close()
    counts = _score_rows(rows)
    by_medium: dict[str, dict[str, object]] = {}
    for medium in ("text", "dm", "email", "public"):
        medium_counts = _score_rows([row for row in rows if row[5] == medium])
        medium_counts["eligible"] = _medium_eligible(medium_counts)
        by_medium[medium] = medium_counts
    bound_candidate = bool(
        model_version
        and candidate_sha256
        and renderer_contract_version == RENDERER_CONTRACT_VERSION
    )
    counts["activation_eligible"] = bool(
        bound_candidate
        and counts["invalid"] == 0
        and counts["completed"] >= 200
        and counts["non_ties"] >= 100
        and counts["tuned_non_tie_rate"] >= 0.6
        and counts["tuned_non_tie_wilson_lower_95"] > 0.5
        and counts["would_send_tuned"] > counts["would_send_baseline"]
        and counts["would_send_tuned"] >= 100
        and all(stats["eligible"] for stats in by_medium.values())
    )
    return {
        **counts,
        "candidate_sha256": candidate_sha256 or "unbound",
        "mediums": by_medium,
        "model_version": model_version or "unbound",
        "renderer_contract_version": renderer_contract_version or "unbound",
    }


def _score_rows(rows: list[tuple]) -> dict[str, object]:
    counts: dict[str, int] = {
        "baseline_wins": 0,
        "completed": 0,
        "invalid": 0,
        "pending": 0,
        "ties": 0,
        "total": len(rows),
        "tuned_wins": 0,
        "would_send_baseline": 0,
        "would_send_neither": 0,
        "would_send_tuned": 0,
    }
    comparisons: set[tuple[str, str, str]] = set()
    for status, choice, would_send, label_a, label_b, medium, candidate_a, candidate_b in rows:
        labels = {label_a: candidate_a, label_b: candidate_b}
        structurally_valid = bool(
            medium in {"text", "dm", "email", "public"}
            and isinstance(candidate_a, str)
            and candidate_a.strip()
            and isinstance(candidate_b, str)
            and candidate_b.strip()
            and candidate_a != candidate_b
            and set(labels) == {"baseline", "tuned"}
            and status in {"pending", "completed"}
            and (
                (status == "pending" and choice is None and would_send is None)
                or (
                    status == "completed"
                    and choice in {"a", "b", "tie"}
                    and would_send in {"a", "b", "both", "neither"}
                )
            )
        )
        comparison = (
            str(medium),
            str(labels.get("baseline") or ""),
            str(labels.get("tuned") or ""),
        )
        if not structurally_valid or comparison in comparisons:
            counts["invalid"] += 1
            continue
        comparisons.add(comparison)
        if status != "completed":
            counts["pending"] += 1
            continue
        counts["completed"] += 1
        positions = {"a": label_a, "b": label_b}
        if choice == "tie":
            counts["ties"] += 1
        else:
            counts[f"{positions[choice]}_wins"] += 1
        if would_send == "neither":
            counts["would_send_neither"] += 1
        elif would_send == "both":
            counts["would_send_baseline"] += 1
            counts["would_send_tuned"] += 1
        else:
            counts[f"would_send_{positions[would_send]}"] += 1
    non_ties = counts["tuned_wins"] + counts["baseline_wins"]
    tuned_rate = counts["tuned_wins"] / non_ties if non_ties else 0.0
    wilson_lower = _wilson_lower_bound(counts["tuned_wins"], non_ties)
    return {
        **counts,
        "non_ties": non_ties,
        "tuned_non_tie_rate": round(tuned_rate, 6),
        "tuned_non_tie_wilson_lower_95": round(wilson_lower, 6),
    }


def _medium_eligible(counts: dict[str, object]) -> bool:
    return bool(
        counts["invalid"] == 0
        and counts["completed"] >= 50
        and counts["non_ties"] >= 20
        and counts["tuned_non_tie_rate"] >= 0.6
        and counts["tuned_non_tie_wilson_lower_95"] > 0.5
        and counts["would_send_tuned"] > counts["would_send_baseline"]
        and counts["would_send_tuned"] >= 20
    )


def _wilson_lower_bound(successes: int, trials: int) -> float:
    """Return the lower end of a two-sided 95% Wilson score interval."""

    if trials <= 0:
        return 0.0
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1 + (z**2 / trials)
    center = proportion + (z**2 / (2 * trials))
    margin = z * math.sqrt((proportion * (1 - proportion) + z**2 / (4 * trials)) / trials)
    return (center - margin) / denominator


def _ensure_pack(path: Path) -> None:
    if inside_git_worktree(path):
        raise ValueError("blind evaluation data must live outside a Git worktree")
    existed = path.parent.exists()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if existed and path.parent.stat().st_mode & 0o077:
        raise PermissionError("blind evaluation directory must be private (0700)")
    path.parent.chmod(0o700)
    if path.exists() and path.stat().st_mode & 0o077:
        raise PermissionError("blind evaluation database must be private (0600)")
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode = DELETE;
            CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY,
                medium TEXT NOT NULL,
                candidate_a TEXT NOT NULL,
                candidate_b TEXT NOT NULL,
                label_a TEXT NOT NULL,
                label_b TEXT NOT NULL,
                status TEXT NOT NULL,
                choice TEXT,
                would_send TEXT,
                model_version TEXT NOT NULL,
                candidate_sha256 TEXT NOT NULL DEFAULT '',
                baseline_source TEXT NOT NULL DEFAULT '',
                evaluation_context TEXT NOT NULL DEFAULT '',
                renderer_contract_version INTEGER NOT NULL DEFAULT 1,
                created_at INTEGER NOT NULL
            );
            """
        )
        columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(cases)").fetchall()}
        if "candidate_sha256" not in columns:
            connection.execute(
                "ALTER TABLE cases ADD COLUMN candidate_sha256 TEXT NOT NULL DEFAULT ''"
            )
        if "evaluation_context" not in columns:
            connection.execute(
                "ALTER TABLE cases ADD COLUMN evaluation_context TEXT NOT NULL DEFAULT ''"
            )
        if "baseline_source" not in columns:
            connection.execute(
                "ALTER TABLE cases ADD COLUMN baseline_source TEXT NOT NULL DEFAULT ''"
            )
        if "renderer_contract_version" not in columns:
            connection.execute(
                "ALTER TABLE cases ADD COLUMN renderer_contract_version INTEGER NOT NULL DEFAULT 1"
            )
        connection.commit()
    finally:
        connection.close()
    path.chmod(0o600)


def _digest(*parts: object) -> str:
    return hashlib.sha256("\0".join(str(part) for part in parts).encode()).hexdigest()[:24]
