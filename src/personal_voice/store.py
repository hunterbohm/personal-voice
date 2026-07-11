"""Private SQLite storage and retrieval for authored reply examples."""

from __future__ import annotations

import os
import re
import sqlite3
import tempfile
from collections.abc import Iterable
from pathlib import Path

_EXAMPLE_FIELDS = (
    "example_id",
    "source",
    "medium",
    "conversation_id",
    "timestamp",
    "incoming_text",
    "context_text",
    "reply_text",
)
_MAX_REPLY_CHARS = 4000
_MAX_RETURN_CONTEXT_CHARS = 8000
_MEDIUM_MISMATCH_PENALTY = 2.5
_REPEATED_SOURCE_PENALTY = 0.35
_QUERY_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
_TURN_LABEL = re.compile(r"(?m)^(?:me|other): ")
_STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "as",
    "can",
    "draft",
    "for",
    "in",
    "is",
    "it",
    "me",
    "message",
    "my",
    "of",
    "on",
    "or",
    "please",
    "reply",
    "the",
    "this",
    "to",
    "write",
    "you",
}


def build_store(path: str | Path, examples: Iterable[dict]) -> dict:
    """Atomically replace a private corpus database and return aggregate counts."""

    database = Path(path).expanduser()
    parent_existed = database.parent.exists()
    database.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if parent_existed and database.parent.stat().st_mode & 0o077:
        raise PermissionError(
            f"personal-voice requires a private directory (0700): {database.parent}"
        )
    if not parent_existed:
        database.parent.chmod(0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{database.name}.", suffix=".tmp", dir=database.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.chmod(0o600)

    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(temporary)
        connection.executescript(
            """
            PRAGMA journal_mode = DELETE;
            PRAGMA synchronous = FULL;
            CREATE TABLE metadata (
                schema_version INTEGER NOT NULL
            );
            INSERT INTO metadata(schema_version) VALUES (1);
            CREATE TABLE examples (
                example_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                medium TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                incoming_text TEXT NOT NULL,
                context_text TEXT NOT NULL,
                reply_text TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE examples_fts USING fts5(
                example_id UNINDEXED,
                incoming_text,
                context_text,
                reply_text,
                tokenize = 'unicode61 remove_diacritics 2'
            );
            """
        )
        seen_ids: set[str] = set()
        for raw in examples:
            row = _normalized_example(raw)
            if row is None or row[0] in seen_ids:
                continue
            seen_ids.add(row[0])
            connection.execute(
                """
                INSERT INTO examples(
                    example_id, source, medium, conversation_id, timestamp,
                    incoming_text, context_text, reply_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )
            connection.execute(
                """
                INSERT INTO examples_fts(
                    example_id, incoming_text, context_text, reply_text
                ) VALUES (?, ?, ?, ?)
                """,
                (row[0], row[5], row[6], row[7]),
            )
        connection.commit()
        connection.close()
        connection = None
        os.replace(temporary, database)
        database.chmod(0o600)
    except Exception:
        if connection is not None:
            connection.close()
        temporary.unlink(missing_ok=True)
        raise

    return store_stats(database)


def retrieve_examples(
    path: str | Path,
    query: str,
    *,
    medium: str = "auto",
    limit: int = 8,
) -> list[dict]:
    """Return a deterministic, channel-aware and source-diverse example pack."""

    if limit <= 0:
        return []
    target_medium = _resolve_medium(medium, query)
    candidate_limit = max(50, limit * 12)
    connection = _open_readonly(path)
    connection.row_factory = sqlite3.Row
    try:
        ranked: list[sqlite3.Row] = []
        query_words = _query_words(query)
        fts_query = _fts_query(query_words)
        if fts_query:
            ranked = connection.execute(
                """
                SELECT e.*,
                       bm25(examples_fts, 5.0, 2.0, 0.75) AS lexical_rank
                FROM examples_fts
                JOIN examples AS e ON e.example_id = examples_fts.example_id
                WHERE examples_fts MATCH ?
                ORDER BY
                    lexical_rank ASC,
                    CASE WHEN e.incoming_text <> '' THEN 0 ELSE 1 END,
                    e.timestamp DESC,
                    e.example_id ASC
                LIMIT ?
                """,
                (fts_query, candidate_limit),
            ).fetchall()
        ranked_ids = {str(row["example_id"]) for row in ranked}
        fallback = connection.execute(
            """
            SELECT examples.*, 0.0 AS lexical_rank
            FROM examples
            ORDER BY
                CASE WHEN medium = ? THEN 0 ELSE 1 END,
                CASE WHEN incoming_text <> '' THEN 0 ELSE 1 END,
                timestamp DESC,
                example_id ASC
            LIMIT ?
            """,
            (target_medium, candidate_limit),
        ).fetchall()
    except sqlite3.Error as error:
        raise ValueError("invalid personal-voice index") from error
    finally:
        connection.close()

    candidates = ranked + [row for row in fallback if str(row["example_id"]) not in ranked_ids]
    reranked = _rank_matches(candidates, query_words, target_medium)
    unique = _dedupe_replies(reranked)
    selected = _select_with_source_penalty(unique, limit)
    results = [{field: row[field] for field in _EXAMPLE_FIELDS} for row in selected]
    for result in results:
        result["context_text"] = _bounded_context(result["context_text"])
    return results


def store_stats(path: str | Path) -> dict:
    """Return corpus counts without exposing source text."""

    connection = _open_readonly(path)
    try:
        total = int(connection.execute("SELECT COUNT(*) FROM examples").fetchone()[0])
        sources = dict(
            connection.execute(
                "SELECT source, COUNT(*) FROM examples GROUP BY source ORDER BY source"
            ).fetchall()
        )
        mediums = dict(
            connection.execute(
                "SELECT medium, COUNT(*) FROM examples GROUP BY medium ORDER BY medium"
            ).fetchall()
        )
    except sqlite3.Error as error:
        raise ValueError("invalid personal-voice index") from error
    finally:
        connection.close()
    return {"mediums": mediums, "sources": sources, "total": total}


def _normalized_example(raw: dict) -> tuple | None:
    reply = str(raw.get("reply_text") or "").strip()
    example_id = str(raw.get("example_id") or "").strip()
    if not reply or len(reply) > _MAX_REPLY_CHARS or not example_id:
        return None
    return (
        example_id,
        str(raw.get("source") or "unknown"),
        str(raw.get("medium") or "unknown"),
        str(raw.get("conversation_id") or "unknown"),
        int(raw.get("timestamp") or 0),
        str(raw.get("incoming_text") or "").strip(),
        str(raw.get("context_text") or "").strip(),
        reply,
    )


def _bounded_context(value: str) -> str:
    if len(value) <= _MAX_RETURN_CONTEXT_CHARS:
        return value
    tail = value[-_MAX_RETURN_CONTEXT_CHARS:]
    first_complete_turn = _TURN_LABEL.search(tail)
    return tail[first_complete_turn.start() :] if first_complete_turn else ""


def _open_readonly(path: str | Path) -> sqlite3.Connection:
    database = Path(path).expanduser()
    if not database.is_file():
        raise FileNotFoundError(database)
    return sqlite3.connect(f"file:{database}?mode=ro", uri=True)


def _resolve_medium(medium: str, query: str) -> str:
    requested = medium.strip().casefold()
    if requested and requested != "auto":
        return requested
    lowered = query.casefold()
    if _matches_intent(lowered, (r"\bemail\b", r"\bmailbox\b", r"\bsubject line\b")):
        return "email"
    if _matches_intent(
        lowered,
        (r"\btweet\b", r"\bx reply\b", r"\bpublic reply\b", r"\bcomment\b", r"\bpost\b"),
    ):
        return "public"
    if _matches_intent(lowered, (r"\bimessage\b", r"\bsms\b", r"\btext message\b", r"\btexting\b")):
        return "text"
    if _matches_intent(
        lowered,
        (r"\bdm\b", r"\bdirect message\b", r"\bdiscord\b", r"\bslack\b", r"\blinkedin message\b"),
    ):
        return "dm"
    return ""


def _matches_intent(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _query_words(query: str) -> list[str]:
    words: list[str] = []
    seen: set[str] = set()
    for word in _QUERY_WORD.findall(query.casefold()):
        if len(word) < 2 or word in _STOPWORDS or word in seen:
            continue
        seen.add(word)
        words.append(word)
        if len(words) == 20:
            break
    return words


def _fts_query(words: list[str]) -> str:
    return " OR ".join(f'"{word.replace(chr(34), chr(34) * 2)}"' for word in words)


def _rank_matches(
    rows: list[sqlite3.Row], query_words: list[str], target_medium: str
) -> list[sqlite3.Row]:
    query_set = set(query_words)

    def score(item: tuple[int, sqlite3.Row]) -> tuple[float, int]:
        index, row = item
        text = " ".join(
            str(row[field]) for field in ("incoming_text", "context_text", "reply_text")
        )
        text_words = set(_QUERY_WORD.findall(text.casefold()))
        coverage = len(query_set & text_words) / max(1, len(query_set))
        medium_penalty = (
            0.0 if not target_medium or row["medium"] == target_medium else _MEDIUM_MISMATCH_PENALTY
        )
        standalone_penalty = 0.0 if row["incoming_text"] else 0.25
        return (-4.0 * coverage + medium_penalty + standalone_penalty, index)

    return [row for _, row in sorted(enumerate(rows), key=score)]


def _dedupe_replies(rows: list[sqlite3.Row]) -> list[sqlite3.Row]:
    seen: set[str] = set()
    unique: list[sqlite3.Row] = []
    for row in rows:
        key = re.sub(r"\s+", " ", str(row["reply_text"]).casefold()).strip()
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _select_with_source_penalty(rows: list[sqlite3.Row], limit: int) -> list[sqlite3.Row]:
    selected: list[sqlite3.Row] = []
    remaining = list(enumerate(rows))
    source_counts: dict[str, int] = {}
    while remaining and len(selected) < limit:
        position = min(
            range(len(remaining)),
            key=lambda candidate_index: _selection_score(remaining[candidate_index], source_counts),
        )
        _, row = remaining.pop(position)
        selected.append(row)
        source = str(row["source"])
        source_counts[source] = source_counts.get(source, 0) + 1
    return selected


def _selection_score(
    candidate: tuple[int, sqlite3.Row], source_counts: dict[str, int]
) -> tuple[float, int]:
    base_rank, row = candidate
    repeated_source_penalty = _REPEATED_SOURCE_PENALTY * source_counts.get(str(row["source"]), 0)
    return (base_rank + repeated_source_penalty, base_rank)
