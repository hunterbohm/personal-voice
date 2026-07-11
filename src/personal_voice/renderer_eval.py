"""Aggregate-only evaluation for the optional local style renderer."""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from personal_voice.renderer import (
    RENDER_SYSTEM_PROMPT,
    RendererBackend,
    parse_renderer_target,
    parse_renderer_user_prompt,
    render_draft,
    render_user_prompt,
)

_MEDIA = ("text", "dm", "email", "public")
_PLACEHOLDER = re.compile(r"\[\[PV_LITERAL_\d+\]\]")


def evaluate_renderer_dataset(
    data_directory: str | Path,
    backend: RendererBackend,
    *,
    split: str = "test",
    limit_per_medium: int = 50,
) -> dict[str, object]:
    """Evaluate the exact private MLX records bound to a staged candidate."""

    if split not in {"train", "valid", "test"}:
        raise ValueError("split must be train, valid, or test")
    if limit_per_medium <= 0:
        raise ValueError("limit_per_medium must be positive")
    path = Path(data_directory).expanduser() / f"{split}.jsonl"
    if not path.is_file() or path.is_symlink() or path.stat().st_mode & 0o077:
        raise ValueError("renderer evaluation dataset must be a private regular file")
    rows: list[tuple[str, str, str]] = []
    counts: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
                medium, canonical, target = _renderer_dataset_row(record)
            except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "renderer evaluation dataset contains an invalid record"
                ) from error
            if counts[medium] >= limit_per_medium:
                continue
            rows.append((medium, canonical, target))
            counts[medium] += 1
    return _evaluate_rows(rows, backend)


def evaluate_renderer_workspace(
    workspace_path: str | Path,
    backend: RendererBackend,
    *,
    split: str = "test",
    limit_per_medium: int = 50,
) -> dict[str, object]:
    """Measure whether guarded rewrites move held-out drafts toward writer targets."""

    if split not in {"train", "valid", "test"}:
        raise ValueError("split must be train, valid, or test")
    if limit_per_medium <= 0:
        raise ValueError("limit_per_medium must be positive")
    workspace = Path(workspace_path).expanduser()
    connection = sqlite3.connect(f"file:{workspace}?mode=ro", uri=True)
    rows: list[tuple[str, str, str]] = []
    try:
        for medium in _MEDIA:
            rows.extend(
                connection.execute(
                    """
                    SELECT medium, neutral_text, target_text
                    FROM pairs
                    WHERE split = ? AND medium = ? AND neutral_status = 'ready'
                    ORDER BY record_id
                    LIMIT ?
                    """,
                    (split, medium, limit_per_medium),
                ).fetchall()
            )
    finally:
        connection.close()

    return _evaluate_rows(rows, backend)


def _evaluate_rows(
    rows: list[tuple[str, str, str]],
    backend: RendererBackend,
) -> dict[str, object]:
    statuses = Counter({"rendered": 0, "rejected": 0, "unavailable": 0})
    mediums: Counter[str] = Counter()
    deltas: list[float] = []
    closer_to_target = 0
    exact_target = 0
    for medium, neutral, target in rows:
        result = render_draft(str(neutral), str(medium), backend)
        status = str(result["status"])
        statuses[status] += 1
        mediums[str(medium)] += 1
        if status != "rendered":
            deltas.append(0.0)
            continue
        rendered = str(result["text"])
        baseline_similarity = _similarity(str(neutral), str(target))
        rendered_similarity = _similarity(rendered, str(target))
        delta = rendered_similarity - baseline_similarity
        deltas.append(delta)
        if delta > 0:
            closer_to_target += 1
        if _normalized(rendered) == _normalized(str(target)):
            exact_target += 1
    return {
        "closer_to_target": closer_to_target,
        "exact_target": exact_target,
        "mean_similarity_delta": round(sum(deltas) / len(deltas), 6) if deltas else 0.0,
        "mediums": dict(sorted(mediums.items())),
        "model_version": str(getattr(backend, "version", "unknown")),
        "statuses": {name: int(statuses[name]) for name in ("rendered", "rejected", "unavailable")},
        "total": len(rows),
    }


def _renderer_dataset_row(record: object) -> tuple[str, str, str]:
    if not isinstance(record, dict) or set(record) != {"messages"}:
        raise ValueError("invalid renderer record")
    messages = record["messages"]
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("invalid renderer messages")
    system, user, assistant = messages
    if (
        not isinstance(system, dict)
        or set(system) != {"role", "content"}
        or system.get("role") != "system"
        or system.get("content") != RENDER_SYSTEM_PROMPT
        or not isinstance(user, dict)
        or set(user) != {"role", "content"}
        or user.get("role") != "user"
        or not isinstance(user.get("content"), str)
        or not isinstance(assistant, dict)
        or set(assistant) != {"role", "content"}
        or assistant.get("role") != "assistant"
        or not isinstance(assistant.get("content"), str)
    ):
        raise ValueError("invalid renderer messages")
    user_content = str(user["content"])
    canonical = parse_renderer_user_prompt(user_content)
    first_line = user_content.partition("\n")[0]
    medium = first_line.removeprefix("Medium: ")
    if medium not in _MEDIA or render_user_prompt(medium, canonical) != user_content:
        raise ValueError("invalid renderer medium")
    target = parse_renderer_target(str(assistant["content"]))
    canonical_placeholders = _PLACEHOLDER.findall(canonical)
    target_placeholders = _PLACEHOLDER.findall(target)
    if (
        Counter(canonical_placeholders) != Counter(target_placeholders)
        or canonical_placeholders != target_placeholders
    ):
        raise ValueError("renderer evaluation literals do not align")
    literals = {
        placeholder: f"@hvliteral{index}"
        for index, placeholder in enumerate(dict.fromkeys(target_placeholders))
    }

    def materialize(value: str) -> str:
        return _PLACEHOLDER.sub(lambda match: literals[match.group()], value)

    return medium, materialize(canonical), materialize(target)


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, _normalized(left), _normalized(right)).ratio()
