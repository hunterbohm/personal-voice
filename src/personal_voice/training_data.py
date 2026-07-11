"""Aggregate-only token preflight for private renderer datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

from personal_voice.local_only import inside_git_worktree


class ChatTokenizer(Protocol):
    """Minimal tokenizer surface used by MLX-LM's chat dataset."""

    def apply_chat_template(self, messages, **kwargs): ...


def load_renderer_tokenizer(model_path: str | Path) -> ChatTokenizer:
    """Load only the pinned local tokenizer without network access."""

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(Path(model_path).expanduser()),
        local_files_only=True,
    )


def renderer_dataset_identity(data_directory: str | Path) -> tuple[str, int]:
    """Return a private dataset digest and record count without exposing records."""

    data = Path(data_directory).expanduser()
    if inside_git_worktree(data) or not data.is_dir() or data.stat().st_mode & 0o077:
        raise ValueError("renderer dataset must be a private directory outside Git")
    digest = hashlib.sha256()
    records = 0
    for split in ("train", "valid", "test"):
        path = data / f"{split}.jsonl"
        if not path.is_file() or path.is_symlink() or path.stat().st_mode & 0o077:
            raise ValueError("renderer dataset splits must be private regular files")
        digest.update(path.name.encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for line in handle:
                digest.update(line)
                records += 1
    if records <= 0:
        raise ValueError("renderer dataset is empty")
    return digest.hexdigest(), records


def training_record_lengths(record: dict, tokenizer: ChatTokenizer) -> tuple[int, int]:
    """Match MLX-LM ChatDataset token and prompt-offset calculations."""

    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("renderer dataset contains an invalid chat record")
    final = messages[-1]
    if not isinstance(final, dict) or final.get("role") != "assistant":
        raise ValueError("renderer dataset must end with an assistant target")
    full = tokenizer.apply_chat_template(messages, return_dict=False)
    prompt = tokenizer.apply_chat_template(
        messages[:-1],
        add_generation_prompt=True,
        return_dict=False,
    )
    total_tokens = len(full)
    supervised_tokens = total_tokens - len(prompt)
    return total_tokens, supervised_tokens


def preflight_renderer_dataset(
    data_directory: str | Path,
    tokenizer: ChatTokenizer,
    *,
    max_seq_length: int,
    min_supervised_tokens: int = 4,
) -> dict[str, object]:
    """Return aggregate truncation checks without emitting dataset content."""

    if max_seq_length <= 0 or min_supervised_tokens <= 0:
        raise ValueError("renderer token limits must be positive")
    data = Path(data_directory).expanduser()
    splits: dict[str, dict[str, int]] = {}
    total = 0
    too_long = 0
    too_short = 0
    max_tokens = 0
    min_target_tokens: int | None = None
    for split in ("train", "valid", "test"):
        path = data / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError("private renderer dataset split is missing")
        split_total = 0
        split_long = 0
        split_short = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError("renderer dataset contains invalid JSON") from error
                if not isinstance(record, dict):
                    raise ValueError("renderer dataset contains an invalid record")
                record_tokens, target_tokens = training_record_lengths(record, tokenizer)
                split_total += 1
                split_long += int(record_tokens > max_seq_length)
                split_short += int(target_tokens < min_supervised_tokens)
                max_tokens = max(max_tokens, record_tokens)
                min_target_tokens = (
                    target_tokens
                    if min_target_tokens is None
                    else min(min_target_tokens, target_tokens)
                )
        splits[split] = {
            "target_too_short": split_short,
            "too_long": split_long,
            "total": split_total,
        }
        total += split_total
        too_long += split_long
        too_short += split_short
    return {
        "max_seq_length": max_seq_length,
        "max_tokens": max_tokens,
        "min_supervised_tokens": min_target_tokens or 0,
        "ok": total > 0 and too_long == 0 and too_short == 0,
        "splits": splits,
        "target_too_short": too_short,
        "too_long": too_long,
        "total": total,
    }
