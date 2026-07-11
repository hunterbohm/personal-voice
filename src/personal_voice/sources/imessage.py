"""Import the privacy-audited JSONL emitted by texts-to-transformer."""

from __future__ import annotations

import json
from pathlib import Path

from personal_voice.normalize import normalize_text, private_id


def load_imessage_jsonl(path: str | Path) -> list[dict]:
    """Load normalized iMessage rows without retaining source identifiers."""

    messages: list[dict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            role = row.get("sender_role")
            text = normalize_text(row.get("text"))
            if role not in {"me", "other"} or not text:
                continue
            timestamp_ns = int(row.get("timestamp_ns") or 0)
            messages.append(
                {
                    "conversation_id": private_id("imessage", row.get("chat_id", "unknown")),
                    "sender_role": role,
                    "timestamp": timestamp_ns // 1_000_000_000,
                    "text": text,
                    "source": "imessage",
                    "medium": "text",
                }
            )
    return messages
