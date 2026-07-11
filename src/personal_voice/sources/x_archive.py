"""Load normalized messages from an official X data archive.

Until the project has a shared Message type, public loaders return dictionaries
with ``conversation_id``, ``sender_role``, ``timestamp``, ``text``, ``source``,
and ``medium`` keys.
"""

from __future__ import annotations

import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from personal_voice.normalize import is_usable_text, normalize_text, private_id


def load_x_archive(path: str | Path) -> list[dict[str, Any]]:
    """Return normalized direct messages authored by or sent to the archive owner."""
    with zipfile.ZipFile(path) as archive:
        owner_id = _account_owner_id(archive)
        messages: list[dict[str, Any]] = []
        for member in archive.infolist():
            if Path(member.filename).name not in {
                "direct-messages.js",
                "direct-messages-group.js",
            }:
                continue
            for item in _read_wrapped_json(archive, member):
                conversation = item.get("dmConversation", {})
                conversation_id = private_id("x_dm", conversation["conversationId"])
                for event in conversation.get("messages", []):
                    created = event.get("messageCreate")
                    if not created:
                        continue
                    text = normalize_text(created["text"])
                    if not is_usable_text(text):
                        continue
                    messages.append(
                        {
                            "conversation_id": conversation_id,
                            "sender_role": "me" if created["senderId"] == owner_id else "other",
                            "timestamp": _unix_timestamp(created["createdAt"]),
                            "text": text,
                            "source": "x_dm",
                            "medium": "dm",
                        }
                    )
        return messages


def load_x_public_replies(path: str | Path) -> list[dict[str, Any]]:
    """Return public replies authored by the account represented by the archive."""
    with zipfile.ZipFile(path) as archive:
        replies: list[dict[str, Any]] = []
        for member in archive.infolist():
            if Path(member.filename).name != "tweets.js":
                continue
            for item in _read_wrapped_json(archive, member):
                tweet = item.get("tweet", {})
                parent_id = tweet.get("in_reply_to_status_id_str")
                if not parent_id:
                    continue
                text = normalize_text(tweet["full_text"])
                if not is_usable_text(text):
                    continue
                replies.append(
                    {
                        "conversation_id": private_id("x_reply", parent_id),
                        "sender_role": "me",
                        "timestamp": _unix_timestamp(tweet["created_at"]),
                        "text": text,
                        "source": "x_reply",
                        "medium": "public",
                    }
                )
        return replies


def _account_owner_id(archive: zipfile.ZipFile) -> str:
    for member in archive.infolist():
        if Path(member.filename).name == "account.js":
            accounts = _read_wrapped_json(archive, member)
            return str(accounts[0]["account"]["accountId"])
    raise ValueError("X archive does not contain account.js")


def _read_wrapped_json(archive: zipfile.ZipFile, member: zipfile.ZipInfo) -> Any:
    source = archive.read(member).decode("utf-8-sig")
    _, separator, payload = source.partition("=")
    if not separator:
        raise ValueError(f"X archive member {Path(member.filename).name} has no JS assignment")
    value, _ = json.JSONDecoder().raw_decode(payload.lstrip())
    return value


def _unix_timestamp(value: str) -> int:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        parsed = datetime.strptime(value, "%a %b %d %H:%M:%S %z %Y")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp())
