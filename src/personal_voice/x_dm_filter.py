"""Remove repeated automated and broadcast-style X direct messages."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable

_MIN_WORDS = 12
_MIN_CONVERSATIONS = 5
_WORD = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)
_DIRECT_IDENTIFIER = re.compile(r"<\|(?:url|email_address|phone)\|>")
_MENTION = re.compile(r"(?<!\w)@[A-Za-z0-9_]{1,30}")
_NUMBER = re.compile(r"(?<!\w)\d+(?:[.,:/\-]\d+)*(?!\w)")
_GREETING_NAME = re.compile(
    r"^((?:hey|hi|hello|yo)\s+)([^\s,!:?]{1,30})(?=[,\s!:?])",
    re.IGNORECASE,
)


def filter_automated_x_dms(messages: Iterable[dict]) -> list[dict]:
    """Drop long X DM templates repeated across at least five conversations.

    The whole-message fingerprint only normalizes variable contact details, numbers,
    mentions, and a name immediately after a greeting. Short conversational replies
    are never candidates for removal.
    """

    rows = list(messages)
    conversations_by_template: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        if not _eligible(row):
            continue
        key = (str(row["sender_role"]), _template_fingerprint(str(row["text"])))
        conversations_by_template[key].add(str(row["conversation_id"]))

    retained: list[dict] = []
    for row in rows:
        if not _eligible(row):
            retained.append(row)
            continue
        key = (str(row["sender_role"]), _template_fingerprint(str(row["text"])))
        if len(conversations_by_template[key]) < _MIN_CONVERSATIONS:
            retained.append(row)
    return retained


def _eligible(row: dict) -> bool:
    return (
        row.get("source") == "x_dm"
        and row.get("sender_role") in {"me", "other"}
        and len(_WORD.findall(str(row.get("text") or ""))) >= _MIN_WORDS
    )


def _template_fingerprint(text: str) -> str:
    value = re.sub(r"\s+", " ", text.casefold()).strip()
    value = _DIRECT_IDENTIFIER.sub("<slot>", value)
    value = _MENTION.sub("<mention>", value)
    value = _NUMBER.sub("<num>", value)
    return _GREETING_NAME.sub(r"\1<name>", value, count=1)
