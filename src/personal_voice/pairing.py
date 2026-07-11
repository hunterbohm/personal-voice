"""Turn normalized conversations into contextual reply examples."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from personal_voice.normalize import is_usable_text, private_id


def _merge_turns(messages: list[dict], *, merge_gap_seconds: int) -> list[dict]:
    turns: list[dict] = []
    for message in messages:
        if (
            turns
            and turns[-1]["sender_role"] == message["sender_role"]
            and message["timestamp"] - turns[-1]["timestamp"] <= merge_gap_seconds
        ):
            turns[-1]["text"] += "\n" + message["text"]
            turns[-1]["timestamp"] = message["timestamp"]
        else:
            turns.append(dict(message))
    return turns


def _example_id(source: str, conversation_id: str, timestamp: int, reply: str) -> str:
    return private_id("example", f"{source}\0{conversation_id}\0{timestamp}\0{reply}")


def pair_replies(
    messages: Iterable[dict],
    *,
    session_gap_seconds: int = 6 * 60 * 60,
    merge_gap_seconds: int = 2 * 60,
) -> list[dict]:
    """Return authored replies paired with the incoming turn that prompted them."""

    conversations: dict[str, list[dict]] = defaultdict(list)
    for message in messages:
        if message.get("sender_role") not in {"me", "other"}:
            continue
        if not is_usable_text(message.get("text")):
            continue
        conversations[str(message["conversation_id"])].append(message)

    examples: list[dict] = []
    for conversation_id in sorted(conversations):
        ordered = sorted(conversations[conversation_id], key=lambda row: row["timestamp"])
        sessions: list[list[dict]] = []
        current: list[dict] = []
        for message in ordered:
            if current and message["timestamp"] - current[-1]["timestamp"] > session_gap_seconds:
                sessions.append(current)
                current = []
            current.append(message)
        if current:
            sessions.append(current)

        for session in sessions:
            turns = _merge_turns(session, merge_gap_seconds=merge_gap_seconds)
            for index, turn in enumerate(turns):
                if turn["sender_role"] != "me" or index == 0:
                    continue
                incoming = turns[index - 1]
                if incoming["sender_role"] != "other":
                    continue
                context = turns[:index]
                source = str(turn.get("source") or session[0].get("source") or "unknown")
                medium = str(turn.get("medium") or session[0].get("medium") or "unknown")
                examples.append(
                    {
                        "example_id": _example_id(
                            source, conversation_id, int(turn["timestamp"]), turn["text"]
                        ),
                        "source": source,
                        "medium": medium,
                        "conversation_id": conversation_id,
                        "timestamp": int(turn["timestamp"]),
                        "incoming_text": incoming["text"],
                        "context_text": "\n".join(
                            f"{item['sender_role']}: {item['text']}" for item in context
                        ),
                        "reply_text": turn["text"],
                    }
                )
    return examples


def standalone_replies(messages: Iterable[dict]) -> list[dict]:
    """Turn authored messages without recoverable prompts into style examples."""

    examples: list[dict] = []
    for message in messages:
        reply = str(message.get("text") or "").strip()
        if message.get("sender_role") != "me" or not is_usable_text(reply):
            continue
        source = str(message.get("source") or "unknown")
        medium = str(message.get("medium") or "unknown")
        conversation_id = str(message.get("conversation_id") or "unknown")
        timestamp = int(message.get("timestamp") or 0)
        examples.append(
            {
                "example_id": _example_id(source, conversation_id, timestamp, reply),
                "source": source,
                "medium": medium,
                "conversation_id": conversation_id,
                "timestamp": timestamp,
                "incoming_text": "",
                "context_text": str(message.get("context_hint") or "").strip(),
                "reply_text": reply,
            }
        )
    return examples
