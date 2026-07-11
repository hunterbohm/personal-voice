from personal_voice.pairing import pair_replies, standalone_replies


def msg(conversation: str, role: str, timestamp: int, text: str) -> dict:
    return {
        "conversation_id": conversation,
        "sender_role": role,
        "timestamp": timestamp,
        "text": text,
        "source": "imessage",
        "medium": "text",
    }


def test_pair_replies_builds_contextual_owner_examples() -> None:
    messages = [
        msg("chat-a", "other", 10, "are you free later?"),
        msg("chat-a", "other", 20, "maybe around seven"),
        msg("chat-a", "me", 30, "yeah that should work"),
        msg("chat-a", "me", 35, "just text me"),
        msg("chat-a", "other", 50, "perfect"),
        msg("chat-a", "me", 60, "sweet"),
        msg("chat-b", "me", 15, "orphan outgoing message"),
    ]

    examples = pair_replies(messages)

    assert [(example["incoming_text"], example["reply_text"]) for example in examples] == [
        ("are you free later?\nmaybe around seven", "yeah that should work\njust text me"),
        ("perfect", "sweet"),
    ]
    assert examples[0]["source"] == "imessage"
    assert examples[0]["medium"] == "text"
    assert examples[0]["conversation_id"] == "chat-a"
    assert examples[0]["example_id"] != examples[1]["example_id"]


def test_pair_replies_does_not_cross_six_hour_session_gap() -> None:
    messages = [
        msg("chat-a", "other", 1, "old question"),
        msg("chat-a", "me", 1 + 6 * 60 * 60 + 1, "late standalone thought"),
    ]

    assert pair_replies(messages) == []


def test_standalone_replies_keep_authored_style_samples() -> None:
    samples = standalone_replies(
        [
            {
                "conversation_id": "public-a",
                "sender_role": "me",
                "timestamp": 100,
                "text": "yeah this is exactly it",
                "source": "x_reply",
                "medium": "public",
                "context_hint": "subject: launch timing",
            },
            {
                "conversation_id": "public-b",
                "sender_role": "other",
                "timestamp": 101,
                "text": "not Writer's writing",
                "source": "x_reply",
                "medium": "public",
            },
        ]
    )

    assert len(samples) == 1
    assert samples[0]["reply_text"] == "yeah this is exactly it"
    assert samples[0]["incoming_text"] == ""
    assert samples[0]["context_text"] == "subject: launch timing"
    assert samples[0]["source"] == "x_reply"


def test_pair_replies_keeps_the_full_session_before_each_reply() -> None:
    messages = [
        msg("chat-a", "other", 10, "first question"),
        msg("chat-a", "me", 20, "first answer"),
        msg("chat-a", "other", 30, "second question"),
        msg("chat-a", "me", 40, "second answer"),
        msg("chat-a", "other", 50, "third question"),
        msg("chat-a", "me", 60, "third answer"),
    ]

    last = pair_replies(messages)[-1]

    assert "other: first question" in last["context_text"]
    assert "me: first answer" in last["context_text"]
