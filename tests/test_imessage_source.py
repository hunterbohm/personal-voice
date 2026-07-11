import json

from personal_voice.sources.imessage import load_imessage_jsonl


def test_load_imessage_jsonl_normalizes_private_messages(tmp_path) -> None:
    source = tmp_path / "messages.jsonl"
    rows = [
        {
            "message_id": "message-a",
            "chat_id": "private-chat-id",
            "timestamp_ns": 1_700_000_000_000_000_000,
            "sender_role": "other",
            "participant_id": "participant-a",
            "text": "email me at person@example.com or see https://example.com",
        },
        {
            "message_id": "message-b",
            "chat_id": "private-chat-id",
            "timestamp_ns": 1_700_000_001_000_000_000,
            "sender_role": "me",
            "participant_id": "me",
            "text": "yeah call 555-867-5309",
        },
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))

    messages = load_imessage_jsonl(source)

    assert len(messages) == 2
    assert messages[0]["timestamp"] == 1_700_000_000
    assert messages[0]["source"] == "imessage"
    assert messages[0]["medium"] == "text"
    assert messages[0]["conversation_id"] != "private-chat-id"
    assert messages[0]["conversation_id"] == messages[1]["conversation_id"]
    assert messages[0]["text"] == "email me at <|email_address|> or see <|url|>"
    assert messages[1]["text"] == "yeah call <|phone|>"
