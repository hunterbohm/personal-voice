import json
import zipfile
from pathlib import Path

from personal_voice.sources.x_archive import load_x_archive, load_x_public_replies


def _write_x_archive(tmp_path: Path, files: dict[str, tuple[str, object]]) -> Path:
    archive_path = tmp_path / "twitter-archive.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for filename, (namespace, payload) in files.items():
            wrapped_json = f"window.YTD.{namespace}.part0 = {json.dumps(payload)};"
            archive.writestr(f"twitter-archive/data/{filename}", wrapped_json)
    return archive_path


def test_load_x_archive_assigns_me_and_other_roles_from_the_archive_account(tmp_path: Path) -> None:
    archive_path = _write_x_archive(
        tmp_path,
        {
            "account.js": (
                "account",
                [{"account": {"accountId": "111", "username": "archive-owner"}}],
            ),
            "direct-messages.js": (
                "direct_messages",
                [
                    {
                        "dmConversation": {
                            "conversationId": "111-222",
                            "messages": [
                                {
                                    "messageCreate": {
                                        "id": "raw-message-id-1",
                                        "senderId": "222",
                                        "recipientId": "111",
                                        "text": "Are you free tomorrow?",
                                        "createdAt": "2024-01-02T03:04:05.000Z",
                                    }
                                },
                                {
                                    "messageCreate": {
                                        "id": "raw-message-id-2",
                                        "senderId": "111",
                                        "recipientId": "222",
                                        "text": "yeah after 3 works",
                                        "createdAt": "2024-01-02T03:05:06.000Z",
                                    }
                                },
                            ],
                        }
                    }
                ],
            ),
        },
    )

    messages = list(load_x_archive(archive_path))

    assert messages == [
        {
            "conversation_id": "7bd738332aa8d7c7aca5d2df",
            "sender_role": "other",
            "timestamp": 1704164645,
            "text": "Are you free tomorrow?",
            "source": "x_dm",
            "medium": "dm",
        },
        {
            "conversation_id": "7bd738332aa8d7c7aca5d2df",
            "sender_role": "me",
            "timestamp": 1704164706,
            "text": "yeah after 3 works",
            "source": "x_dm",
            "medium": "dm",
        },
    ]


def test_load_x_archive_includes_group_dms_and_redacts_contact_details(tmp_path: Path) -> None:
    archive_path = _write_x_archive(
        tmp_path,
        {
            "account.js": ("account", [{"account": {"accountId": "111"}}]),
            "direct-messages-group.js": (
                "direct_messages_group",
                [
                    {
                        "dmConversation": {
                            "conversationId": "group-777",
                            "messages": [
                                {
                                    "messageCreate": {
                                        "senderId": "333",
                                        "text": (
                                            "Email Me@Example.COM or call +1 (415) 555-0123; "
                                            "docs at https://example.com/a?b=1."
                                        ),
                                        "createdAt": "2024-02-03T04:05:06Z",
                                    }
                                },
                                {
                                    "messageCreate": {
                                        "senderId": "111",
                                        "text": "https://example.com/attachment-only",
                                        "createdAt": "2024-02-03T04:06:06Z",
                                    }
                                },
                            ],
                        }
                    }
                ],
            ),
        },
    )

    assert list(load_x_archive(archive_path)) == [
        {
            "conversation_id": "22f5ad4e786fb3d58d378056",
            "sender_role": "other",
            "timestamp": 1706933106,
            "text": ("Email <|email_address|> or call <|phone|>; docs at <|url|>."),
            "source": "x_dm",
            "medium": "dm",
        }
    ]


def test_load_x_public_replies_returns_only_authored_replies(tmp_path: Path) -> None:
    archive_path = _write_x_archive(
        tmp_path,
        {
            "account.js": ("account", [{"account": {"accountId": "111"}}]),
            "tweets.js": (
                "tweets",
                [
                    {
                        "tweet": {
                            "id_str": "901",
                            "in_reply_to_status_id_str": "900",
                            "in_reply_to_user_id_str": "222",
                            "full_text": (
                                "@friend yep email owner@example.com "
                                "or check https://example.com/private"
                            ),
                            "created_at": "Sat Feb 03 04:05:06 +0000 2024",
                        }
                    },
                    {
                        "tweet": {
                            "id_str": "902",
                            "in_reply_to_status_id_str": None,
                            "full_text": "A standalone public post",
                            "created_at": "Sun Feb 04 04:05:06 +0000 2024",
                        }
                    },
                ],
            ),
        },
    )

    assert list(load_x_public_replies(archive_path)) == [
        {
            "conversation_id": "faf05d5e319124a7bb52f2d2",
            "sender_role": "me",
            "timestamp": 1706933106,
            "text": "@friend yep email <|email_address|> or check <|url|>",
            "source": "x_reply",
            "medium": "public",
        }
    ]
