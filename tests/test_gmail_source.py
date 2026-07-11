import base64
import json

from personal_voice.sources.gmail import load_gmail_threads


def _gmail_data(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


def test_load_gmail_threads_normalizes_a_combined_sanitized_export(tmp_path) -> None:
    source = tmp_path / "gmail-threads.json"
    source.write_text(
        json.dumps(
            {
                "threads": [
                    {
                        "id": "private-thread-id",
                        "messages": [
                            {
                                "id": "message-in",
                                "threadId": "private-thread-id",
                                "internalDate": "1704164645000",
                                "headers": {
                                    "from": "Alex Person <alex@example.com>",
                                    "date": "Tue, 2 Jan 2024 03:04:05 +0000",
                                },
                                "body": (
                                    "want to talk tomorrow?\n\n"
                                    "On Mon, Jan 1, 2024 at 8:00 PM Writer wrote:\n"
                                    "> an older message"
                                ),
                            },
                            {
                                "id": "message-out",
                                "threadId": "private-thread-id",
                                "internalDate": 1704164706000,
                                "headers": {
                                    "from": "Writer <WRITER@EXAMPLE.COM>",
                                    "date": "Tue, 2 Jan 2024 03:05:06 +0000",
                                },
                                "body": "yeah after 3 works\n\n-- \nWriter\nFounder, Example",
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    messages = load_gmail_threads(source, {"writer@example.com"})

    assert messages == [
        {
            "conversation_id": "aa14ad2bf6331841a4c05b87",
            "sender_role": "other",
            "timestamp": 1704164645,
            "text": "want to talk tomorrow?",
            "source": "gmail",
            "medium": "email",
        },
        {
            "conversation_id": "aa14ad2bf6331841a4c05b87",
            "sender_role": "me",
            "timestamp": 1704164706,
            "text": "yeah after 3 works",
            "source": "gmail",
            "medium": "email",
        },
    ]


def test_load_gmail_threads_accepts_one_sanitized_thread_and_cleans_html(tmp_path) -> None:
    source = tmp_path / "one-thread.json"
    source.write_text(
        json.dumps(
            {
                "thread": {
                    "id": "single-thread-id",
                    "messages": [
                        {
                            "headers": {
                                "From": "Writer <Writer@Example.com>",
                                "Date": "Wed, 3 Jan 2024 04:05:06 +0000",
                            },
                            "body": (
                                "<p>sounds good</p><p>send it over</p>"
                                "<blockquote><p>older quoted message</p></blockquote>"
                            ),
                        }
                    ],
                },
                "downloaded": [],
            }
        ),
        encoding="utf-8",
    )

    assert load_gmail_threads(source, {"writer@example.com"}) == [
        {
            "conversation_id": "42b1d863788700fd03ad9fc6",
            "sender_role": "me",
            "timestamp": 1704254706,
            "text": "sounds good\nsend it over",
            "source": "gmail",
            "medium": "email",
        }
    ]


def test_load_gmail_threads_cleans_legacy_html_tags(tmp_path) -> None:
    source = tmp_path / "legacy-html.json"
    source.write_text(
        json.dumps(
            {
                "thread": {
                    "id": "legacy-html-thread",
                    "messages": [
                        {
                            "from": "writer@example.com",
                            "internalDate": "1704254706000",
                            "body": '<font color="#333">sounds good</font>',
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    assert load_gmail_threads(source, {"writer@example.com"})[0]["text"] == "sounds good"


def test_load_gmail_threads_decodes_raw_gmail_payloads(tmp_path) -> None:
    source = tmp_path / "raw-thread.json"
    source.write_text(
        json.dumps(
            {
                "thread": {
                    "id": "raw-thread-id",
                    "messages": [
                        {
                            "internalDate": "1704341107000",
                            "payload": {
                                "mimeType": "multipart/alternative",
                                "headers": [
                                    {
                                        "name": "From",
                                        "value": "Client <client@example.com>",
                                    },
                                    {
                                        "name": "Date",
                                        "value": "Thu, 4 Jan 2024 04:05:07 +0000",
                                    },
                                ],
                                "parts": [
                                    {
                                        "mimeType": "text/plain",
                                        "body": {
                                            "data": _gmail_data(
                                                "could you take a look?\n\n"
                                                "---------- Forwarded message ---------\n"
                                                "From: Someone Else <else@example.com>\n"
                                                "an older message"
                                            )
                                        },
                                    },
                                    {
                                        "mimeType": "text/html",
                                        "body": {"data": _gmail_data("<p>html alternative</p>")},
                                    },
                                ],
                            },
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    assert load_gmail_threads(source, {"writer@example.com"}) == [
        {
            "conversation_id": "aa05cccffa0fb16581abef07",
            "sender_role": "other",
            "timestamp": 1704341107,
            "text": "could you take a look?",
            "source": "gmail",
            "medium": "email",
        }
    ]


def test_load_gmail_threads_uses_direct_fields_and_skips_non_messages(tmp_path) -> None:
    source = tmp_path / "mixed-thread.json"
    source.write_text(
        json.dumps(
            {
                "threads": [
                    {
                        "messages": [
                            {
                                "threadId": "direct-thread-id",
                                "from": "WRITER@example.com",
                                "date": "Fri, 5 Jan 2024 04:05:08 +0000",
                                "text": "got it\n\nSent from my iPhone",
                            },
                            {
                                "threadId": "direct-thread-id",
                                "from": "Friend <friend@example.com>",
                                "internalDate": 1704427569,
                                "body": (
                                    '<span>thanks</span><span class="gmail_quote">'
                                    "older quoted message</span>"
                                ),
                            },
                            {
                                "threadId": "direct-thread-id",
                                "from": "friend@example.com",
                                "body": "[Attachment: photo.jpg]",
                                "attachments": [{"filename": "photo.jpg"}],
                            },
                            {
                                "threadId": "direct-thread-id",
                                "payload": {
                                    "headers": [{"name": "From", "value": "friend@example.com"}],
                                    "mimeType": "application/pdf",
                                    "body": {"attachmentId": "attachment-id"},
                                },
                            },
                            {
                                "threadId": "direct-thread-id",
                                "internalDate": "1704427570000",
                                "body": "internal export status row",
                            },
                            {
                                "from": "friend@example.com",
                                "internalDate": "1704427571000",
                                "body": "message without a conversation key",
                            },
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert load_gmail_threads(source, {"writer@example.com"}) == [
        {
            "conversation_id": "abfc2228689b4b1510d2e524",
            "sender_role": "me",
            "timestamp": 1704427508,
            "text": "got it",
            "source": "gmail",
            "medium": "email",
        },
        {
            "conversation_id": "abfc2228689b4b1510d2e524",
            "sender_role": "other",
            "timestamp": 1704427569,
            "text": "thanks",
            "source": "gmail",
            "medium": "email",
        },
    ]


def test_load_gmail_threads_accepts_a_composio_message_page(tmp_path) -> None:
    source = tmp_path / "composio-page.json"
    source.write_text(
        json.dumps(
            {
                "successful": True,
                "data": {
                    "messages": [
                        {
                            "messageId": "composio-message-id",
                            "threadId": "composio-thread-id",
                            "messageText": "yep that works\n\n> previous quoted message",
                            "messageTimestamp": "2024-01-06T04:05:10Z",
                            "sender": "Writer <Writer@Example.com>",
                            "to": "friend@example.com",
                            "subject": "Re: timing",
                            "labelIds": ["SENT"],
                            "payload": {},
                        },
                        {
                            "messageId": "attachment-message-id",
                            "threadId": "composio-thread-id",
                            "messageText": "[Attachment]",
                            "messageTimestamp": "2024-01-06T04:05:11Z",
                            "sender": "friend@example.com",
                        },
                    ],
                    "nextPageToken": "opaque-next-page-token",
                },
            }
        ),
        encoding="utf-8",
    )

    assert load_gmail_threads(source, {"writer@example.com"}) == [
        {
            "conversation_id": "bad4556de451cdd76485a360",
            "sender_role": "me",
            "timestamp": 1704513910,
            "text": "yep that works",
            "source": "gmail",
            "medium": "email",
            "context_hint": "subject: Re: timing",
        }
    ]


def test_load_gmail_threads_accepts_a_gog_message_page(tmp_path) -> None:
    source = tmp_path / "gog-page.json"
    source.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "id": "message-id",
                        "threadId": "thread-id",
                        "from": "Writer <writer@example.com>",
                        "date": "Sun, 7 Jan 2024 04:05:10 +0000",
                        "body": "sent from a synthetic gog export",
                    }
                ],
                "nextPageToken": "",
            }
        ),
        encoding="utf-8",
    )

    assert load_gmail_threads(source, {"writer@example.com"}) == [
        {
            "conversation_id": "7fe627f6526e9dfa0e802f3e",
            "sender_role": "me",
            "timestamp": 1704600310,
            "text": "sent from a synthetic gog export",
            "source": "gmail",
            "medium": "email",
        }
    ]
