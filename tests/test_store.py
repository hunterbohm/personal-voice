import stat

import pytest

from personal_voice.store import _resolve_medium, build_store, retrieve_examples, store_stats


def example(
    example_id: str,
    *,
    source: str,
    medium: str,
    incoming: str,
    reply: str,
    timestamp: int,
) -> dict:
    return {
        "example_id": example_id,
        "source": source,
        "medium": medium,
        "conversation_id": f"conversation-{example_id}",
        "timestamp": timestamp,
        "incoming_text": incoming,
        "context_text": f"other: {incoming}" if incoming else "",
        "reply_text": reply,
    }


def test_build_store_is_private_and_reports_only_aggregate_counts(tmp_path) -> None:
    database = tmp_path / "private" / "corpus.sqlite"
    examples = [
        example(
            "email-a",
            source="gmail",
            medium="email",
            incoming="Could you review the proposal?",
            reply="yeah I can look today",
            timestamp=10,
        ),
        example(
            "text-a",
            source="imessage",
            medium="text",
            incoming="want to meet tomorrow?",
            reply="yeah tomorrow works",
            timestamp=20,
        ),
        example(
            "oversized",
            source="gmail",
            medium="email",
            incoming="newsletter draft",
            reply="x" * 4001,
            timestamp=30,
        ),
    ]

    assert build_store(database, examples) == {
        "mediums": {"email": 1, "text": 1},
        "sources": {"gmail": 1, "imessage": 1},
        "total": 2,
    }
    assert store_stats(database) == {
        "mediums": {"email": 1, "text": 1},
        "sources": {"gmail": 1, "imessage": 1},
        "total": 2,
    }
    assert stat.S_IMODE(database.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(database.stat().st_mode) == 0o600


def test_retrieve_boosts_the_destination_then_diversifies_and_dedupes(tmp_path) -> None:
    database = tmp_path / "corpus.sqlite"
    build_store(
        database,
        [
            example(
                "email-a",
                source="gmail",
                medium="email",
                incoming="Could you review the launch proposal?",
                reply="yeah send it over and I can look today",
                timestamp=10,
            ),
            example(
                "public-a",
                source="x_reply",
                medium="public",
                incoming="What do you think of the launch?",
                reply="yeah this is exactly it",
                timestamp=20,
            ),
            example(
                "text-a",
                source="imessage",
                medium="text",
                incoming="can you review the launch tomorrow?",
                reply="yeah tomorrow works",
                timestamp=30,
            ),
            example(
                "text-duplicate",
                source="imessage",
                medium="text",
                incoming="review launch timing",
                reply="yeah tomorrow works",
                timestamp=40,
            ),
        ],
    )

    matches = retrieve_examples(
        database,
        "reply to this email about reviewing the launch proposal",
        medium="auto",
        limit=3,
    )

    assert matches[0]["medium"] == "email"
    assert len(matches) == 3
    assert len({match["source"] for match in matches}) == 3
    assert len({match["reply_text"] for match in matches}) == 3


def test_retrieve_falls_back_to_relevant_medium_when_words_do_not_match(tmp_path) -> None:
    database = tmp_path / "corpus.sqlite"
    build_store(
        database,
        [
            example(
                "email-a",
                source="gmail",
                medium="email",
                incoming="old unrelated question",
                reply="sounds good to me",
                timestamp=10,
            ),
            example(
                "text-a",
                source="imessage",
                medium="text",
                incoming="another old thing",
                reply="yep",
                timestamp=20,
            ),
        ],
    )

    matches = retrieve_examples(database, "quasar nebula", medium="email", limit=2)

    assert [match["medium"] for match in matches] == ["email", "text"]


def test_medium_is_a_boost_not_a_relevance_override(tmp_path) -> None:
    database = tmp_path / "corpus.sqlite"
    build_store(
        database,
        [
            example(
                "email-a",
                source="gmail",
                medium="email",
                incoming="launch",
                reply="sounds good",
                timestamp=10,
            ),
            example(
                "public-a",
                source="x_reply",
                medium="public",
                incoming="launch proposal review schedule tomorrow",
                reply="yeah I would change the timing",
                timestamp=20,
            ),
        ],
    )

    matches = retrieve_examples(
        database,
        "email reply about the launch proposal review schedule tomorrow",
        medium="auto",
        limit=2,
    )

    assert matches[0]["source"] == "x_reply"


def test_comparable_matches_favor_the_inferred_destination(tmp_path) -> None:
    database = tmp_path / "corpus.sqlite"
    rows = []
    for index in range(4):
        rows.append(
            example(
                f"public-{index}",
                source="x_reply",
                medium="public",
                incoming="ai product launch update",
                reply=f"public wording {index}",
                timestamp=10 + index,
            )
        )
        rows.append(
            example(
                f"dm-{index}",
                source="x_dm",
                medium="dm",
                incoming="ai product launch update",
                reply=f"dm wording {index}",
                timestamp=20 + index,
            )
        )
    build_store(database, rows)

    matches = retrieve_examples(
        database,
        "write an X reply about an ai product launch update",
        medium="auto",
        limit=4,
    )

    assert sum(match["medium"] == "public" for match in matches) >= 3


def test_build_store_does_not_repermission_an_existing_shared_directory(tmp_path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)

    with pytest.raises(PermissionError, match="private directory"):
        build_store(shared / "corpus.sqlite", [])

    assert stat.S_IMODE(shared.stat().st_mode) == 0o755


def test_retrieved_context_starts_at_a_complete_labeled_turn(tmp_path) -> None:
    database = tmp_path / "corpus.sqlite"
    row = example(
        "long-context",
        source="imessage",
        medium="text",
        incoming="latest question",
        reply="latest reply",
        timestamp=10,
    )
    row["context_text"] = "other: " + ("x" * 9000) + "\nme: short prior turn"
    build_store(database, [row])

    match = retrieve_examples(database, "latest question", limit=1)[0]

    assert match["context_text"] == "me: short prior turn"


def test_auto_medium_detection_uses_intent_words_not_substrings() -> None:
    assert _resolve_medium("auto", "please postpone our meeting") == ""
    assert _resolve_medium("auto", "draft commentary for Sarah") == ""
    assert _resolve_medium("auto", "DM Alex about tomorrow") == "dm"
    assert _resolve_medium("auto", "write an X post about this") == "public"
