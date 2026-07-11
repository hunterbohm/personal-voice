import io
import json
import sqlite3
from pathlib import Path

from personal_voice.renderer_data import (
    OllamaNeutralizer,
    _neutralization_error,
    _reply_signature,
    export_renderer_dataset,
    neutralize_renderer_workspace,
    prepare_renderer_workspace,
    renderer_workspace_stats,
)
from personal_voice.store import build_store


def test_ollama_neutralizer_uses_guarded_loopback_transport(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_open_loopback(request, *, path, timeout):
        captured["path"] = path
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data)
        response = {"response": json.dumps({"drafts": [{"index": 0, "draft": "Yes, that works."}]})}
        return io.BytesIO(json.dumps(response).encode())

    monkeypatch.setattr(
        "personal_voice.renderer_data.open_loopback_http_request",
        fake_open_loopback,
    )
    neutralizer = OllamaNeutralizer()

    assert neutralizer.neutralize_batch([("text", "yeah that works", 1)]) == ["Yes, that works."]
    assert captured["path"] == "/api/generate"
    assert captured["timeout"] == 180
    assert captured["payload"]["model"] == "qwen3.5:9b"


def _example(
    example_id: str,
    medium: str,
    conversation: str,
    timestamp: int,
    reply: str,
) -> dict:
    return {
        "example_id": example_id,
        "source": {"text": "imessage", "dm": "x_dm", "email": "gmail", "public": "x_reply"}[medium],
        "medium": medium,
        "conversation_id": conversation,
        "timestamp": timestamp,
        "incoming_text": "synthetic incoming context",
        "context_text": "other: synthetic incoming context",
        "reply_text": reply,
    }


def test_reply_signature_ignores_punctuation_only_style_variants() -> None:
    assert _reply_signature("yeah, that works!!") == _reply_signature("Yeah that works 😂")


def test_neutralization_rejects_reordered_literal_associations() -> None:
    assert (
        _neutralization_error(
            "The first review is at 3:30 and the second is at 5:00.",
            "The second review is at 5:00 and the first is at 3:30.",
        )
        == "literal_order"
    )


def test_prepare_renderer_workspace_is_private_balanced_and_leak_free(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus-private" / "corpus.sqlite"
    rows = []
    for medium in ("text", "dm", "email", "public"):
        for conversation in range(120):
            topic = chr(97 + conversation // 26) + chr(97 + conversation % 26)
            rows.append(
                _example(
                    f"{medium}-{conversation}-long",
                    medium,
                    f"{medium}-conversation-{conversation}",
                    conversation,
                    f"A distinct synthetic {medium} reply about topic{topic} with useful detail.",
                )
            )
            rows.append(
                _example(
                    f"{medium}-{conversation}-short",
                    medium,
                    f"{medium}-conversation-{conversation}",
                    conversation + 100,
                    f"sounds good topic{topic}",
                )
            )
    build_store(corpus, rows)
    workspace = tmp_path / "renderer-private" / "workspace.sqlite"

    stats = prepare_renderer_workspace(
        corpus,
        workspace,
        per_medium={"text": 100, "dm": 100, "email": 100, "public": 100},
        conversation_cap=2,
        identity_fraction=0.05,
        seed="synthetic-v1",
    )

    assert stats == renderer_workspace_stats(workspace)
    assert stats["total"] <= 400
    assert set(stats["mediums"]) == {"dm", "email", "public", "text"}
    assert stats["conversation_leaks"] == 0
    assert stats["canonical_leaks"] == 0
    assert stats["reply_leaks"] == 0
    assert 0.20 <= stats["short_fraction"] <= 0.30
    assert all(values["valid"] >= 3 for values in stats["mediums"].values())
    assert all(values["test"] >= 3 for values in stats["mediums"].values())
    assert 0 < stats["neutralization"]["ready"] < stats["total"]
    assert workspace.parent.stat().st_mode & 0o077 == 0
    assert workspace.stat().st_mode & 0o077 == 0


def test_neutralize_renderer_workspace_rejects_literal_changes_and_can_retry(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus-private" / "corpus.sqlite"
    build_store(
        corpus,
        [
            _example(
                "reply-one",
                "text",
                "conversation-one",
                1,
                "yeah 3 works and i'll send the final version by Friday",
            )
        ],
    )
    workspace = tmp_path / "renderer-private" / "workspace.sqlite"
    prepare_renderer_workspace(
        corpus,
        workspace,
        per_medium={"text": 1},
        seed="synthetic-v1",
    )

    failed = neutralize_renderer_workspace(
        workspace,
        lambda _medium, _target, _attempt: "That works and it will be sent by Friday.",
        max_attempts=1,
    )

    assert failed["neutralization"] == {
        "excluded": 0,
        "failed": 1,
        "pending": 0,
        "ready": 0,
    }

    recovered = neutralize_renderer_workspace(
        workspace,
        lambda _medium, _target, _attempt: "3 works, and I will send the final version by Friday.",
        max_attempts=1,
        retry_failed=True,
    )

    assert recovered["neutralization"] == {
        "excluded": 0,
        "failed": 0,
        "pending": 0,
        "ready": 1,
    }


def test_neutralize_renderer_workspace_uses_batch_teacher_when_available(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus-private" / "corpus.sqlite"
    build_store(
        corpus,
        [
            _example(
                f"reply-{index}",
                "public",
                f"conversation-{index}",
                index,
                f"yeah this synthetic topic{chr(97 + index)} makes a lot more sense now",
            )
            for index in range(4)
        ],
    )
    workspace = tmp_path / "renderer-private" / "workspace.sqlite"
    prepare_renderer_workspace(corpus, workspace, per_medium={"public": 4})

    class BatchTeacher:
        calls = 0

        def neutralize_batch(self, items):
            self.calls += 1
            return [target.replace("yeah", "Yes,") for _medium, target, _attempt in items]

    teacher = BatchTeacher()
    stats = neutralize_renderer_workspace(workspace, teacher, batch_size=4)

    assert teacher.calls == 1
    assert stats["neutralization"] == {
        "excluded": 0,
        "failed": 0,
        "pending": 0,
        "ready": 4,
    }


def test_export_excludes_canonical_drafts_that_collapse_across_splits(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus-private" / "corpus.sqlite"
    build_store(
        corpus,
        [
            _example(
                "reply-one",
                "text",
                "conversation-one",
                1,
                "yeah this plan works for me today",
            ),
            _example(
                "reply-two",
                "text",
                "conversation-two",
                2,
                "totally this plan is fine for me today",
            ),
        ],
    )
    workspace = tmp_path / "renderer-private" / "workspace.sqlite"
    prepare_renderer_workspace(corpus, workspace, per_medium={"text": 2})
    connection = sqlite3.connect(workspace)
    try:
        record_ids = [row[0] for row in connection.execute("SELECT record_id FROM pairs")]
        connection.execute("UPDATE pairs SET split = 'train' WHERE record_id = ?", (record_ids[0],))
        connection.execute("UPDATE pairs SET split = 'test' WHERE record_id = ?", (record_ids[1],))
        connection.commit()
    finally:
        connection.close()
    neutralize_renderer_workspace(
        workspace,
        lambda _medium, _target, _attempt: "This plan works for me today.",
    )
    assert renderer_workspace_stats(workspace)["canonical_leaks"] == 1

    exported = export_renderer_dataset(workspace, tmp_path / "dataset-private")
    stats = renderer_workspace_stats(workspace)

    assert exported["total"] == 1
    assert stats["canonical_leaks"] == 0
    assert stats["neutralization"]["excluded"] == 1


def test_export_renderer_dataset_writes_private_prompt_maskable_chat_records(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus-private" / "corpus.sqlite"
    target = "yeah that works and I can review the final version at 3:30 tomorrow"
    build_store(corpus, [_example("reply-one", "dm", "conversation-one", 1, target)])
    workspace = tmp_path / "renderer-private" / "workspace.sqlite"
    prepare_renderer_workspace(corpus, workspace, per_medium={"dm": 1})
    neutralize_renderer_workspace(
        workspace,
        lambda _medium, _target, _attempt: (
            "That works, and I can review the final version at 3:30 tomorrow."
        ),
    )
    output = tmp_path / "dataset-private"

    stats = export_renderer_dataset(workspace, output)

    assert sum(stats["splits"].values()) == 1
    records = []
    for path in output.glob("*.jsonl"):
        records.extend(json.loads(line) for line in path.read_text().splitlines())
        assert path.stat().st_mode & 0o077 == 0
    assert output.stat().st_mode & 0o077 == 0
    assert records == [
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Rewrite the complete canonical draft in the writer's learned voice. "
                        "Preserve its meaning and every literal. Do not add facts, promises, "
                        "questions, permissions, deadlines, or commitments. Return exactly "
                        "PV_RENDERED_START, a newline, the rewritten draft, a newline, and "
                        "PV_RENDERED_END, with nothing else."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Medium: dm\nCanonical draft:\n"
                        "That works, and I can review the final version at "
                        "[[PV_LITERAL_0]] tomorrow."
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "PV_RENDERED_START\nyeah that works and I can review the final version "
                        "at [[PV_LITERAL_0]] tomorrow\nPV_RENDERED_END"
                    ),
                },
            ]
        }
    ]
    assert "conversation-one" not in json.dumps(records)
