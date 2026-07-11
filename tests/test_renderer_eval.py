import json
import sqlite3
from pathlib import Path

import pytest

from personal_voice.renderer import (
    RENDER_SYSTEM_PROMPT,
    format_renderer_target,
    render_user_prompt,
)
from personal_voice.renderer_data import (
    neutralize_renderer_workspace,
    prepare_renderer_workspace,
)
from personal_voice.renderer_eval import evaluate_renderer_dataset, evaluate_renderer_workspace
from personal_voice.store import build_store


class SyntheticBackend:
    version = "synthetic-style-v1"

    def generate(self, _medium: str, draft: str) -> str:
        return format_renderer_target(
            draft.replace("That works", "yeah that works").replace(
                "I can review", "yeah I can review"
            )
        )


class PartiallyUnavailableBackend:
    version = "synthetic-partial-v1"

    def generate(self, medium: str, draft: str) -> str:
        if medium == "dm":
            raise RuntimeError("synthetic unavailable row")
        return format_renderer_target(draft.replace("That works", "yeah that works"))


class PartiallyRejectedBackend:
    version = "synthetic-rejected-v1"

    def generate(self, medium: str, draft: str) -> str:
        if medium == "dm":
            return draft
        return format_renderer_target(draft.replace("That works", "yeah that works"))


def test_evaluate_renderer_workspace_reports_aggregate_style_delta_only(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus-private" / "corpus.sqlite"
    build_store(
        corpus,
        [
            {
                "example_id": "one",
                "source": "imessage",
                "medium": "text",
                "conversation_id": "conversation-one",
                "timestamp": 1,
                "incoming_text": "synthetic",
                "context_text": "other: synthetic",
                "reply_text": "yeah that works for me today",
            },
            {
                "example_id": "two",
                "source": "x_dm",
                "medium": "dm",
                "conversation_id": "conversation-two",
                "timestamp": 2,
                "incoming_text": "synthetic",
                "context_text": "other: synthetic",
                "reply_text": "yeah I can review that tomorrow",
            },
        ],
    )
    workspace = tmp_path / "renderer-private" / "workspace.sqlite"
    prepare_renderer_workspace(
        corpus,
        workspace,
        per_medium={"text": 1, "dm": 1},
    )
    neutralize_renderer_workspace(
        workspace,
        lambda medium, _target, _attempt: (
            "That works for me today." if medium == "text" else "I can review that tomorrow."
        ),
    )

    results = [
        evaluate_renderer_workspace(
            workspace,
            SyntheticBackend(),
            split=split,
            limit_per_medium=10,
        )
        for split in ("train", "valid", "test")
    ]

    assert sum(result["total"] for result in results) == 2
    assert sum(result["statuses"]["rendered"] for result in results) == 2
    assert sum(result["closer_to_target"] for result in results) == 2
    assert all(result["statuses"]["rejected"] == 0 for result in results)
    assert any(result["mean_similarity_delta"] > 0 for result in results)
    assert set().union(*(result["mediums"] for result in results)) == {"dm", "text"}
    assert all("text" not in result for result in results)


@pytest.mark.parametrize(
    ("backend", "failure_status"),
    (
        (PartiallyUnavailableBackend(), "unavailable"),
        (PartiallyRejectedBackend(), "rejected"),
    ),
)
def test_evaluation_uses_every_heldout_row_as_the_similarity_denominator(
    tmp_path: Path,
    backend: PartiallyUnavailableBackend | PartiallyRejectedBackend,
    failure_status: str,
) -> None:
    workspace = tmp_path / "renderer-private" / "workspace.sqlite"
    workspace.parent.mkdir(mode=0o700)
    connection = sqlite3.connect(workspace)
    try:
        connection.execute(
            """
            CREATE TABLE pairs (
                record_id TEXT PRIMARY KEY,
                medium TEXT NOT NULL,
                neutral_text TEXT,
                target_text TEXT NOT NULL,
                split TEXT NOT NULL,
                neutral_status TEXT NOT NULL
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO pairs(record_id, medium, neutral_text, target_text, split, neutral_status)
            VALUES (?, ?, ?, ?, 'test', 'ready')
            """,
            [
                (
                    "one",
                    "text",
                    "That works for me today.",
                    "yeah that works for me today",
                ),
                (
                    "two",
                    "dm",
                    "I can review that tomorrow.",
                    "yeah I can review that tomorrow",
                ),
            ],
        )
        connection.commit()
    finally:
        connection.close()

    all_rows = evaluate_renderer_workspace(workspace, backend)
    connection = sqlite3.connect(workspace)
    try:
        connection.execute("DELETE FROM pairs WHERE medium = 'dm'")
        connection.commit()
    finally:
        connection.close()
    rendered_row_only = evaluate_renderer_workspace(workspace, backend)

    assert all_rows["total"] == 2
    assert all_rows["statuses"] == {
        "rendered": 1,
        "rejected": int(failure_status == "rejected"),
        "unavailable": int(failure_status == "unavailable"),
    }
    assert all_rows["mean_similarity_delta"] == round(
        rendered_row_only["mean_similarity_delta"] / 2,
        6,
    )


def test_evaluate_renderer_dataset_reads_the_exact_exported_test_records(tmp_path: Path) -> None:
    data = tmp_path / "renderer-private" / "mlx"
    data.mkdir(mode=0o700, parents=True)
    for split in ("train", "valid"):
        path = data / f"{split}.jsonl"
        path.write_text("")
        path.chmod(0o600)
    test = data / "test.jsonl"
    test.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": RENDER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": render_user_prompt(
                            "text", "That works for me at [[PV_LITERAL_0]]."
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": format_renderer_target(
                            "yeah that works for me at [[PV_LITERAL_0]]"
                        ),
                    },
                ]
            }
        )
        + "\n"
    )
    test.chmod(0o600)

    class RecordingBackend(SyntheticBackend):
        seen: list[str] = []

        def generate(self, medium: str, draft: str) -> str:
            self.seen.append(draft)
            return super().generate(medium, draft)

    backend = RecordingBackend()
    result = evaluate_renderer_dataset(data, backend)

    assert result["total"] == 1
    assert result["mediums"] == {"text": 1}
    assert result["statuses"] == {"rendered": 1, "rejected": 0, "unavailable": 0}
    assert result["closer_to_target"] == 1
    assert backend.seen == ["That works for me at [[PV_LITERAL_0]]."]
    assert "text" not in result
