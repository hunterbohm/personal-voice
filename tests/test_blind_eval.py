import sqlite3
from pathlib import Path

import pytest

from personal_voice.blind_eval import (
    add_blind_case,
    blind_eval_stats,
    next_blind_case,
    record_blind_vote,
)
from personal_voice.renderer import RENDERER_CONTRACT_VERSION, format_renderer_target


class SyntheticBackend:
    artifact_hash = "synthetic-adapter-hash"
    candidate_hash = "synthetic-candidate-hash"
    version = "synthetic-style-v1"

    def generate(self, _medium: str, draft: str) -> str:
        return format_renderer_target(draft.replace("That works for me.", "yeah that works for me"))


def test_blind_case_requires_retrieval_augmented_draft_attestation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="retrieval_augmented_draft"):
        add_blind_case(
            tmp_path / "private/blind.sqlite",
            "That works for me.",
            "text",
            SyntheticBackend(),
            baseline_source="unverified",
        )


def test_blind_eval_randomizes_candidates_and_scores_hidden_labels(tmp_path: Path) -> None:
    pack = tmp_path / "blind-private" / "evaluation.sqlite"
    baseline = "That works for me."
    tuned = "yeah that works for me"

    added = add_blind_case(
        pack,
        baseline,
        "text",
        SyntheticBackend(),
        baseline_source="retrieval_augmented_draft",
        evaluation_context="Synthetic incoming context for the evaluator only.",
    )
    duplicate = add_blind_case(
        pack,
        baseline,
        "text",
        SyntheticBackend(),
        baseline_source="retrieval_augmented_draft",
        evaluation_context="Different context must not create a repeated comparison.",
    )
    case = next_blind_case(pack)

    assert added["added"] is True
    assert duplicate["added"] is False
    assert case is not None
    assert set((case["candidate_a"], case["candidate_b"])) == {baseline, tuned}
    assert case["evaluation_context"] == "Synthetic incoming context for the evaluator only."
    tuned_choice = "a" if case["candidate_a"] == tuned else "b"
    record_blind_vote(pack, case["case_id"], tuned_choice, would_send="both")

    stats = blind_eval_stats(
        pack,
        model_version="synthetic-style-v1",
        candidate_sha256="synthetic-candidate-hash",
        renderer_contract_version=RENDERER_CONTRACT_VERSION,
    )
    assert stats["activation_eligible"] is False
    assert stats["completed"] == 1
    assert stats["non_ties"] == 1
    assert stats["tuned_wins"] == 1
    assert stats["tuned_non_tie_rate"] == 1.0
    assert stats["tuned_non_tie_wilson_lower_95"] == 0.206549
    assert stats["would_send_tuned"] == 1
    assert stats["would_send_baseline"] == 1
    assert stats["mediums"]["text"]["completed"] == 1
    assert pack.parent.stat().st_mode & 0o077 == 0
    assert pack.stat().st_mode & 0o077 == 0
    assert not blind_eval_stats(
        pack,
        model_version="synthetic-style-v1",
        candidate_sha256="synthetic-candidate-hash",
        renderer_contract_version=2,
    )["activation_eligible"]


def test_blind_activation_is_bound_to_exact_candidate_and_balanced_media(
    tmp_path: Path,
) -> None:
    pack = tmp_path / "blind-private" / "evaluation.sqlite"
    add_blind_case(
        pack,
        "That works for me.",
        "text",
        SyntheticBackend(),
        baseline_source="retrieval_augmented_draft",
    )
    connection = sqlite3.connect(pack)
    try:
        connection.execute("DELETE FROM cases")
        rows = []
        media = ("text", "dm", "email", "public")
        for index in range(200):
            medium = media[index % len(media)]
            tuned_wins = index % 5 != 0
            rows.append(
                (
                    f"eligible-{index}",
                    medium,
                    f"tuned-{index}",
                    f"baseline-{index}",
                    "tuned",
                    "baseline",
                    "completed",
                    "a" if tuned_wins else "b",
                    "a" if tuned_wins else "b",
                    "candidate-v1",
                    "candidate-hash-v1",
                    "retrieval_augmented_draft",
                    "context",
                    index,
                )
            )
        rows.append(
            (
                "untested-new-version",
                "text",
                "a",
                "b",
                "tuned",
                "baseline",
                "pending",
                None,
                None,
                "candidate-v2",
                "candidate-hash-v2",
                "retrieval_augmented_draft",
                "context",
                999,
            )
        )
        connection.executemany(
            """
            INSERT INTO cases(
                case_id, medium, candidate_a, candidate_b, label_a, label_b,
                status, choice, would_send, model_version, candidate_sha256,
                baseline_source, evaluation_context, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()
    finally:
        connection.close()

    eligible = blind_eval_stats(
        pack,
        model_version="candidate-v1",
        candidate_sha256="candidate-hash-v1",
        renderer_contract_version=RENDERER_CONTRACT_VERSION,
    )
    untested = blind_eval_stats(
        pack,
        model_version="candidate-v2",
        candidate_sha256="candidate-hash-v2",
        renderer_contract_version=RENDERER_CONTRACT_VERSION,
    )
    mixed = blind_eval_stats(pack)

    assert eligible["activation_eligible"] is True
    assert eligible["invalid"] == 0
    assert eligible["completed"] == 200
    assert all(values["eligible"] for values in eligible["mediums"].values())
    assert untested["activation_eligible"] is False
    assert mixed["activation_eligible"] is False

    connection = sqlite3.connect(pack)
    try:
        connection.execute(
            """
            INSERT INTO cases(
                case_id, medium, candidate_a, candidate_b, label_a, label_b,
                status, choice, would_send, model_version, candidate_sha256,
                baseline_source, evaluation_context, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "duplicate-comparison",
                "text",
                "tuned-0",
                "baseline-0",
                "tuned",
                "baseline",
                "completed",
                "a",
                "a",
                "candidate-v1",
                "candidate-hash-v1",
                "retrieval_augmented_draft",
                "different context",
                1000,
            ),
        )
        connection.commit()
    finally:
        connection.close()

    duplicated = blind_eval_stats(
        pack,
        model_version="candidate-v1",
        candidate_sha256="candidate-hash-v1",
        renderer_contract_version=RENDERER_CONTRACT_VERSION,
    )
    assert duplicated["invalid"] == 1
    assert duplicated["activation_eligible"] is False
