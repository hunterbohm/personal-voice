import io
import json
from pathlib import Path

import pytest

from personal_voice.renderer_audit import (
    OllamaSemanticJudge,
    audit_renderer_workspace,
    prepare_semantic_audit,
    semantic_audit_stats,
)
from personal_voice.renderer_data import (
    export_renderer_dataset,
    neutralize_renderer_workspace,
    prepare_renderer_workspace,
    renderer_workspace_stats,
)
from personal_voice.store import build_store


class SyntheticJudge:
    def judge_batch(self, items):
        return [
            {
                "category": "uncertainty" if target.startswith("maybe") else "equivalent",
                "equivalent": not target.startswith("maybe"),
            }
            for target, _canonical in items
        ]


class UnavailableJudge:
    def judge_batch(self, _items):
        raise RuntimeError("synthetic local outage")


def test_ollama_judge_reserves_complete_batch_response_budget(monkeypatch) -> None:
    captured: list[dict] = []

    def fake_open_loopback(request, *, path, timeout):
        assert path == "/api/generate"
        assert timeout == 240
        captured.append(json.loads(request.data))
        response = {"response": json.dumps({"verdicts": ["equivalent"] * 16})}
        return io.BytesIO(json.dumps(response).encode())

    monkeypatch.setattr(
        "personal_voice.renderer_audit.open_loopback_http_request",
        fake_open_loopback,
    )
    judge = OllamaSemanticJudge()
    results = judge.judge_batch([("target", "canonical")] * 16)

    worst_case = json.dumps({"verdicts": ["uncertainty"] * 16})
    assert captured[0]["options"]["num_predict"] >= len(worst_case)
    assert len(results) == 16


def _example(example_id: str, reply: str) -> dict:
    return {
        "context_text": "other: synthetic",
        "conversation_id": example_id,
        "example_id": example_id,
        "incoming_text": "synthetic",
        "medium": "text",
        "reply_text": reply,
        "source": "imessage",
        "timestamp": 1,
    }


def test_semantic_audit_excludes_drift_and_unlocks_production_export(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus-private/corpus.sqlite"
    build_store(
        corpus,
        [
            _example("uncertain", "maybe I will send it tomorrow"),
            _example("control", "yeah that works for me today"),
        ],
    )
    workspace = tmp_path / "renderer-private/workspace.sqlite"
    prepare_renderer_workspace(corpus, workspace, per_medium={"text": 2})

    def neutralize(_medium, target, _attempt):
        if target.startswith("maybe"):
            return "I will send it tomorrow."
        return "That works for me today."

    neutralize_renderer_workspace(workspace, neutralize)
    with pytest.raises(RuntimeError, match="semantic audit is incomplete"):
        export_renderer_dataset(
            workspace,
            tmp_path / "blocked-export",
            require_semantic_audit=True,
        )

    queued = prepare_semantic_audit(workspace)
    assert queued["statuses"]["pending"] == 2
    interrupted = audit_renderer_workspace(workspace, UnavailableJudge(), batch_size=2)
    assert interrupted["complete"] is False
    assert interrupted["statuses"]["pending"] == 2
    assert renderer_workspace_stats(workspace)["neutralization"]["excluded"] == 0
    audited = audit_renderer_workspace(workspace, SyntheticJudge(), batch_size=2)

    assert audited["complete"] is True
    assert audited["statuses"]["failed"] == 1
    assert renderer_workspace_stats(workspace)["neutralization"]["excluded"] == 1
    exported = export_renderer_dataset(
        workspace,
        tmp_path / "dataset-private",
        require_semantic_audit=True,
    )
    assert exported["total"] == 1
    assert semantic_audit_stats(workspace)["unchecked_ready"] == 0
