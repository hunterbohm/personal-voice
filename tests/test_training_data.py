import json
import os
import subprocess
import sys
from pathlib import Path

from personal_voice.training_data import preflight_renderer_dataset, renderer_dataset_identity


class WordTokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        words = sum(len(str(message.get("content") or "").split()) for message in messages)
        return list(range(words + len(messages) * 2))


def _write_split(path: Path, target: str) -> None:
    record = {
        "messages": [
            {"role": "system", "content": "rewrite"},
            {"role": "user", "content": "canonical draft"},
            {"role": "assistant", "content": target},
        ]
    }
    path.write_text(json.dumps(record) + "\n")
    path.chmod(0o600)


def test_preflight_renderer_dataset_detects_truncation_without_returning_text(
    tmp_path: Path,
) -> None:
    data = tmp_path / "private-data"
    data.mkdir(mode=0o700)
    _write_split(data / "train.jsonl", "one two three four")
    _write_split(data / "valid.jsonl", "one two three four")
    _write_split(data / "test.jsonl", " ".join(["word"] * 30))

    result = preflight_renderer_dataset(
        data,
        WordTokenizer(),
        max_seq_length=20,
        min_supervised_tokens=4,
    )

    assert result["ok"] is False
    assert result["too_long"] == 1
    assert result["target_too_short"] == 0
    assert "word" not in json.dumps(result)


def test_renderer_dataset_identity_binds_all_private_splits(tmp_path: Path) -> None:
    data = tmp_path / "private-data"
    data.mkdir(mode=0o700)
    for split in ("train", "valid", "test"):
        _write_split(data / f"{split}.jsonl", "one two three four")

    first_hash, first_records = renderer_dataset_identity(data)
    _write_split(data / "test.jsonl", "changed target words here")
    second_hash, second_records = renderer_dataset_identity(data)

    assert first_records == second_records == 3
    assert first_hash != second_hash


def test_training_wrapper_rejects_repo_local_adapter_before_writing_weights(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    data = tmp_path / "private-data"
    data.mkdir(mode=0o700)
    for split in ("train", "valid", "test"):
        _write_split(data / f"{split}.jsonl", "one two three four")
    config = tmp_path / "config.yaml"
    config.write_text("max_seq_length: 512\n")
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / ".git").mkdir()
    adapter = repository / "private-adapter"
    script = Path(__file__).resolve().parents[1] / "scripts/train_renderer.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--model",
            str(model),
            "--data",
            str(data),
            "--adapter",
            str(adapter),
            "--config",
            str(config),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "outside a Git worktree" in result.stderr
    assert not adapter.exists()


def test_training_wrapper_rejects_a_resident_ollama_model_before_loading_data(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    data = tmp_path / "private-data"
    data.mkdir(mode=0o700)
    for split in ("train", "valid", "test"):
        _write_split(data / f"{split}.jsonl", "one two three four")
    config = tmp_path / "config.yaml"
    config.write_text("max_seq_length: 512\n")
    adapter = tmp_path / "private-adapter"
    binaries = tmp_path / "bin"
    binaries.mkdir()
    ollama = binaries / "ollama"
    ollama.write_text(
        "#!/bin/sh\n"
        "printf 'NAME ID SIZE PROCESSOR CONTEXT UNTIL\\n'\n"
        "printf 'local-model synthetic 8GB GPU 32768 4m\\n'\n"
    )
    ollama.chmod(0o700)
    script = Path(__file__).resolve().parents[1] / "scripts/train_renderer.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--model",
            str(model),
            "--data",
            str(data),
            "--adapter",
            str(adapter),
            "--config",
            str(config),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PATH": f"{binaries}{os.pathsep}{os.environ['PATH']}"},
    )

    assert result.returncode != 0
    assert "unload resident Ollama models" in result.stderr
    assert not adapter.exists()
