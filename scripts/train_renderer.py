#!/usr/bin/env python3
"""Run a private, telemetry-free MLX QLoRA renderer training job."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from personal_voice.local_only import inside_git_worktree, write_private_json  # noqa: E402
from personal_voice.training_data import (  # noqa: E402
    load_renderer_tokenizer,
    preflight_renderer_dataset,
    renderer_dataset_identity,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "training/renderer-lora.yaml",
    )
    return parser


def _private_directory(path: Path, *, create: bool = False) -> None:
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.chmod(0o700)
    if not path.is_dir() or path.stat().st_mode & 0o077:
        raise SystemExit(f"private directory required: {path}")


def _config_integer(config_text: str, key: str, default: int) -> int:
    match = re.search(rf"^\s*{re.escape(key)}:\s*(\d+)\s*$", config_text, re.MULTILINE)
    return int(match.group(1)) if match else default


def _ollama_model_is_resident() -> bool:
    executable = shutil.which("ollama")
    if not executable:
        return False
    try:
        result = subprocess.run(
            [executable, "ps"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if result.returncode:
        return False
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return len(lines) > 1


def main() -> int:
    args = _parser().parse_args()
    model = args.model.expanduser().resolve()
    data = args.data.expanduser().resolve()
    adapter = args.adapter.expanduser().resolve()
    config = args.config.expanduser().resolve()
    if not model.is_dir() or not config.is_file():
        raise SystemExit("model or training config is missing")
    if inside_git_worktree(adapter):
        raise SystemExit("private adapter must live outside a Git worktree")
    if adapter.exists() and any(adapter.iterdir()):
        raise SystemExit("adapter output directory must be empty")
    if _ollama_model_is_resident():
        raise SystemExit("unload resident Ollama models before MLX training")
    _private_directory(data)
    for split in ("train", "valid", "test"):
        path = data / f"{split}.jsonl"
        if not path.is_file() or path.stat().st_mode & 0o077:
            raise SystemExit(f"private {split} dataset is missing")
    os.environ.update(
        {
            "DO_NOT_TRACK": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_DISABLED": "true",
            "WANDB_MODE": "disabled",
        }
    )
    try:
        config_text = config.read_text(encoding="utf-8")
        max_seq_length = _config_integer(config_text, "max_seq_length", 2048)
        preflight = preflight_renderer_dataset(
            data,
            load_renderer_tokenizer(model),
            max_seq_length=max_seq_length,
        )
        dataset_sha256, dataset_records = renderer_dataset_identity(data)
        if dataset_records != int(preflight["total"]):
            raise ValueError("renderer dataset identity count does not match preflight")
    except (OSError, TypeError, ValueError) as error:
        raise SystemExit("renderer dataset token preflight failed") from error
    batch_size = _config_integer(config_text, "batch_size", 4)
    iterations = _config_integer(config_text, "iters", 1000)
    accumulation = _config_integer(config_text, "grad_accumulation_steps", 1)
    train_examples = int(preflight["splits"]["train"]["total"])
    training_plan = {
        "approximate_epochs": round(iterations * batch_size / max(1, train_examples), 3),
        "example_exposures": iterations * batch_size,
        "optimizer_updates": iterations // accumulation,
        "train_examples": train_examples,
    }
    print(
        json.dumps(
            {"dataset_preflight": preflight, "training_plan": training_plan},
            sort_keys=True,
        ),
        flush=True,
    )
    if not preflight["ok"]:
        raise SystemExit("renderer dataset would truncate supervised targets")
    _private_directory(adapter, create=True)
    executable = shutil.which("mlx_lm.lora")
    if not executable:
        raise SystemExit("mlx_lm.lora is unavailable")

    command = [
        executable,
        "--model",
        str(model),
        "--data",
        str(data),
        "--adapter-path",
        str(adapter),
        "--config",
        str(config),
    ]
    environment = os.environ.copy()
    started = time.time()
    result = subprocess.run(command, env=environment, check=False)
    for child in adapter.rglob("*"):
        child.chmod(0o700 if child.is_dir() else 0o600)
    try:
        final_dataset_sha256, final_dataset_records = renderer_dataset_identity(data)
        dataset_unchanged = bool(
            final_dataset_sha256 == dataset_sha256 and final_dataset_records == dataset_records
        )
    except (OSError, ValueError):
        dataset_unchanged = False
    effective_exit_code = result.returncode if result.returncode else int(not dataset_unchanged)
    public_report = {
        "dataset_unchanged": dataset_unchanged,
        "elapsed_seconds": round(time.time() - started, 3),
        "exit_code": effective_exit_code,
        "model_local": True,
        "telemetry_disabled": True,
        "training_plan": training_plan,
    }
    private_report = {
        **public_report,
        "dataset_records": dataset_records,
        "dataset_sha256": dataset_sha256,
    }
    report_path = adapter / "training-run.json"
    write_private_json(report_path, private_report)
    print(json.dumps(public_report, sort_keys=True))
    return effective_exit_code


if __name__ == "__main__":
    sys.exit(main())
