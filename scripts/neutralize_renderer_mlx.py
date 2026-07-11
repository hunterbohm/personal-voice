#!/usr/bin/env python3
"""Resume private renderer neutralization with native MLX batching."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from personal_voice.renderer_data import (  # noqa: E402
    MlxNeutralizer,
    neutralize_renderer_workspace,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.home() / ".local/share/personal-voice/renderer/data/workspace.sqlite",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    os.environ.update(
        {
            "DO_NOT_TRACK": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )

    def report(progress: dict[str, int]) -> None:
        print(json.dumps({"neutralize_progress": progress}, sort_keys=True), file=sys.stderr)

    stats = neutralize_renderer_workspace(
        args.workspace,
        MlxNeutralizer(args.model),
        batch_size=args.batch_size,
        limit=args.limit,
        progress=report,
        retry_failed=args.retry_failed,
    )
    print(json.dumps(stats, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
