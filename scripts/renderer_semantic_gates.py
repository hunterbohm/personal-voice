#!/usr/bin/env python3
"""Run synthetic commitment, negation, question, and literal preservation gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from personal_voice.renderer import load_mlx_inprocess_backend  # noqa: E402
from personal_voice.renderer_semantic import evaluate_semantic_gates  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_semantic_gates(load_mlx_inprocess_backend(args.manifest))
    print(json.dumps(report, sort_keys=True))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
