#!/usr/bin/env python3
"""Benchmark private renderer load and generation with synthetic drafts only."""

from __future__ import annotations

import argparse
import json
import math
import resource
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from personal_voice.renderer import load_mlx_inprocess_backend, render_draft  # noqa: E402
from personal_voice.runtime_gate import (  # noqa: E402
    runtime_report_eligible,
    write_runtime_report,
)

_MAX_ENVELOPE_DRAFT = " ".join(
    [
        "This synthetic benchmark message exercises the maximum supported reply length using "
        "neutral wording without private information."
    ]
    * 16
)
_MAX_ENVELOPE_CASE = ("text", _MAX_ENVELOPE_DRAFT)
_DRAFTS = (
    ("text", "That works for me, and I can review it tomorrow."),
    ("dm", "I appreciate you sending this. I will take a look this afternoon."),
    ("email", "Thanks for the update. Tuesday at 2:30 works for me."),
    ("public", "This is a useful breakdown, especially the point about distribution."),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.iterations < 2:
        raise SystemExit("benchmark requires at least two iterations")
    started = time.perf_counter()
    backend = load_mlx_inprocess_backend(args.manifest)
    load_seconds = time.perf_counter() - started
    latencies: list[float] = []
    statuses: dict[str, int] = {}
    for index in range(args.iterations):
        medium, draft = _MAX_ENVELOPE_CASE if index == 0 else _DRAFTS[(index - 1) % len(_DRAFTS)]
        started = time.perf_counter()
        result = render_draft(draft, medium, backend)
        latencies.append(time.perf_counter() - started)
        status = str(result["status"])
        statuses[status] = statuses.get(status, 0) + 1
    warm = sorted(latencies[1:])
    p95_index = min(len(warm) - 1, max(0, round(0.95 * len(warm) + 0.5) - 1))
    rendered = statuses.get("rendered", 0)
    minimum_rendered = math.ceil(args.iterations * 0.8)
    report = {
        "first_generation_seconds": round(latencies[0], 3),
        "iterations": args.iterations,
        "load_seconds": round(load_seconds, 3),
        "ok": rendered >= minimum_rendered,
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3),
        "rendered_rate": round(rendered / args.iterations, 4),
        "statuses": dict(sorted(statuses.items())),
        "warm_mean_seconds": round(statistics.mean(warm), 3),
        "warm_p95_seconds": round(warm[p95_index], 3),
    }
    report_path = (
        args.report.expanduser()
        if args.report is not None
        else args.manifest.expanduser().parent.parent
        / "evaluation/runtime"
        / f"{backend.version}.json"
    )
    metrics = {
        key: report[key]
        for key in (
            "first_generation_seconds",
            "load_seconds",
            "peak_rss_mb",
            "warm_mean_seconds",
            "warm_p95_seconds",
        )
    }
    evidence = write_runtime_report(
        report_path,
        model_version=backend.version,
        adapter_sha256=backend.artifact_hash,
        candidate_sha256=backend.candidate_hash,
        iterations=args.iterations,
        statuses=statuses,
        metrics=metrics,
        workload={"max_reply_words": 256, "stress_cases": 1},
    )
    report["activation_eligible"] = runtime_report_eligible(
        report_path,
        model_version=backend.version,
        adapter_sha256=backend.artifact_hash,
        candidate_sha256=backend.candidate_hash,
    )
    report["report_path"] = str(report_path)
    report["ok"] = evidence["ok"]
    print(json.dumps(report, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
