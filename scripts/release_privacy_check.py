#!/usr/bin/env python3
"""Fail when the public release tree contains likely private material."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from personal_voice.release_privacy import scan_paths  # noqa: E402

_IGNORED_FALLBACK_PARTS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
}


def _release_paths() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    )
    tracked = [
        REPO_ROOT / item.decode()
        for item in result.stdout.split(b"\0")
        if item
    ]
    if tracked:
        return tracked
    return [
        path
        for path in REPO_ROOT.rglob("*")
        if not any(part in _IGNORED_FALLBACK_PARTS for part in path.relative_to(REPO_ROOT).parts)
    ]


def main() -> int:
    issues = scan_paths(REPO_ROOT, _release_paths())
    if issues:
        for path, category in issues:
            print(f"FAIL {category}: {path}")
        print(f"privacy check failed: {len(issues)} issue(s)")
        return 1
    print("privacy check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
