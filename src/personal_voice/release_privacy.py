"""Aggregate-only checks for accidental private material in a release tree."""

from __future__ import annotations

import re
from pathlib import Path

_BANNED_SUFFIXES = {
    ".arrow",
    ".ckpt",
    ".db",
    ".eml",
    ".gguf",
    ".heic",
    ".jsonl",
    ".mbox",
    ".mov",
    ".mp4",
    ".npy",
    ".npz",
    ".parquet",
    ".safetensors",
    ".sqlite",
    ".zip",
}
_BANNED_PARTS = {
    ".venv",
    "adapters",
    "artifacts",
    "build",
    "checkpoints",
    "evaluation",
    "exports",
    "logs",
    "manifests",
    "models",
    "dist",
    "outputs",
    "private",
    "renderer-data",
    "runs",
    "tokenizers",
    "work",
}
_MAX_FILE_BYTES = 1_000_000
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@([A-Z0-9.-]+\.[A-Z]{2,})\b", re.IGNORECASE)
_HOME_PATH = re.compile(r"/(?:Users|home)/[^/\s'\"]+")
_PHONE = re.compile(r"(?<!\w)\+?\d(?:[ .()/-]*\d){9,14}(?!\w)")
_SECRET_PATTERNS = {
    "aws-access-key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github-token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "openai-style-key": re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    "private-key": re.compile(
        "-----BEGIN " + r"(?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    ),
}


def scan_paths(root: Path, paths: list[Path]) -> list[tuple[str, str]]:
    """Return path/category pairs without returning matched private content."""

    issues: set[tuple[str, str]] = set()
    for path in paths:
        relative = path.relative_to(root)
        label = relative.as_posix()
        if path.is_symlink():
            issues.add((label, "symlink"))
            continue
        if not path.is_file():
            continue
        if any(part in _BANNED_PARTS for part in relative.parts):
            issues.add((label, "private-artifact-directory"))
        suffixes = {suffix.casefold() for suffix in path.suffixes}
        if suffixes & _BANNED_SUFFIXES:
            issues.add((label, "private-artifact-extension"))
        if path.stat().st_size > _MAX_FILE_BYTES:
            issues.add((label, "oversized-file"))
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            issues.add((label, "non-text-file"))
            continue
        if _HOME_PATH.search(text):
            issues.add((label, "absolute-home-path"))
        for match in _EMAIL.finditer(text):
            domain = match.group(1).casefold()
            if domain not in {"example.com", "users.noreply.github.com"}:
                issues.add((label, "non-example-email"))
        for match in _PHONE.finditer(text):
            value = match.group(0)
            digits = re.sub(r"\D", "", value)
            phone_shaped = bool(
                value.lstrip().startswith("+")
                or "(" in value
                or ")" in value
                or sum(character in " -" for character in value) >= 2
                or (len(digits) in {10, 11} and not value.isdigit())
            )
            if phone_shaped and "555" not in digits:
                issues.add((label, "non-example-phone"))
        for category, pattern in _SECRET_PATTERNS.items():
            if pattern.search(text):
                issues.add((label, category))
    return sorted(issues)
