"""Shared privacy-preserving text normalization."""

from __future__ import annotations

import hashlib
import re
import unicodedata

_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_INTERNATIONAL_PHONE = re.compile(r"(?<!\w)\+\d(?:[ .()/-]*\d){7,14}(?!\w)")
_PHONE = re.compile(r"(?<!\w)(?:\+?1[ .-]?)?(?:\(?\d{3}\)?[ .-]?)\d{3}[ .-]?\d{4}(?!\w)")
_DIRECT_PLACEHOLDER = re.compile(r"<\|(?:email_address|phone|url)\|>")


def private_id(namespace: str, value: object) -> str:
    """Return a stable namespaced identifier that does not expose the source ID."""

    material = f"{namespace}\0{value}".encode()
    return hashlib.sha256(material).hexdigest()[:24]


def normalize_text(value: object) -> str:
    """Normalize text and redact direct identifiers without flattening its rhythm."""

    text = unicodedata.normalize("NFC", str(value or ""))
    text = _EMAIL.sub("<|email_address|>", text)
    text = _URL.sub(_redact_url, text)
    text = _INTERNATIONAL_PHONE.sub("<|phone|>", text)
    text = _PHONE.sub("<|phone|>", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def is_usable_text(value: object) -> bool:
    """Return false for empty or direct-identifier/attachment-only text."""

    text = str(value or "").strip()
    if not text:
        return False
    remainder = _DIRECT_PLACEHOLDER.sub("", text)
    return bool(remainder.strip(" \t\r\n.,;:!?()[]{}<>-—_/\\|"))


def _redact_url(match: re.Match[str]) -> str:
    value = match.group()
    suffix = ""
    while value and value[-1] in ".,;:!?)]}":
        suffix = value[-1] + suffix
        value = value[:-1]
    return f"<|url|>{suffix}"
