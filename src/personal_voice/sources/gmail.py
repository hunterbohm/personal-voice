"""Load private Gmail thread exports into the shared message contract."""

from __future__ import annotations

import base64
import binascii
import json
import re
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from email.utils import getaddresses, parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from personal_voice.normalize import is_usable_text, normalize_text, private_id

_QUOTED_REPLY = re.compile(r"^\s*On .+ wrote:\s*$", re.IGNORECASE)
_QUOTED_LINE = re.compile(r"^\s*>")
_SIGNATURE_DELIMITER = re.compile(r"^\s*--\s*$")
_FORWARDED_MESSAGE = re.compile(
    r"^\s*(?:-+\s*Forwarded message\s*-+|Begin forwarded message:)\s*$",
    re.IGNORECASE,
)
_MOBILE_SIGNATURE = re.compile(
    r"^\s*(?:Sent from my (?:Android|iPad|iPhone|mobile device)|Get Outlook for (?:Android|iOS))\b",
    re.IGNORECASE,
)
_HTML_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_HTML_BREAK_TAGS = {
    "br",
    "div",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "ol",
    "p",
    "pre",
    "table",
    "td",
    "th",
    "tr",
    "ul",
}
_HTML_SKIP_TAGS = {"blockquote", "head", "script", "style"}
_HTML_VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta"}
_NON_MESSAGE_TEXT = {
    "[attachment]",
    "[attachment omitted]",
    "[image]",
    "[image omitted]",
    "<attachment>",
    "attachment",
    "attachment omitted",
    "image omitted",
    "sent an attachment",
}
_ATTACHMENT_ONLY = re.compile(
    r"^\[?(?:attachments?|images?)(?:\s+(?:omitted|only))?(?::[^\]\n]+)?\]?[.!]?$",
    re.IGNORECASE,
)


def load_gmail_threads(path: str | Path, owner_addresses: Iterable[str]) -> list[dict[str, Any]]:
    """Return normalized messages from a decoded Gmail thread export."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    owners = {address.strip().casefold() for address in owner_addresses if address.strip()}
    messages: list[dict[str, Any]] = []

    for thread in _threads(payload):
        thread_id = thread.get("id")
        thread_messages = thread.get("messages")
        if not isinstance(thread_messages, list):
            continue
        for message in thread_messages:
            if not isinstance(message, dict):
                continue
            headers = _message_headers(message)
            from_header = _header(headers, "from")
            if not from_header:
                continue
            text = normalize_text(_clean_body(_message_body(message)))
            if not text or _is_non_message_text(text):
                continue
            source_thread_id = thread_id or _case_insensitive_value(message, "threadId")
            if not source_thread_id:
                source_thread_id = _case_insensitive_value(message, "thread_id")
            if not source_thread_id:
                continue
            normalized = {
                "conversation_id": private_id("gmail", source_thread_id),
                "sender_role": _sender_role(from_header, owners),
                "timestamp": _timestamp(message, headers),
                "text": text,
                "source": "gmail",
                "medium": "email",
            }
            subject = normalize_text(_header(headers, "subject"))
            if is_usable_text(subject):
                normalized["context_hint"] = f"subject: {subject}"
            messages.append(normalized)

    return messages


def _threads(payload: dict[str, Any]) -> list[dict[str, Any]]:
    threads = payload.get("threads")
    if isinstance(threads, list):
        return [thread for thread in threads if isinstance(thread, dict)]
    thread = payload.get("thread")
    if isinstance(thread, dict):
        return [thread]

    page_messages = payload.get("messages")
    if isinstance(page_messages, list):
        return _message_page_threads(page_messages)

    data = payload.get("data")
    page_messages = data.get("messages") if isinstance(data, dict) else None
    if payload.get("successful") is True and isinstance(page_messages, list):
        return _message_page_threads(page_messages)
    return []


def _message_page_threads(messages: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": _case_insensitive_value(message, "threadId")
            or _case_insensitive_value(message, "thread_id"),
            "messages": [message],
        }
        for message in messages
        if isinstance(message, dict)
    ]


def _header(headers: object, name: str) -> str:
    if not isinstance(headers, dict):
        return ""
    wanted = name.casefold()
    for key, value in headers.items():
        if str(key).casefold() == wanted:
            return str(value or "")
    return ""


def _message_headers(message: dict[str, Any]) -> dict[str, str]:
    headers: dict[str, str] = {}
    payload = message.get("payload")
    payload_headers = payload.get("headers") if isinstance(payload, dict) else None
    sources = [payload_headers, message.get("headers")]
    for source in sources:
        if isinstance(source, dict):
            headers.update({str(key).casefold(): str(value or "") for key, value in source.items()})
        elif isinstance(source, list):
            for item in source:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "").casefold()
                if name:
                    headers[name] = str(item.get("value") or "")

    for name in ("from", "to", "cc", "bcc", "date", "subject"):
        if name not in headers:
            direct = _case_insensitive_value(message, name)
            if direct is None and name == "from":
                direct = _case_insensitive_value(message, "sender")
            if direct is not None:
                headers[name] = str(direct)
    return headers


def _case_insensitive_value(mapping: dict[str, Any], name: str) -> object | None:
    wanted = name.casefold()
    for key, value in mapping.items():
        if str(key).casefold() == wanted:
            return value
    return None


def _message_body(message: dict[str, Any]) -> str:
    for name in ("body", "text", "messageText", "decodedBody", "decoded_body"):
        value = _case_insensitive_value(message, name)
        if isinstance(value, str) and value:
            return value

    payload = message.get("payload")
    return _payload_body(payload) if isinstance(payload, dict) else ""


def _payload_body(payload: dict[str, Any]) -> str:
    parts = list(_walk_parts(payload))
    for wanted_type in ("text/plain", "text/html"):
        for part in parts:
            mime_type = str(part.get("mimeType") or "").partition(";")[0].casefold()
            if mime_type != wanted_type or part.get("filename"):
                continue
            body = part.get("body")
            data = body.get("data") if isinstance(body, dict) else None
            if isinstance(data, str) and data:
                decoded = _decode_gmail_data(data)
                if decoded:
                    return decoded
    return ""


def _walk_parts(part: dict[str, Any]) -> Iterator[dict[str, Any]]:
    yield part
    children = part.get("parts")
    if isinstance(children, list):
        for child in children:
            if isinstance(child, dict):
                yield from _walk_parts(child)


def _decode_gmail_data(data: str) -> str:
    try:
        raw = base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))
    except (binascii.Error, ValueError):
        return ""
    return raw.decode("utf-8", errors="replace")


def _sender_role(from_header: str, owners: set[str]) -> str:
    senders = {address.casefold() for _, address in getaddresses([from_header]) if address}
    return "me" if senders & owners else "other"


def _timestamp(message: dict[str, Any], headers: object) -> int:
    for name in ("internalDate", "messageTimestamp", "timestamp"):
        raw_timestamp = _case_insensitive_value(message, name)
        if raw_timestamp is None or not str(raw_timestamp).strip():
            continue
        parsed_timestamp = _parse_timestamp(raw_timestamp)
        if parsed_timestamp is not None:
            return parsed_timestamp

    raw_date = _header(headers, "date")
    if not raw_date:
        return 0
    try:
        parsed = parsedate_to_datetime(raw_date)
    except (TypeError, ValueError):
        return 0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp())


def _parse_timestamp(value: object) -> int | None:
    text = str(value).strip()
    try:
        numeric = int(text)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return int(parsed.timestamp())
    return numeric // 1000 if numeric >= 10_000_000_000 else numeric


def _clean_body(value: object) -> str:
    body = str(value or "")
    if _HTML_TAG.search(body):
        parser = _EmailHTMLParser()
        parser.feed(body)
        parser.close()
        body = parser.text()

    kept: list[str] = []
    for line in body.splitlines():
        if (
            _QUOTED_REPLY.match(line)
            or _QUOTED_LINE.match(line)
            or _SIGNATURE_DELIMITER.match(line)
            or _FORWARDED_MESSAGE.match(line)
            or _MOBILE_SIGNATURE.match(line)
        ):
            break
        kept.append(line)
    return "\n".join(kept).strip()


def _is_non_message_text(text: str) -> bool:
    compact = re.sub(r"\s+", " ", text).strip().casefold()
    return compact in _NON_MESSAGE_TEXT or bool(_ATTACHMENT_ONLY.fullmatch(compact))


class _EmailHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._pieces: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        classes = {
            class_name.casefold()
            for name, value in attrs
            if name.casefold() == "class" and value
            for class_name in value.split()
        }
        should_skip = tag in _HTML_SKIP_TAGS or bool(classes & {"gmail_attr", "gmail_quote"})
        if self._skip_depth or should_skip:
            if tag not in _HTML_VOID_TAGS:
                self._skip_depth += 1
            return
        if tag in _HTML_BREAK_TAGS:
            self._break()

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._skip_depth == 0 and tag.casefold() in _HTML_BREAK_TAGS:
            self._break()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if self._skip_depth:
            self._skip_depth -= 1
            return
        if tag in _HTML_BREAK_TAGS:
            self._break()

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._pieces.append(data)

    def text(self) -> str:
        return "".join(self._pieces).strip()

    def _break(self) -> None:
        if self._pieces and not self._pieces[-1].endswith("\n"):
            self._pieces.append("\n")
