"""Local-only endpoint validation for private corpus payloads."""

from __future__ import annotations

import ipaddress
import json
import os
import tempfile
import urllib.request
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlsplit


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    """Turn every redirect into an HTTP error instead of following it."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def inside_git_worktree(path: str | Path) -> bool:
    """Return whether a private artifact path resolves inside a Git worktree."""

    resolved = Path(path).expanduser().resolve()
    candidates = (
        (resolved, *resolved.parents) if resolved.is_dir() else (resolved.parent, *resolved.parents)
    )
    return any((candidate / ".git").exists() for candidate in candidates)


def require_private_input(path: str | Path) -> Path:
    """Return a private source file or reject it before parsing any content."""

    source = Path(path).expanduser()
    if source.is_symlink() or not source.is_file():
        raise ValueError("private input must be a regular file, not a symlink")
    if inside_git_worktree(source):
        raise ValueError("private input must live outside every Git worktree")
    if source.stat().st_mode & 0o077:
        raise PermissionError("private input must have mode 0600")
    if source.parent.stat().st_mode & 0o077:
        raise PermissionError("private input directory must have mode 0700")
    return source


def write_private_json(path: str | Path, payload: Mapping[str, object]) -> None:
    """Atomically write mode-0600 JSON in an existing private directory."""

    destination = Path(path).expanduser()
    if not destination.parent.is_dir() or destination.parent.stat().st_mode & 0o077:
        raise PermissionError("private JSON requires a private directory (0700)")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, destination)
        destination.chmod(0o600)
    except Exception:
        with suppress(OSError):
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def write_private_json_exclusive(path: str | Path, payload: Mapping[str, object]) -> None:
    """Atomically create mode-0600 JSON without replacing an existing path."""

    destination = Path(path).expanduser()
    if not destination.parent.is_dir() or destination.parent.stat().st_mode & 0o077:
        raise PermissionError("private JSON requires a private directory (0700)")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        descriptor = -1
        os.link(temporary, destination)
        destination.chmod(0o600)
    finally:
        if descriptor >= 0:
            with suppress(OSError):
                os.close(descriptor)
        temporary.unlink(missing_ok=True)


def require_loopback_http_endpoint(url: str, *, path: str) -> str:
    """Return a canonical URL only when it targets an explicit loopback host."""

    parsed = urlsplit(str(url))
    if parsed.scheme != "http" or parsed.path != path or parsed.query or parsed.fragment:
        raise ValueError("private model endpoint must be an exact loopback HTTP URL")
    if parsed.username or parsed.password or not parsed.hostname:
        raise ValueError("private model endpoint must be an exact loopback HTTP URL")
    hostname = parsed.hostname.casefold()
    if hostname == "localhost":
        return str(url)
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError as error:
        raise ValueError("private model endpoint must use a loopback host") from error
    if not address.is_loopback:
        raise ValueError("private model endpoint must use a loopback host")
    return str(url)


def open_loopback_http_request(
    request: urllib.request.Request,
    *,
    path: str,
    timeout: float,
):
    """Open a validated loopback request without proxies or redirects."""

    require_loopback_http_endpoint(request.full_url, path=path)
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _RejectRedirects(),
    )
    return opener.open(request, timeout=timeout)
