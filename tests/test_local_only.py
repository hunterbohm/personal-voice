import json
import urllib.request

import pytest

from personal_voice.local_only import (
    inside_git_worktree,
    open_loopback_http_request,
    require_loopback_http_endpoint,
    require_private_input,
    write_private_json,
    write_private_json_exclusive,
)


def test_private_model_endpoints_must_be_exact_loopback_http_urls() -> None:
    assert (
        require_loopback_http_endpoint(
            "http://127.0.0.1:11434/api/generate",
            path="/api/generate",
        )
        == "http://127.0.0.1:11434/api/generate"
    )
    assert (
        require_loopback_http_endpoint(
            "http://[::1]:11434/api/generate",
            path="/api/generate",
        )
        == "http://[::1]:11434/api/generate"
    )
    for unsafe in (
        "https://127.0.0.1:11434/api/generate",
        "http://example.com/api/generate",
        "http://localhost.evil/api/generate",
        "http://127.0.0.1:11434/other",
        "http://user@localhost:11434/api/generate",
    ):
        with pytest.raises(ValueError, match="loopback"):
            require_loopback_http_endpoint(unsafe, path="/api/generate")


def test_private_loopback_transport_disables_proxies_and_redirects(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeOpener:
        def open(self, request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return object()

    def fake_build_opener(*handlers):
        captured["handlers"] = handlers
        return FakeOpener()

    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:8080")
    monkeypatch.setattr("urllib.request.build_opener", fake_build_opener)
    request = urllib.request.Request("http://127.0.0.1:11434/api/generate")

    response = open_loopback_http_request(
        request,
        path="/api/generate",
        timeout=30,
    )

    handlers = captured["handlers"]
    proxy_handler = next(
        handler for handler in handlers if isinstance(handler, urllib.request.ProxyHandler)
    )
    redirect_handler = next(
        handler for handler in handlers if isinstance(handler, urllib.request.HTTPRedirectHandler)
    )
    assert proxy_handler.proxies == {}
    assert redirect_handler.redirect_request(None, None, 307, "", {}, "http://example.com") is None
    assert captured["request"] is request
    assert captured["timeout"] == 30
    assert response is not None


def test_inside_git_worktree_handles_existing_and_future_private_paths(tmp_path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / ".git").mkdir()

    assert inside_git_worktree(repository)
    assert inside_git_worktree(repository / "future/private.sqlite")
    assert not inside_git_worktree(tmp_path / "outside/private.sqlite")


def test_private_inputs_reject_git_paths_symlinks_and_broad_permissions(tmp_path) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    source = private / "messages.json"
    source.write_text("{}")
    source.chmod(0o600)

    assert require_private_input(source) == source

    source.chmod(0o644)
    with pytest.raises(PermissionError, match="0600"):
        require_private_input(source)
    source.chmod(0o600)

    link = private / "messages-link.json"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="symlink"):
        require_private_input(link)

    repository = tmp_path / "repo"
    repository.mkdir(mode=0o700)
    (repository / ".git").mkdir()
    tracked_risk = repository / "messages.json"
    tracked_risk.write_text("{}")
    tracked_risk.chmod(0o600)
    with pytest.raises(ValueError, match="Git"):
        require_private_input(tracked_risk)


def test_private_json_writer_is_atomic_private_and_cleans_failed_temporary_files(
    tmp_path, monkeypatch
) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    destination = private / "report.json"
    write_private_json(destination, {"version": 1})

    assert json.loads(destination.read_text()) == {"version": 1}
    assert destination.stat().st_mode & 0o077 == 0

    monkeypatch.setattr(
        "personal_voice.local_only.os.replace",
        lambda _source, _destination: (_ for _ in ()).throw(OSError("synthetic failure")),
    )
    with pytest.raises(OSError, match="synthetic failure"):
        write_private_json(destination, {"version": 2})

    assert json.loads(destination.read_text()) == {"version": 1}
    assert not list(private.glob("*.tmp"))


def test_private_json_exclusive_writer_never_replaces_an_existing_version(tmp_path) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    destination = private / "candidate.json"

    write_private_json_exclusive(destination, {"version": 1})
    with pytest.raises(FileExistsError):
        write_private_json_exclusive(destination, {"version": 2})

    assert json.loads(destination.read_text()) == {"version": 1}
    assert destination.stat().st_mode & 0o077 == 0
    assert not list(private.glob("*.tmp"))
