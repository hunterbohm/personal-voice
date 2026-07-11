import io
import json
import sqlite3
import zipfile

import pytest

from personal_voice.cli import main
from personal_voice.renderer import install_renderer_manifest


def _mark_private(path):
    path.parent.chmod(0o700)
    path.chmod(0o600)
    return path


def _install_synthetic_renderer(tmp_path, monkeypatch):
    private = tmp_path / "renderer-private"
    private.mkdir(mode=0o700)
    model = private / "model"
    adapter = private / "adapter"
    model.mkdir(mode=0o700)
    (model / "config.json").write_text(
        json.dumps({"model_type": "synthetic", "quantization": {"bits": 4}})
    )
    (model / "model.safetensors").write_bytes(b"synthetic-model")
    adapter.mkdir(mode=0o700)
    (adapter / "adapters.safetensors").write_bytes(b"synthetic-adapter")
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "fine_tune_type": "lora",
                "mask_prompt": True,
                "model": str(model.resolve()),
                "rank": 8,
            }
        )
    )
    (adapter / "training-run.json").write_text(
        json.dumps(
            {
                "dataset_records": 100,
                "dataset_sha256": "d" * 64,
                "dataset_unchanged": True,
                "exit_code": 0,
            }
        )
    )
    executable = tmp_path / "fake-mlx-generate"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "prompt = sys.stdin.read()\n"
        "draft = prompt.split('Canonical draft:\\n', 1)[1]\n"
        "print('PV_RENDERED_START')\n"
        "print(draft.replace('I can', 'yeah I can').replace('Alex', 'Taylor'))\n"
        "print('PV_RENDERED_END')\n"
    )
    executable.chmod(0o700)
    manifest = install_renderer_manifest(
        private,
        model_path=model,
        adapter_path=adapter,
        version="synthetic-adapter-v1",
        base_model="synthetic/base",
        base_revision=model.name,
    )
    monkeypatch.setenv("PERSONAL_VOICE_MLX_GENERATE", str(executable))
    return manifest


def test_bare_cli_prints_a_small_quickstart(capsys) -> None:
    assert main([]) == 0

    output = capsys.readouterr().out
    assert "personal-voice quickstart" in output
    assert "personal-voice selftest" in output
    assert "personal-voice build --help" in output
    assert "personal-voice doctor" in output


def test_default_help_keeps_advanced_renderer_commands_out_of_the_way(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--help"])

    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "try (preview)" in output
    assert "renderer-neutralize" not in output
    assert "docs/renderer.md" in output


@pytest.mark.parametrize(
    "arguments",
    (
        ["stats", "--json"],
        ["retrieve", "--query", "synthetic scheduling question", "--json"],
        ["renderer-prepare"],
    ),
)
def test_missing_index_errors_are_actionable_without_paths(tmp_path, arguments) -> None:
    database = tmp_path / "private-user-name" / "missing.sqlite"
    command = [*arguments[:1], "--db", str(database), *arguments[1:]]

    with pytest.raises(SystemExit, match="private index not found") as error:
        main(command)

    assert str(database) not in str(error.value)
    assert "personal-voice build" in str(error.value)


@pytest.mark.parametrize("command", ("stats", "retrieve"))
def test_invalid_index_errors_are_actionable_without_paths(tmp_path, command) -> None:
    database = tmp_path / "private-user-name" / "invalid.sqlite"
    database.parent.mkdir()
    database.write_text("not a SQLite index")
    arguments = [command, "--db", str(database)]
    if command == "retrieve":
        arguments.extend(("--query", "synthetic scheduling question"))

    with pytest.raises(SystemExit, match="private index is invalid") as error:
        main(arguments)

    assert str(database) not in str(error.value)
    assert "personal-voice build" in str(error.value)


def test_retrieve_checks_the_index_before_reading_private_stdin(tmp_path, monkeypatch) -> None:
    private_input = io.StringIO("synthetic private draft")
    monkeypatch.setattr("sys.stdin", private_input)

    with pytest.raises(SystemExit, match="private index not found"):
        main(["retrieve", "--db", str(tmp_path / "missing.sqlite")])

    assert private_input.tell() == 0


@pytest.mark.parametrize("command", ("stats", "retrieve"))
def test_index_open_errors_do_not_expose_paths(tmp_path, monkeypatch, command) -> None:
    database = tmp_path / "private-user-name" / "corpus.sqlite"

    def fail_to_open(_path):
        raise sqlite3.OperationalError(f"unable to open {database}")

    monkeypatch.setattr("personal_voice.store.store_stats", fail_to_open)
    arguments = [command, "--db", str(database)]
    if command == "retrieve":
        arguments.extend(("--query", "synthetic scheduling question"))

    with pytest.raises(SystemExit, match="private index cannot be read") as error:
        main(arguments)

    assert str(database) not in str(error.value)


@pytest.mark.parametrize(
    "command",
    ("renderer-neutralize", "renderer-data-stats", "renderer-audit"),
)
def test_missing_renderer_workspace_errors_without_a_traceback(tmp_path, command) -> None:
    workspace = tmp_path / "private-user-name" / "missing.sqlite"

    with pytest.raises(SystemExit, match="renderer workspace not found") as error:
        main([command, "--workspace", str(workspace)])

    assert str(workspace) not in str(error.value)
    assert "renderer-prepare" in str(error.value)


@pytest.mark.parametrize(
    "command",
    ("renderer-neutralize", "renderer-data-stats", "renderer-audit"),
)
def test_invalid_renderer_workspace_errors_without_a_traceback(tmp_path, command) -> None:
    workspace = tmp_path / "private-user-name" / "invalid.sqlite"
    workspace.parent.mkdir()
    workspace.write_text("not a SQLite workspace")

    with pytest.raises(SystemExit, match="renderer workspace is invalid") as error:
        main([command, "--workspace", str(workspace)])

    assert str(workspace) not in str(error.value)
    assert "renderer-prepare" in str(error.value)


def test_build_and_retrieve_use_aggregate_output_then_return_examples(
    tmp_path, capsys, monkeypatch
) -> None:
    source = tmp_path / "messages.jsonl"
    rows = [
        {
            "chat_id": "chat-a",
            "timestamp_ns": 1_700_000_000_000_000_000,
            "sender_role": "other",
            "text": "can you meet tomorrow?",
        },
        {
            "chat_id": "chat-a",
            "timestamp_ns": 1_700_000_001_000_000_000,
            "sender_role": "me",
            "text": "yeah afternoon should work",
        },
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _mark_private(source)
    database = tmp_path / "private" / "corpus.sqlite"

    assert main(["build", "--imessage", str(source), "--output", str(database)]) == 0
    build_output = capsys.readouterr().out
    build_stats = json.loads(build_output)
    assert build_stats["total"] == 1
    assert "yeah afternoon should work" not in build_output

    assert (
        main(
            [
                "retrieve",
                "--db",
                str(database),
                "--query",
                "meeting tomorrow",
                "--limit",
                "3",
                "--json",
            ]
        )
        == 0
    )
    retrieved = json.loads(capsys.readouterr().out)
    assert retrieved["examples"][0]["reply_text"] == "yeah afternoon should work"
    assert retrieved["examples"][0]["incoming_text"] == "can you meet tomorrow?"

    assert (
        main(
            [
                "retrieve",
                "--db",
                str(database),
                "--query",
                "meeting tomorrow",
                "--limit",
                "1",
            ]
        )
        == 0
    )
    assert "context: other: can you meet tomorrow?" in capsys.readouterr().out

    assert main(["stats", "--db", str(database)]) == 0
    human_stats = capsys.readouterr().out
    assert human_stats.startswith("total: 1\n")
    assert not human_stats.startswith("{")

    monkeypatch.setattr("sys.stdin", io.StringIO("meeting tomorrow"))
    assert (
        main(
            [
                "retrieve",
                "--db",
                str(database),
                "--limit",
                "1",
                "--json",
            ]
        )
        == 0
    )
    private_stdin_result = json.loads(capsys.readouterr().out)
    assert private_stdin_result["examples"][0]["reply_text"] == "yeah afternoon should work"


def test_doctor_reports_a_missing_private_index_without_throwing(tmp_path, capsys) -> None:
    database = tmp_path / "missing.sqlite"

    assert main(["doctor", "--db", str(database), "--json"]) == 1

    assert json.loads(capsys.readouterr().out) == {
        "database": database.name,
        "exists": False,
        "ok": False,
    }


def test_human_doctor_explains_the_next_setup_step(tmp_path, capsys) -> None:
    database = tmp_path / "missing.sqlite"

    assert main(["doctor", "--db", str(database)]) == 1

    output = capsys.readouterr().out
    assert "personal-voice: not ready" in output
    assert "next: personal-voice build --help" in output


def test_selftest_labels_its_synthetic_scope(capsys) -> None:
    assert main(["selftest"]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result == {
        "next": "personal-voice doctor",
        "ok": True,
        "scope": "synthetic plumbing only",
    }


def test_renderer_install_returns_nonzero_when_staged_doctor_fails(
    tmp_path, capsys, monkeypatch
) -> None:
    manifest = tmp_path / "renderer/versions/candidate.json"
    monkeypatch.setattr(
        "personal_voice.renderer.install_renderer_manifest",
        lambda *_args, **_kwargs: manifest,
    )
    monkeypatch.setattr(
        "personal_voice.renderer.renderer_doctor",
        lambda path: {"manifest": str(path), "ok": False},
    )

    result = main(
        [
            "renderer-install",
            "--root",
            str(tmp_path / "renderer"),
            "--model-path",
            str(tmp_path / "model"),
            "--adapter-path",
            str(tmp_path / "adapter"),
            "--version",
            "candidate",
            "--base-model",
            "synthetic/model",
            "--base-revision",
            "revision",
        ]
    )

    assert result == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_build_refuses_to_place_a_private_corpus_inside_git(tmp_path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / ".git").mkdir()
    source = tmp_path / "messages.jsonl"
    source.write_text("")

    with pytest.raises(SystemExit, match="outside a Git worktree"):
        main(
            [
                "build",
                "--imessage",
                str(source),
                "--output",
                str(repository / "corpus.db"),
            ]
        )


def test_build_pairs_a_conversation_split_across_repeated_source_files(tmp_path, capsys) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(
        json.dumps(
            {
                "chat_id": "chat-a",
                "timestamp_ns": 1_700_000_000_000_000_000,
                "sender_role": "other",
                "text": "can you look tomorrow?",
            }
        )
        + "\n"
    )
    second.write_text(
        json.dumps(
            {
                "chat_id": "chat-a",
                "timestamp_ns": 1_700_000_001_000_000_000,
                "sender_role": "me",
                "text": "yeah I can",
            }
        )
        + "\n"
    )
    _mark_private(first)
    _mark_private(second)
    database = tmp_path / "private" / "corpus.sqlite"

    assert (
        main(
            [
                "build",
                "--imessage",
                str(first),
                "--imessage",
                str(second),
                "--output",
                str(database),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                "retrieve",
                "--db",
                str(database),
                "--query",
                "look tomorrow",
                "--limit",
                "1",
                "--json",
            ]
        )
        == 0
    )

    retrieved = json.loads(capsys.readouterr().out)
    assert retrieved["examples"][0]["incoming_text"] == "can you look tomorrow?"


def test_build_filters_repeated_x_dm_outreach_before_pairing(tmp_path, capsys) -> None:
    archive_path = tmp_path / "x-archive.zip"
    broadcast = (
        "Hey! We help founders automate their entire outbound pipeline and book more "
        "calls. Would you be open to learning how it works?"
    )
    conversations = []
    for index, recipient in enumerate(("222", "333", "444", "555", "666"), start=1):
        conversations.append(
            {
                "dmConversation": {
                    "conversationId": f"111-{recipient}",
                    "messages": [
                        {
                            "messageCreate": {
                                "senderId": "111",
                                "text": broadcast,
                                "createdAt": f"2024-01-0{index}T03:00:00Z",
                            }
                        },
                        {
                            "messageCreate": {
                                "senderId": recipient,
                                "text": "is this automated?",
                                "createdAt": f"2024-01-0{index}T03:01:00Z",
                            }
                        },
                        {
                            "messageCreate": {
                                "senderId": "111",
                                "text": f"no, happy to answer your question {index}",
                                "createdAt": f"2024-01-0{index}T03:02:00Z",
                            }
                        },
                    ],
                }
            }
        )
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "data/account.js",
            'window.YTD.account.part0 = [{"account":{"accountId":"111"}}];',
        )
        archive.writestr(
            "data/direct-messages.js",
            f"window.YTD.direct_messages.part0 = {json.dumps(conversations)};",
        )
    _mark_private(archive_path)
    database = tmp_path / "private" / "corpus.sqlite"

    assert main(["build", "--x-archive", str(archive_path), "--output", str(database)]) == 0

    stats = json.loads(capsys.readouterr().out)
    assert stats["sources"] == {"x_dm": 5}
    assert stats["filtered_messages"] == {"x_dm": 5}


def test_render_reads_private_request_from_stdin_and_returns_guarded_json(
    tmp_path, capsys, monkeypatch
) -> None:
    manifest = _install_synthetic_renderer(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(json.dumps({"draft": "I can meet at 3:30.", "medium": "text"})),
    )

    assert main(["render", "--manifest", str(manifest)]) == 0

    assert json.loads(capsys.readouterr().out) == {
        "model_version": "synthetic-adapter-v1",
        "reason_code": "",
        "status": "rendered",
        "text": "yeah I can meet at 3:30.",
    }


def test_preview_is_a_simple_interactive_staged_renderer_interface(
    tmp_path, capsys, monkeypatch
) -> None:
    manifest = _install_synthetic_renderer(tmp_path, monkeypatch)
    monkeypatch.setenv("PERSONAL_VOICE_RENDERER_ROOT", str(manifest.parent.parent))
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            "email\nI can meet Alex at 3:30.\n\nSend the invite when you can.\n/done\nAlex\n"
        ),
    )

    assert main(["preview"]) == 0

    output = capsys.readouterr().out
    assert "Medium [text]:" in output
    assert "Type /done on its own line" in output
    assert "Exact names/terms to keep" in output
    assert "Original\n--------\nI can meet Alex at 3:30.\n\nSend the invite when you can." in output
    assert "Personalized draft\n------------------\nyeah I can meet Alex at 3:30." in output
    assert "Taylor" not in output
    assert "Candidate: synthetic-adapter-v1" in output
    assert str(manifest) not in output
    assert not output.lstrip().startswith("{")


def test_preview_checks_for_a_renderer_before_accepting_a_draft(
    tmp_path, capsys, monkeypatch
) -> None:
    monkeypatch.setenv("PERSONAL_VOICE_RENDERER_ROOT", str(tmp_path / "missing-renderer"))
    private_draft = "synthetic draft that should never be read"
    monkeypatch.setattr("sys.stdin", io.StringIO(private_draft))

    with pytest.raises(SystemExit, match="no active or staged voice renderer"):
        main(["preview", "--medium", "text"])

    output = capsys.readouterr().out
    assert "Paste the finished draft" not in output
    assert private_draft not in output


def test_preview_handles_end_of_input_without_a_traceback(tmp_path, monkeypatch) -> None:
    manifest = _install_synthetic_renderer(tmp_path, monkeypatch)
    monkeypatch.setenv("PERSONAL_VOICE_RENDERER_ROOT", str(manifest.parent.parent))
    monkeypatch.setattr("sys.stdin", io.StringIO(""))

    with pytest.raises(SystemExit, match="preview cancelled"):
        main(["preview"])


def test_preview_explains_that_long_replies_keep_the_frontier_draft(
    tmp_path, capsys, monkeypatch
) -> None:
    manifest = _install_synthetic_renderer(tmp_path, monkeypatch)
    long_reply = " ".join(["word"] * 257)
    monkeypatch.setattr("sys.stdin", io.StringIO(f"{long_reply}\n/done\n"))

    assert main(["preview", "--manifest", str(manifest), "--medium", "email"]) == 0

    output = capsys.readouterr().out
    assert "Status: rejected" in output
    assert "Unchanged frontier draft" in output
    assert "Personalized draft" not in output
    assert (
        "Over 256 words: the local renderer was skipped and the frontier draft was kept unchanged."
    ) in output


def test_preview_never_silently_chooses_between_staged_candidates(tmp_path, monkeypatch) -> None:
    manifest = _install_synthetic_renderer(tmp_path, monkeypatch)
    second = manifest.with_name("another-candidate.json")
    second.write_bytes(manifest.read_bytes())
    second.chmod(0o600)
    monkeypatch.setenv("PERSONAL_VOICE_RENDERER_ROOT", str(manifest.parent.parent))
    monkeypatch.setattr("sys.stdin", io.StringIO("text\nA complete draft.\n/done\n"))

    with pytest.raises(SystemExit, match="multiple staged renderers.*--version"):
        main(["preview"])
