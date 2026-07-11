"""Command-line interface for a private reply-example index."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from personal_voice.local_only import inside_git_worktree, require_private_input
from personal_voice.pairing import pair_replies, standalone_replies

_CLI_NAME = "personal-voice"


def _default_database() -> Path:
    configured = os.environ.get("PERSONAL_VOICE_DB")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".local/share/personal-voice/corpus.sqlite"


def _default_renderer_root() -> Path:
    configured = os.environ.get("PERSONAL_VOICE_RENDERER_ROOT")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".local/share/personal-voice/renderer"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="personal-voice",
        description="Build and query a private personal-voice index.",
        epilog=(
            "Advanced renderer setup and evaluation commands are documented in "
            "docs/renderer.md."
        ),
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    build = commands.add_parser("build", help="Build the private example index")
    build.add_argument("--imessage", action="append", type=Path, default=[])
    build.add_argument("--x-archive", action="append", type=Path, default=[])
    build.add_argument("--gmail-export", action="append", type=Path, default=[])
    build.add_argument(
        "--gmail-owner",
        action="append",
        default=[],
        help="owner address; use --gmail-owner-file for real addresses to keep them out of argv",
    )
    build.add_argument(
        "--gmail-owner-file",
        action="append",
        type=Path,
        default=[],
        help="private mode-0600 file with one owner address per line",
    )
    build.add_argument("--output", type=Path, default=_default_database())
    build.set_defaults(handler=_build)

    retrieve = commands.add_parser("retrieve", help="Retrieve relevant voice examples")
    retrieve.add_argument("--db", type=Path, default=_default_database())
    retrieve.add_argument(
        "--query",
        help="synthetic query text; omit for private content and write it through stdin",
    )
    retrieve.add_argument(
        "--medium",
        choices=("auto", "text", "dm", "email", "public"),
        default="auto",
        help="destination hint used to rank examples (default: auto)",
    )
    retrieve.add_argument("--limit", type=int, default=8)
    retrieve.add_argument("--json", action="store_true")
    retrieve.set_defaults(handler=_retrieve)

    stats = commands.add_parser("stats", help="Show aggregate corpus counts")
    stats.add_argument("--db", type=Path, default=_default_database())
    stats.add_argument("--json", action="store_true")
    stats.set_defaults(handler=_stats)

    doctor = commands.add_parser("doctor", help="Check whether the private index is usable")
    doctor.add_argument("--db", type=Path, default=_default_database())
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(handler=_doctor)

    selftest = commands.add_parser("selftest", help="Exercise the index with synthetic text only")
    selftest.set_defaults(handler=_selftest)

    render = commands.add_parser(
        "render",
        help="Apply the optional local voice renderer",
        description=(
            "Read one JSON object from stdin with draft, medium, and preserve_literals; "
            "write one guarded JSON result."
        ),
    )
    render.add_argument(
        "--manifest",
        type=Path,
        default=Path.home() / ".local/share/personal-voice/renderer/current.json",
    )
    render.set_defaults(handler=_render)

    preview = commands.add_parser(
        "try",
        aliases=["preview"],
        help="Try the active or staged renderer on a finished draft",
        description=(
            "Interactively test wording only. The renderer does not decide what the reply says "
            "and this command never sends anything."
        ),
    )
    preview.add_argument("--medium", choices=("text", "dm", "email", "public"))
    preview_candidate = preview.add_mutually_exclusive_group()
    preview_candidate.add_argument("--version", help="Staged version to preview")
    preview_candidate.add_argument("--manifest", type=Path, help=argparse.SUPPRESS)
    preview.set_defaults(handler=_preview)

    renderer_prepare = commands.add_parser(
        "renderer-prepare", description="Prepare a private style-renderer workspace"
    )
    renderer_prepare.add_argument("--db", type=Path, default=_default_database())
    renderer_prepare.add_argument(
        "--workspace",
        type=Path,
        default=_default_renderer_root() / "data/workspace.sqlite",
    )
    renderer_prepare.add_argument("--text-limit", type=int, default=6_000)
    renderer_prepare.add_argument("--dm-limit", type=int, default=4_000)
    renderer_prepare.add_argument("--email-limit", type=int, default=1_000)
    renderer_prepare.add_argument("--public-limit", type=int, default=5_000)
    renderer_prepare.add_argument("--conversation-cap", type=int, default=250)
    renderer_prepare.add_argument("--identity-fraction", type=float, default=0.05)
    renderer_prepare.set_defaults(handler=_renderer_prepare)

    renderer_neutralize = commands.add_parser(
        "renderer-neutralize",
        description="Create canonical drafts with a local teacher model",
    )
    renderer_neutralize.add_argument(
        "--workspace",
        type=Path,
        default=_default_renderer_root() / "data/workspace.sqlite",
    )
    renderer_neutralize.add_argument("--model", default="qwen3.5:9b")
    renderer_neutralize.add_argument("--limit", type=int)
    renderer_neutralize.add_argument("--retry-failed", action="store_true")
    renderer_neutralize.set_defaults(handler=_renderer_neutralize)

    renderer_export = commands.add_parser(
        "renderer-export", description="Export prompt-masked MLX training data"
    )
    renderer_export.add_argument(
        "--workspace",
        type=Path,
        default=_default_renderer_root() / "data/workspace.sqlite",
    )
    renderer_export.add_argument(
        "--output",
        type=Path,
        default=_default_renderer_root() / "data/mlx",
    )
    renderer_export.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Pinned local model used for exact 512-token export checks",
    )
    renderer_export.set_defaults(handler=_renderer_export)

    renderer_stats = commands.add_parser(
        "renderer-data-stats", description="Show aggregate renderer-data counts"
    )
    renderer_stats.add_argument(
        "--workspace",
        type=Path,
        default=_default_renderer_root() / "data/workspace.sqlite",
    )
    renderer_stats.set_defaults(handler=_renderer_data_stats)

    renderer_audit = commands.add_parser(
        "renderer-audit",
        description="Audit neutral pairs with a stronger local semantic judge",
    )
    renderer_audit.add_argument(
        "--workspace",
        type=Path,
        default=_default_renderer_root() / "data/workspace.sqlite",
    )
    renderer_audit.add_argument("--model", default="qwen3.5:9b")
    renderer_audit.add_argument("--batch-size", type=int, default=8)
    renderer_audit.set_defaults(handler=_renderer_audit)

    renderer_install = commands.add_parser(
        "renderer-install", description="Stage a private versioned renderer adapter"
    )
    renderer_install.add_argument("--root", type=Path, default=_default_renderer_root())
    renderer_install.add_argument("--model-path", type=Path, required=True)
    renderer_install.add_argument("--adapter-path", type=Path, required=True)
    renderer_install.add_argument("--version", required=True)
    renderer_install.add_argument("--base-model", required=True)
    renderer_install.add_argument("--base-revision", required=True)
    renderer_install.set_defaults(handler=_renderer_install)

    renderer_activate = commands.add_parser(
        "renderer-activate",
        description="Activate an installed renderer after evaluation",
    )
    renderer_activate.add_argument("--root", type=Path, default=_default_renderer_root())
    renderer_activate.add_argument("--version", required=True)
    renderer_activate.add_argument(
        "--blind-pack",
        type=Path,
        default=_default_renderer_root() / "evaluation/blind.sqlite",
    )
    renderer_activate.add_argument(
        "--data",
        type=Path,
        default=_default_renderer_root() / "data/mlx",
        help="Exact private MLX dataset bound to the staged candidate",
    )
    renderer_activate.add_argument(
        "--privacy-report",
        type=Path,
        help="Candidate-bound aggregate privacy evidence; defaults under renderer/evaluation",
    )
    renderer_activate.add_argument(
        "--runtime-report",
        type=Path,
        help="Candidate-bound local runtime evidence; defaults under renderer/evaluation",
    )
    renderer_activate.set_defaults(handler=_renderer_activate)

    renderer_doctor_parser = commands.add_parser(
        "renderer-doctor",
        description="Check whether the optional local renderer is usable",
    )
    renderer_doctor_parser.add_argument(
        "--manifest",
        type=Path,
        default=_default_renderer_root() / "current.json",
    )
    renderer_doctor_parser.set_defaults(handler=_renderer_doctor)

    renderer_evaluate = commands.add_parser(
        "renderer-evaluate",
        description="Evaluate aggregate held-out renderer style movement",
    )
    renderer_evaluate.add_argument(
        "--manifest",
        type=Path,
        default=_default_renderer_root() / "current.json",
    )
    renderer_evaluate.add_argument(
        "--data",
        type=Path,
        default=_default_renderer_root() / "data/mlx",
        help="Exact private MLX dataset bound to the staged candidate",
    )
    renderer_evaluate.add_argument("--split", choices=("train", "valid", "test"), default="test")
    renderer_evaluate.add_argument("--limit-per-medium", type=int, default=50)
    renderer_evaluate.add_argument("--base-only", action="store_true")
    renderer_evaluate.set_defaults(handler=_renderer_evaluate)

    blind_add = commands.add_parser(
        "renderer-blind-add",
        description="Add one private retrieval-versus-renderer draft comparison",
    )
    blind_add.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Explicit staged candidate manifest under renderer/versions",
    )
    blind_add.add_argument(
        "--pack",
        type=Path,
        default=_default_renderer_root() / "evaluation/blind.sqlite",
    )
    blind_add.set_defaults(handler=_renderer_blind_add)

    blind_next = commands.add_parser(
        "renderer-blind-next", description="Show the next private blind comparison"
    )
    blind_next.add_argument(
        "--pack",
        type=Path,
        default=_default_renderer_root() / "evaluation/blind.sqlite",
    )
    blind_next.set_defaults(handler=_renderer_blind_next)

    blind_vote = commands.add_parser(
        "renderer-blind-vote", description="Rate one blind comparison"
    )
    blind_vote.add_argument(
        "--pack",
        type=Path,
        default=_default_renderer_root() / "evaluation/blind.sqlite",
    )
    blind_vote.add_argument("--case-id", required=True)
    blind_vote.add_argument("--choice", choices=("a", "b", "tie"), required=True)
    blind_vote.add_argument("--would-send", choices=("a", "b", "both", "neither"), required=True)
    blind_vote.set_defaults(handler=_renderer_blind_vote)

    blind_stats = commands.add_parser(
        "renderer-blind-stats",
        description="Show aggregate hidden-label preference counts",
    )
    blind_stats.add_argument(
        "--pack",
        type=Path,
        default=_default_renderer_root() / "evaluation/blind.sqlite",
    )
    blind_stats.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Exact candidate manifest whose votes should be scored",
    )
    blind_stats.set_defaults(handler=_renderer_blind_stats)
    return parser


def _quickstart() -> int:
    print(f"{_CLI_NAME} quickstart")
    print(f"1. Check the install: {_CLI_NAME} selftest")
    print(f"2. Build your index: {_CLI_NAME} build --help")
    print(f"3. Check your setup: {_CLI_NAME} doctor")
    print(f"Retrieve style examples: {_CLI_NAME} retrieve --help")
    print(f"Try a staged renderer from the checkout: uv run {_CLI_NAME} try")
    return 0


def _index_stats(database: Path) -> dict:
    from personal_voice.store import store_stats

    try:
        return store_stats(database.expanduser())
    except FileNotFoundError as error:
        raise SystemExit(
            f"private index not found; run '{_CLI_NAME} build --help' to create it"
        ) from error
    except ValueError as error:
        raise SystemExit(
            f"private index is invalid; rebuild it with '{_CLI_NAME} build'"
        ) from error
    except PermissionError as error:
        raise SystemExit("private index cannot be read; check its permissions") from error
    except (OSError, sqlite3.Error) as error:
        raise SystemExit("private index cannot be read; check its permissions") from error


def _require_renderer_workspace(workspace: Path) -> Path:
    from personal_voice.renderer_data import renderer_workspace_stats

    resolved = workspace.expanduser()
    if not resolved.is_file():
        raise SystemExit(
            f"renderer workspace not found; run '{_CLI_NAME} renderer-prepare' first"
        )
    try:
        renderer_workspace_stats(resolved)
    except (OSError, ValueError, sqlite3.Error) as error:
        raise SystemExit(
            f"renderer workspace is invalid; recreate it with '{_CLI_NAME} renderer-prepare'"
        ) from error
    return resolved


def _dedupe(examples: list[dict]) -> list[dict]:
    unique: dict[str, dict] = {}
    for example in examples:
        unique.setdefault(example["example_id"], example)
    return list(unique.values())


def _examples_from_messages(messages: list[dict]) -> list[dict]:
    # Put contextual replies first so their richer version wins an ID collision.
    paired = pair_replies(messages)
    return _dedupe(paired + standalone_replies(messages))


def _dedupe_messages(messages: list[dict]) -> list[dict]:
    unique: dict[tuple, dict] = {}
    for message in messages:
        key = (
            message.get("source"),
            message.get("conversation_id"),
            message.get("sender_role"),
            message.get("timestamp"),
            message.get("text"),
        )
        unique.setdefault(key, message)
    return list(unique.values())


def _build(args: argparse.Namespace) -> int:
    from personal_voice.sources.imessage import load_imessage_jsonl
    from personal_voice.sources.x_archive import load_x_archive, load_x_public_replies
    from personal_voice.store import build_store
    from personal_voice.x_dm_filter import filter_automated_x_dms

    if inside_git_worktree(args.output):
        raise SystemExit("private corpus output must be outside a Git worktree")

    source_paths = [*args.imessage, *args.x_archive, *args.gmail_export, *args.gmail_owner_file]
    try:
        private_sources = {path: require_private_input(path) for path in source_paths}
    except (PermissionError, ValueError) as error:
        raise SystemExit(str(error)) from error

    gmail_owners = list(args.gmail_owner)
    for path in args.gmail_owner_file:
        gmail_owners.extend(
            line.strip()
            for line in private_sources[path].read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    imessage_messages: list[dict] = []
    for path in args.imessage:
        imessage_messages.extend(load_imessage_jsonl(private_sources[path]))
    x_messages: list[dict] = []
    public_replies: list[dict] = []
    for path in args.x_archive:
        x_messages.extend(load_x_archive(private_sources[path]))
        public_replies.extend(load_x_public_replies(private_sources[path]))
    gmail_messages: list[dict] = []
    if args.gmail_export:
        if not gmail_owners:
            raise SystemExit("--gmail-owner or --gmail-owner-file is required with --gmail-export")
        from personal_voice.sources.gmail import load_gmail_threads

        for path in args.gmail_export:
            gmail_messages.extend(load_gmail_threads(private_sources[path], gmail_owners))
    if not (args.imessage or args.x_archive or args.gmail_export):
        raise SystemExit("build requires at least one private corpus source")

    deduped_x_messages = _dedupe_messages(x_messages)
    filtered_x_messages = filter_automated_x_dms(deduped_x_messages)

    examples = _examples_from_messages(_dedupe_messages(imessage_messages))
    examples.extend(_examples_from_messages(filtered_x_messages))
    examples.extend(standalone_replies(_dedupe_messages(public_replies)))
    examples.extend(_examples_from_messages(_dedupe_messages(gmail_messages)))
    stats = build_store(args.output.expanduser(), _dedupe(examples))
    stats["filtered_messages"] = {
        "x_dm": len(deduped_x_messages) - len(filtered_x_messages),
    }
    print(json.dumps(stats, sort_keys=True))
    return 0


def _retrieve(args: argparse.Namespace) -> int:
    from personal_voice.store import retrieve_examples

    database = args.db.expanduser()
    _index_stats(database)
    query = str(args.query if args.query is not None else sys.stdin.read()).strip()
    if not query:
        raise SystemExit("retrieve requires query text through stdin or --query")
    try:
        examples = retrieve_examples(database, query, medium=args.medium, limit=args.limit)
    except FileNotFoundError as error:
        raise SystemExit(
            f"private index not found; run '{_CLI_NAME} build --help' to create it"
        ) from error
    except ValueError as error:
        raise SystemExit(
            f"private index is invalid; rebuild it with '{_CLI_NAME} build'"
        ) from error
    except (OSError, sqlite3.Error) as error:
        raise SystemExit("private index cannot be read; check its permissions") from error
    if args.json:
        print(json.dumps({"count": len(examples), "examples": examples}, sort_keys=True))
    else:
        for index, example in enumerate(examples, start=1):
            print(f"[{index}] {example['source']}/{example['medium']}")
            if example["incoming_text"]:
                print(f"incoming: {example['incoming_text']}")
            if example["context_text"]:
                print(f"context: {example['context_text']}")
            print(f"reply: {example['reply_text']}")
            if index != len(examples):
                print()
    return 0


def _stats(args: argparse.Namespace) -> int:
    stats = _index_stats(args.db)
    if args.json:
        print(json.dumps(stats, sort_keys=True))
    else:
        print(f"total: {stats['total']}")
        print("sources: " + ", ".join(f"{key}={value}" for key, value in stats["sources"].items()))
        print("mediums: " + ", ".join(f"{key}={value}" for key, value in stats["mediums"].items()))
    return 0


def _doctor(args: argparse.Namespace) -> int:
    from personal_voice.store import store_stats

    database = args.db.expanduser()
    result: dict[str, object] = {
        "database": database.name,
        "exists": database.is_file(),
        "ok": False,
    }
    if result["exists"]:
        try:
            stats = store_stats(database)
        except (OSError, ValueError):
            pass
        else:
            result["ok"] = True
            result["total"] = stats["total"]
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        status = "ok" if result["ok"] else "not ready"
        print(f"personal-voice: {status}")
        print(f"database: {result['database']}")
        if "total" in result:
            print(f"examples: {result['total']}")
        else:
            print(f"next: {_CLI_NAME} build --help")
    return 0 if result["ok"] else 1


def _selftest(_args: argparse.Namespace | None = None) -> int:
    from personal_voice.store import build_store, retrieve_examples

    example = {
        "example_id": "synthetic",
        "source": "selftest",
        "medium": "text",
        "conversation_id": "synthetic",
        "timestamp": 1,
        "incoming_text": "can we test this?",
        "context_text": "other: can we test this?",
        "reply_text": "yeah absolutely",
    }
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "selftest.sqlite"
        build_store(database, [example])
        ok = bool(retrieve_examples(database, "test", limit=1))
    print(
        json.dumps(
            {
                "next": f"{_CLI_NAME} doctor",
                "ok": ok,
                "scope": "synthetic plumbing only",
            },
            sort_keys=True,
        )
    )
    return 0 if ok else 1


def _render(args: argparse.Namespace) -> int:
    from personal_voice.renderer import load_mlx_backend, render_draft

    try:
        request = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError) as error:
        raise SystemExit("render requires one JSON request on stdin") from error
    if not isinstance(request, dict):
        raise SystemExit("render request must be a JSON object")
    draft = str(request.get("draft") or "")
    medium = str(request.get("medium") or "").strip()
    raw_literals = request.get("preserve_literals") or []
    if not isinstance(raw_literals, list):
        raise SystemExit("preserve_literals must be a JSON list")
    try:
        backend = load_mlx_backend(args.manifest)
    except (FileNotFoundError, PermissionError, ValueError, json.JSONDecodeError):
        result = {
            "model_version": "unavailable",
            "reason_code": "backend_unavailable",
            "status": "unavailable",
            "text": draft,
        }
    else:
        result = render_draft(
            draft,
            medium,
            backend,
            preserve_literals=[str(value) for value in raw_literals],
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


def _preview(args: argparse.Namespace) -> int:
    from personal_voice.renderer import load_mlx_backend, render_draft

    manifest = (
        args.manifest.expanduser() if args.manifest else _preview_renderer_manifest(args.version)
    )
    try:
        backend = load_mlx_backend(manifest)
    except (FileNotFoundError, PermissionError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit("no usable active or staged voice renderer was found") from error

    medium = args.medium
    if medium is None:
        try:
            medium = input("Medium [text]: ").strip().casefold() or "text"
        except EOFError as error:
            raise SystemExit("preview cancelled: no input received") from error
    if medium not in {"text", "dm", "email", "public"}:
        raise SystemExit("medium must be text, dm, email, or public")
    print("Paste the finished draft. Type /done on its own line when finished.")
    lines: list[str] = []
    for line in sys.stdin:
        value = line.rstrip("\r\n")
        if value == "/done":
            break
        lines.append(value)
    draft = "\n".join(lines).strip()
    if not draft:
        raise SystemExit("preview requires a draft")
    try:
        exact = input("Exact names/terms to keep (comma-separated, Enter to skip): ").strip()
    except EOFError:
        exact = ""
    preserve_literals = [value.strip() for value in exact.split(",") if value.strip()]
    result = render_draft(draft, medium, backend, preserve_literals=preserve_literals)
    print(f"\nCandidate: {backend.version}")
    print(f"Medium: {medium}")
    print("\nOriginal\n--------")
    print(draft)
    output_label = (
        "Personalized draft" if result["status"] == "rendered" else "Unchanged frontier draft"
    )
    print(f"\n{output_label}\n{'-' * len(output_label)}")
    print(result["text"])
    print(f"\nStatus: {result['status']}")
    if result["reason_code"] == "long_form_unsupported":
        print(
            "Over 256 words: the local renderer was skipped and the frontier draft was kept "
            "unchanged."
        )
    elif result["status"] != "rendered":
        print(f"Not rewritten: {result['reason_code']}. The original draft was kept unchanged.")
    print("Preview only: this does not run the frontier meaning check or send anything.")
    return 0


def _preview_renderer_manifest(version: str | None) -> Path:
    root = _default_renderer_root()
    current = root / "current.json"
    if version is None and current.is_file() and not current.is_symlink():
        return current
    versions = root / "versions"
    candidates = [
        path
        for path in versions.glob("*.json")
        if path.is_file() and not path.is_symlink() and not path.stat().st_mode & 0o077
    ]
    if version is not None:
        selected = versions / f"{version}.json"
        if selected not in candidates:
            raise SystemExit(f"staged renderer {version!r} was not found")
        return selected
    if not candidates:
        raise SystemExit("no active or staged voice renderer was found")
    if len(candidates) > 1:
        names = ", ".join(sorted(path.stem for path in candidates))
        raise SystemExit(f"multiple staged renderers found; rerun with --version NAME ({names})")
    return candidates[0]


def _renderer_prepare(args: argparse.Namespace) -> int:
    from personal_voice.renderer_data import prepare_renderer_workspace

    _index_stats(args.db)
    stats = prepare_renderer_workspace(
        args.db,
        args.workspace,
        per_medium={
            "dm": args.dm_limit,
            "email": args.email_limit,
            "public": args.public_limit,
            "text": args.text_limit,
        },
        conversation_cap=args.conversation_cap,
        identity_fraction=args.identity_fraction,
    )
    print(json.dumps(stats, sort_keys=True))
    return 0


def _renderer_neutralize(args: argparse.Namespace) -> int:
    from personal_voice.renderer_data import OllamaNeutralizer, neutralize_renderer_workspace

    workspace = _require_renderer_workspace(args.workspace)

    def report(progress: dict[str, int]) -> None:
        print(json.dumps({"neutralize_progress": progress}, sort_keys=True), file=sys.stderr)

    stats = neutralize_renderer_workspace(
        workspace,
        OllamaNeutralizer(args.model),
        limit=args.limit,
        progress=report,
        retry_failed=args.retry_failed,
    )
    print(json.dumps(stats, sort_keys=True))
    return 0


def _renderer_export(args: argparse.Namespace) -> int:
    from personal_voice.renderer_data import export_renderer_dataset
    from personal_voice.training_data import load_renderer_tokenizer

    workspace = _require_renderer_workspace(args.workspace)
    stats = export_renderer_dataset(
        workspace,
        args.output,
        tokenizer=load_renderer_tokenizer(args.model),
        require_semantic_audit=True,
    )
    print(json.dumps(stats, sort_keys=True))
    return 0


def _renderer_data_stats(args: argparse.Namespace) -> int:
    from personal_voice.renderer_data import renderer_workspace_stats

    workspace = _require_renderer_workspace(args.workspace)
    print(json.dumps(renderer_workspace_stats(workspace), sort_keys=True))
    return 0


def _renderer_audit(args: argparse.Namespace) -> int:
    from personal_voice.renderer_audit import (
        OllamaSemanticJudge,
        audit_renderer_workspace,
        prepare_semantic_audit,
        semantic_audit_stats,
    )
    from personal_voice.renderer_data import finalize_renderer_workspace

    def report(progress: dict[str, int]) -> None:
        print(json.dumps({"semantic_audit_progress": progress}, sort_keys=True), file=sys.stderr)

    workspace = _require_renderer_workspace(args.workspace)
    finalize_renderer_workspace(workspace)
    existing = semantic_audit_stats(workspace)
    if (
        not existing["total"]
        or existing.get("unchecked_ready")
        or existing.get("legacy_shape_pass")
    ):
        prepare_semantic_audit(workspace)
    result = audit_renderer_workspace(
        workspace,
        OllamaSemanticJudge(args.model),
        batch_size=args.batch_size,
        progress=report,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["complete"] else 1


def _renderer_install(args: argparse.Namespace) -> int:
    from personal_voice.renderer import install_renderer_manifest, renderer_doctor

    manifest = install_renderer_manifest(
        args.root,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        version=args.version,
        base_model=args.base_model,
        base_revision=args.base_revision,
    )
    result = renderer_doctor(manifest)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


def _renderer_activate(args: argparse.Namespace) -> int:
    from personal_voice.renderer import activate_renderer_manifest, renderer_doctor

    manifest = activate_renderer_manifest(
        args.root,
        args.version,
        blind_pack=args.blind_pack,
        privacy_report=args.privacy_report,
        runtime_report=args.runtime_report,
        data_path=args.data,
    )
    result = renderer_doctor(manifest)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


def _renderer_doctor(args: argparse.Namespace) -> int:
    from personal_voice.renderer import renderer_doctor

    result = renderer_doctor(args.manifest)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


def _renderer_evaluate(args: argparse.Namespace) -> int:
    from personal_voice.privacy_probe import require_candidate_dataset_binding
    from personal_voice.renderer import load_mlx_inprocess_backend
    from personal_voice.renderer_eval import evaluate_renderer_dataset

    try:
        require_candidate_dataset_binding(args.manifest, args.data)
    except ValueError as error:
        raise SystemExit("renderer evaluation dataset is not bound to this candidate") from error
    backend = load_mlx_inprocess_backend(
        args.manifest,
        use_adapter=not args.base_only,
    )
    result = evaluate_renderer_dataset(
        args.data,
        backend,
        split=args.split,
        limit_per_medium=args.limit_per_medium,
    )
    try:
        require_candidate_dataset_binding(args.manifest, args.data)
    except ValueError as error:
        raise SystemExit("renderer evaluation dataset changed during evaluation") from error
    print(json.dumps(result, sort_keys=True))
    return 0


def _renderer_blind_add(args: argparse.Namespace) -> int:
    from personal_voice.blind_eval import add_blind_case
    from personal_voice.renderer import load_mlx_backend

    try:
        request = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError) as error:
        raise SystemExit("renderer-blind-add requires one JSON request on stdin") from error
    if not isinstance(request, dict):
        raise SystemExit("blind evaluation request must be a JSON object")
    literals = request.get("preserve_literals") or []
    if not isinstance(literals, list):
        raise SystemExit("preserve_literals must be a JSON list")
    baseline_source = str(request.get("baseline_source") or "")
    if baseline_source != "retrieval_augmented_draft":
        raise SystemExit("renderer-blind-add requires baseline_source=retrieval_augmented_draft")
    result = add_blind_case(
        args.pack,
        str(request.get("draft") or ""),
        str(request.get("medium") or ""),
        load_mlx_backend(args.manifest),
        baseline_source=baseline_source,
        preserve_literals=[str(value) for value in literals],
        evaluation_context=str(request.get("evaluation_context") or ""),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def _renderer_blind_next(args: argparse.Namespace) -> int:
    from personal_voice.blind_eval import next_blind_case

    result = next_blind_case(args.pack)
    print(json.dumps({"case": result}, ensure_ascii=False, sort_keys=True))
    return 0


def _renderer_blind_vote(args: argparse.Namespace) -> int:
    from personal_voice.blind_eval import record_blind_vote

    record_blind_vote(
        args.pack,
        args.case_id,
        args.choice,
        would_send=args.would_send,
    )
    print(json.dumps({"ok": True}))
    return 0


def _renderer_blind_stats(args: argparse.Namespace) -> int:
    from personal_voice.blind_eval import blind_eval_stats
    from personal_voice.renderer import RENDERER_CONTRACT_VERSION, load_mlx_backend

    backend = load_mlx_backend(args.manifest)
    print(
        json.dumps(
            blind_eval_stats(
                args.pack,
                model_version=backend.version,
                candidate_sha256=backend.candidate_hash,
                renderer_contract_version=RENDERER_CONTRACT_VERSION,
            ),
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return an exit status."""

    args = _parser().parse_args(argv)
    if not hasattr(args, "handler"):
        return _quickstart()
    return args.handler(args)
