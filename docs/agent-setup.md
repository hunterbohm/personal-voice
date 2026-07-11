# Agent setup runbook

You are setting up `personal-voice` for the local user. Finish with a working private index and, when the user's data-boundary choice allows it, an installed drafting skill. The tool retrieves examples of how the user writes; it does not decide what a reply should say and it never sends messages.

Do the work when it is safe and local. Ask the user only for account access, exports, privacy choices, or approval before optional model downloads and training.

## Read before acting

Read these files completely:

1. [`../README.md`](../README.md)
2. [`privacy.md`](privacy.md)
3. [`data-contract.md`](data-contract.md)
4. [`../examples/personal-voice-skill/SKILL.md`](../examples/personal-voice-skill/SKILL.md)
5. [`renderer.md`](renderer.md) only if the user wants the optional local renderer

Inspect the repository's current CLI help before using commands. Do not rely on remembered flags from another version.

## Hard rules

- Never commit, upload, sync, or publish message exports, indexes, datasets, adapters, model weights, manifests, or evaluation files.
- Keep every `personal-voice` source copy and generated artifact outside Git worktrees and cloud-synced folders. The pinned iMessage extractor's ignored `work/` directory is the one temporary exception described below.
- Use directories with mode `0700` and private files with mode `0600`.
- Never put a real message, draft, email address, or retrieval query in agent-authored command arguments, shell history, temporary files, logs, issues, or chat output. `gog auth add` requires an address argument; the user must run that one OAuth step in a private, non-recorded terminal.
- Use process stdin for private retrieval queries and renderer requests. Use `--gmail-owner-file` for real account addresses.
- Do not show retrieved examples in the setup transcript. Report source categories and aggregate counts only.
- Do not call a sending, posting, or publishing tool. Installing this workflow never grants permission to send.
- Treat a LoRA adapter trained on private messages as private data.

If a requested step would cross one of these boundaries, stop and explain the exact boundary.

## 1. Inspect the machine

Confirm the operating system, architecture, Python version, available private disk space, installed agent runtimes, and whether `uv` is available. Locate an appropriate code directory for the repository, then clone or update it.

Do not search broadly for personal archives. Ask the user which sources they want to use and where those exports are located. Supported sources are:

| Source | Input | User action that may be required |
|---|---|---|
| iMessage | Pseudonymized JSONL from `texts-to-transformer` | Grant local Messages access and run the audited upstream extractor |
| X | Official account archive ZIP | Request and download the archive from X |
| Gmail | Decoded message or thread JSON | Complete read-only OAuth authorization |

One source is enough to start. The index can be rebuilt later with more sources.

## 2. Install and test the CLI

From the repository checkout, record its canonical path. Keep
`PERSONAL_VOICE_REPO` set for the rest of setup; set it again from the known
checkout path whenever the agent opens a new shell. Do not rely on the shell's
current directory after this point.

```bash
PERSONAL_VOICE_REPO="$(git rev-parse --show-toplevel)"
test -f "$PERSONAL_VOICE_REPO/pyproject.toml"
uv tool install "$PERSONAL_VOICE_REPO"
personal-voice selftest
personal-voice
```

If an older install is already present, rebuild it from the current checkout:

```bash
uv tool install --force --reinstall --no-cache "$PERSONAL_VOICE_REPO"
```

`selftest` uses synthetic data only. It proves that the local index plumbing works; it does not prove that the user's corpus is ready or that generated writing matches the user.

## 3. Prepare private source storage

Create both levels of the local storage root explicitly. This prevents an intermediate directory from being created with broader default permissions:

```bash
install -d -m 700 ~/.local/share/personal-voice
install -d -m 700 ~/.local/share/personal-voice/sources
```

Copy each selected export into that directory with mode `0600`. Do not parse a source until the CLI's privacy checks accept it.

For iMessage, use only the audited upstream commit. Keep its checkout under the
private storage root, never inside `PERSONAL_VOICE_REPO`. Clone or refresh it,
then detach at the exact revision without changing the shell's working
directory:

```bash
EXTRACTOR_REPO="$HOME/.local/share/personal-voice/tools/texts-to-transformer"
install -d -m 700 "$HOME/.local/share/personal-voice/tools"

if test -d "$EXTRACTOR_REPO/.git"; then
  test "$(git -C "$EXTRACTOR_REPO" remote get-url origin)" = \
    https://github.com/Doriandarko/texts-to-transformer.git
  git -C "$EXTRACTOR_REPO" fetch --prune origin
else
  git clone https://github.com/Doriandarko/texts-to-transformer.git \
    "$EXTRACTOR_REPO"
fi

chmod 700 "$EXTRACTOR_REPO"
git -C "$EXTRACTOR_REPO" checkout --detach \
  90419977f4b0ab80c7c160faa2adcbcc2bfe7943
test "$(git -C "$EXTRACTOR_REPO" rev-parse HEAD)" = \
  90419977f4b0ab80c7c160faa2adcbcc2bfe7943

(
  cd "$EXTRACTOR_REPO"
  uv sync
  uv run imessage-mlx doctor
)
```

`doctor` must pass before accessing Messages. If Full Disk Access is missing,
ask the user to grant it to the terminal or agent application, fully restart
that application, set `EXTRACTOR_REPO` again, and rerun the `doctor` subshell.
Do not change `chat.db` permissions or copy the live database manually.

After the user approves the iMessage source, run only the extraction stages needed by `personal-voice`:

```bash
EXTRACTOR_REPO="$HOME/.local/share/personal-voice/tools/texts-to-transformer"

(
  cd "$EXTRACTOR_REPO"
  uv run imessage-mlx snapshot --config configs/data.yaml
  uv run imessage-mlx inspect-schema \
    --database work/snapshot/chat.db \
    --output work/schema/schema.json
  uv run imessage-mlx prepare --config configs/data.yaml
  uv run imessage-mlx privacy-audit
  test -z "$(git ls-files work outputs)"
  install -m 600 work/extracted/messages.jsonl \
    ~/.local/share/personal-voice/sources/imessage.jsonl
)
```

The pinned extractor intentionally writes ignored private files under its own `work/` directory. That is the temporary exception to the no-Git-worktree storage rule. Never stage or commit those files. `personal-voice` reads only the copied JSONL outside the extractor checkout.

For X, use the official archive ZIP as downloaded. The loader reads DMs, group DMs, and authored public replies. It filters repeated long outreach templates before pairing replies.

For Gmail, create a private owner file with one account address per line. The user must run the `gog auth add` step from the main README because `gog` requires the address as a process argument. After authorization, use the owner file to set `GOG_ACCOUNT` without embedding the address in the agent command. Use read-only OAuth, `--gmail-no-send`, and `in:sent` by default. Use `in:anywhere` only if the user explicitly wants incoming-thread context and accepts the broader export. Do not pass a real address through the build command's `--gmail-owner` flag.

Before continuing, verify that every source is a regular file, each source file is mode `0600`, its parent is mode `0700`, and no source resolves inside a Git worktree.

## 4. Build the private index

Run `personal-voice build` with only the source flags that are available. A complete example looks like this:

```bash
personal-voice build \
  --imessage ~/.local/share/personal-voice/sources/imessage.jsonl \
  --x-archive ~/.local/share/personal-voice/sources/x-archive.zip \
  --gmail-export ~/.local/share/personal-voice/sources/gmail-page.json \
  --gmail-owner-file ~/.local/share/personal-voice/sources/gmail-owners.txt \
  --output ~/.local/share/personal-voice/corpus.sqlite
```

Omit unavailable sources. Never replace missing data with scraped, inferred, or third-party messages.

Verify without printing corpus text:

```bash
personal-voice doctor --json
personal-voice stats --json
```

`doctor` must report `ok: true`, and `stats` must report a nonzero total. Keep the counts aggregate. Do not run `retrieve` merely to prove setup in a recorded terminal because retrieval intentionally returns private examples.

Check repository cleanliness without printing accidental filenames into the transcript:

```bash
test -n "$PERSONAL_VOICE_REPO"
git -C "$PERSONAL_VOICE_REPO" diff --quiet
git -C "$PERSONAL_VOICE_REPO" diff --cached --quiet
test -z "$(git -C "$PERSONAL_VOICE_REPO" \
  ls-files --others --exclude-standard)"
```

Setup should not create private files in the repository. If a clean check fails, inspect and fix it in a private, non-recorded terminal before continuing.

## 5. Install the drafting skill

Before installing the skill into a hosted runtime, explain that the incoming message and retrieved examples may be sent to that model provider. Ask the user to accept that boundary.

If the user declines, install the skill only into an already configured local drafting runtime that keeps both retrieval and drafting on the device. This repository does not install a capable local reasoning model. If no suitable local runtime exists, leave the skill disabled and report the blocker; the private index can still be complete.

After the data-boundary choice, use the repository's skill file instead of rewriting it:

```text
examples/personal-voice-skill/SKILL.md
```

Install that file as `personal-voice/SKILL.md` in each agent runtime the user actually uses. Common roots include `~/.codex/skills`, `~/.claude/skills`, and `~/.hermes/skills`. Detect the runtime's real convention before writing. Do not create guessed runtime directories, and do not overwrite a customized copy without reviewing the difference.

Restart or open a fresh agent session after installation. Confirm that the runtime discovers the skill and can find `personal-voice` on its PATH.

If the user accepts the boundary, test with a synthetic incoming message in a private, non-recorded agent session. The incoming message is synthetic, but the retrieved style examples are still real private data and may reach the chosen model provider:

```text
Draft a short reply in my voice, but do not send it: "Can we move our call to Thursday afternoon?"
```

The agent should return only a draft. It must not quote retrieved examples or call a sending tool.

## 6. Optional local renderer

The retriever is the default voice layer. The renderer is an experimental wording pass for short, completed replies.

Ask before downloading a base model or starting training. The automated setup path requires macOS on `arm64`, at least 24 GiB of unified memory, and at least 20 GiB of free local disk before downloads. A user can explicitly approve a lower-resource experiment; otherwise leave retrieval working and do not invent an unsupported CUDA or cloud-training path.

If the user wants the renderer and the machine is supported, read [`renderer.md`](renderer.md) completely and follow it in order. Keep all data and model artifacts under private local storage, outside Git. Use the renderer-enabled checkout for its commands:

```bash
test -n "$PERSONAL_VOICE_REPO"
cd "$PERSONAL_VOICE_REPO"
uv sync --extra renderer
```

Training completion is not activation. Stage the candidate, run the documented integrity, held-out, semantic, runtime, and privacy checks, then use `uv run personal-voice try` for manual drafts. If more than one candidate is staged, pass `--version <version>`.

Do not manufacture blind comparisons or call `renderer-activate` during setup, even if every automated check passes. Blind voting and activation are separate later tasks that require real frontier-plus-retrieval drafts, direct user ratings, and explicit approval. Verify that `current.json` was not created.

Never split a long reply into chunks to bypass the renderer's evaluated length limit. Keep the complete frontier-model draft when rendering is unavailable or rejected.

## 7. Completion report

Report only:

- where the code was installed, using a `~`-relative path rather than the local account name
- which source categories were included
- aggregate example counts by source and medium
- which agent runtimes received the skill
- whether the user accepted hosted-model retrieval or chose local-only drafting
- whether the optional renderer was skipped or staged, and that it remains inactive
- any remaining user action

Do not include message excerpts, contact identifiers, archive paths that reveal identities, retrieved examples, model outputs based on real messages, or secrets.

The setup is complete when the CLI self-test passes, `doctor` reports a usable nonempty index, the repository contains no private artifacts, and the user understands the data boundary. Approved agent runtimes must load the skill; if no runtime is approved or capable, report that blocker instead of weakening the privacy rule. If a renderer was staged, `current.json` must still be absent. Do not claim that the voice quality is good until the user has tested real drafts.
