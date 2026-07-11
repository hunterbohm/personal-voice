# personal-voice

`personal-voice` helps an AI draft in your writing style without turning a small style model into the brain of your agent.

It builds a private SQLite index from messages you wrote, retrieves relevant examples for the current conversation, and leaves intent and factual reasoning to the calling model. An optional local MLX renderer can make one guarded wording pass over the completed draft.

This repository contains code and synthetic tests only. It contains no messages, email exports, contact identifiers, datasets, adapters, model weights, or private evaluation results.

## Let an agent set it up

Paste this into a coding agent running on the computer that holds your messages:

```text
Set up https://github.com/hunterbohm/personal-voice on this machine. Clone or update it, then read docs/agent-setup.md completely and follow it end to end. Do the work instead of just explaining it. Use only data sources I explicitly approve, obey every privacy gate, install and verify the drafting skill only for agent runtimes already present, and never send or publish anything. Stage and test the optional local renderer if this machine supports it, but do not activate it. Ask me only when the runbook marks a user action or permission gate.
```

The linked [agent setup runbook](docs/agent-setup.md) contains the full process.

## How it works

```text
your local exports
  -> normalize and filter
  -> pair incoming turns with your replies
  -> private SQLite/FTS index
  -> retrieve a few relevant examples
  -> capable model writes the complete reply
  -> optional local style renderer
  -> unchanged draft if any guard fails
```

The retriever supplies phrasing, rhythm, warmth, and level of formality. It is not a source of facts. The optional renderer never receives the incoming message and never decides what the reply should say.

## Privacy boundary

The project keeps generated data outside Git and creates private directories with mode `0700` and files with mode `0600`. Build, health, training, and aggregate evaluation commands avoid printing message text. `retrieve`, `try` (also available as `preview`), and `renderer-blind-next` are deliberate exceptions: they show private text to the active drafter or evaluator.

That does not make the data anonymous. Your corpus still contains your words and other people's messages. If a cloud-hosted agent receives retrieved examples, those examples have left your machine. A fine-tuned adapter can also memorize training text. Read [docs/privacy.md](docs/privacy.md) before using real data.

## Install

You need Python 3.11 or newer and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/hunterbohm/personal-voice.git
cd personal-voice
uv tool install .
personal-voice selftest
personal-voice
```

This puts `personal-voice` on your user PATH so agents can call it outside the checkout. For development, run `uv sync` and use `uv run personal-voice ...`.

The base retriever uses only the Python standard library and SQLite FTS5. The optional renderer requires a source checkout, Apple Silicon, and the extra MLX dependencies:

```bash
uv sync --extra renderer
```

## Prepare your sources

You can combine any of these inputs:

- iMessage JSONL produced by [texts-to-transformer](https://github.com/Doriandarko/texts-to-transformer)
- the ZIP file from X's official data archive
- decoded Gmail thread or message-page JSON exports

Create the private storage root explicitly so both levels use mode `0700`:

```bash
install -d -m 700 ~/.local/share/personal-voice
install -d -m 700 ~/.local/share/personal-voice/sources
```

For iMessage, follow the upstream snapshot and `prepare` workflow after checking out `texts-to-transformer` commit `90419977f4b0ab80c7c160faa2adcbcc2bfe7943`. The [agent setup runbook](docs/agent-setup.md) has the exact extraction commands. That extractor reads a consistent copy of `chat.db`, filters reactions and system rows, skips attachment-only messages, and replaces database identities with keyed pseudonyms. Copy the normalized output out of its Git worktree before building:

```bash
install -m 600 work/extracted/messages.jsonl \
  ~/.local/share/personal-voice/sources/imessage.jsonl
```

The normalized input contract is documented in [docs/data-contract.md](docs/data-contract.md).

For Gmail, install [`gogcli`](https://github.com/openclaw/gogcli) and create `gmail-owners.txt` with a private editor. Put one account address on each line and set the file to mode `0600`.

`gog auth add` requires the account address as a process argument. Run this OAuth step yourself in a private, non-recorded terminal. The variable keeps the address out of shell history, but it can still be visible in the local process list while `gog` runs:

```bash
IFS= read -r GMAIL_OWNER < \
  ~/.local/share/personal-voice/sources/gmail-owners.txt

gog auth add "$GMAIL_OWNER" \
  --services gmail --readonly --gmail-scope readonly

(
  umask 077
  cd ~/.local/share/personal-voice/sources
  GOG_ACCOUNT="$GMAIL_OWNER" gog gmail messages search 'in:sent' \
    --all --full --json --gmail-no-send --no-input > gmail-page.json
)

unset GMAIL_OWNER
```

Use `in:anywhere` only when you explicitly want incoming-thread context and accept exporting a much broader slice of the mailbox.

## Build the private index

Keep every source and output path outside the repository:

```bash
personal-voice build \
  --imessage ~/.local/share/personal-voice/sources/imessage.jsonl \
  --x-archive ~/.local/share/personal-voice/sources/x-archive.zip \
  --gmail-export ~/.local/share/personal-voice/sources/gmail-page.json \
  --gmail-owner-file ~/.local/share/personal-voice/sources/gmail-owners.txt \
  --output ~/.local/share/personal-voice/corpus.sqlite
```

Source flags may be repeated. Put one address per line in `gmail-owners.txt`. Source files must be mode `0600`, their directories must be mode `0700`, and they must live outside every Git worktree. The build stops before parsing when any source fails those checks.

X ingestion removes long outreach templates repeated across several conversations before pairing replies. Short messages are retained. Channels are retrieval hints, not separate personality profiles.

Check the index without printing its contents:

```bash
personal-voice doctor --json
personal-voice stats --json
```

## Retrieve examples

For private correspondence, omit `--query`. Start the process, write the request through stdin, then close stdin:

```text
command: personal-voice retrieve --medium text --limit 8 --json
stdin:
<incoming message, useful context, and what the reply should accomplish>
```

The JSON response contains private examples. Give them only to the model that is drafting the reply, and treat them as style evidence rather than factual context.

The reusable agent workflow is in [examples/personal-voice-skill/SKILL.md](examples/personal-voice-skill/SKILL.md). It is draft-only and never grants permission to send a message.

Install that file only into a runtime that is already present. Set `SKILL_ROOT` to that runtime's existing skill directory, and repeat for other detected runtimes. If a copy already exists, review its differences before replacing it.

```bash
SKILL_ROOT=~/.codex/skills
test -d "$SKILL_ROOT"
install -d -m 755 "$SKILL_ROOT/personal-voice"
install -m 644 examples/personal-voice-skill/SKILL.md \
  "$SKILL_ROOT/personal-voice/SKILL.md"
```

Start a new agent session after installation. Other agent runtimes can use the same `SKILL.md` as long as they can run local commands and write private input through process stdin.

To manually test an installed staged renderer, use the short interactive command below. It checks that the renderer is usable before accepting a draft, never sends anything, and keeps the original draft whenever a guard rejects the rewrite.

```bash
uv run personal-voice try
```

## Optional local renderer

The renderer is experimental. It trains a prompt-masked LoRA that maps a neutral, complete draft to your wording while deterministic guards preserve names, dates, numbers, links, questions, negations, and commitments.

Training completion is not an activation gate. The included workflow requires held-out evaluation, privacy probes, runtime evidence, semantic checks, and real blind comparisons before a candidate becomes the default.

See [docs/renderer.md](docs/renderer.md) for the full local workflow.

## What this project does not do

- It does not send, publish, or react to messages.
- It does not make a small style model responsible for reasoning.
- It does not claim pseudonymization is anonymization.
- It does not include a pretrained adapter or sample private corpus.
- It does not make long-form content eligible for the short-reply renderer.

## Development

```bash
uv sync --extra renderer
uv run pytest -q
uv run ruff check .
uv run python scripts/release_privacy_check.py
uv build
```

The privacy check inspects tracked files for private artifact types, oversized blobs, absolute home paths, non-example email addresses, and common credential patterns. It reports paths and categories, never matching content.

## License

Apache-2.0. See [LICENSE](LICENSE).
