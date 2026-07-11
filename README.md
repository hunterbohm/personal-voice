# personal-voice

`personal-voice` helps an AI draft in your writing style without turning a small style model into the brain of your agent.

It builds a private SQLite index from messages you wrote, retrieves relevant examples for the current conversation, and leaves intent and factual reasoning to the calling model. An optional local MLX renderer can make one guarded wording pass over the completed draft.

This repository contains code and synthetic tests only. It contains no messages, email exports, contact identifiers, datasets, adapters, model weights, or private evaluation results.

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

The project keeps generated data outside Git and creates private directories with mode `0700` and files with mode `0600`. Build, health, training, and aggregate evaluation commands avoid printing message text. `retrieve`, `preview`, and `renderer-blind-next` are deliberate exceptions: they show private text to the active drafter or evaluator.

That does not make the data anonymous. Your corpus still contains your words and other people's messages. If a cloud-hosted agent receives retrieved examples, those examples have left your machine. A fine-tuned adapter can also memorize training text. Read [docs/privacy.md](docs/privacy.md) before using real data.

## Install

You need Python 3.11 or newer and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/hunterbohm/personal-voice.git
cd personal-voice
uv tool install .
personal-voice selftest
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

For iMessage, follow the upstream snapshot and `prepare` workflow. This release was tested against `texts-to-transformer` commit `90419977f4b0ab80c7c160faa2adcbcc2bfe7943`. That extractor reads a consistent copy of `chat.db`, filters reactions and system rows, skips attachment-only messages, and replaces database identities with keyed pseudonyms. Copy the normalized output out of its Git worktree before building:

```bash
install -d -m 700 ~/.local/share/personal-voice/sources
install -m 600 work/extracted/messages.jsonl \
  ~/.local/share/personal-voice/sources/imessage.jsonl
```

The normalized input contract is documented in [docs/data-contract.md](docs/data-contract.md).

For Gmail, install [`gogcli`](https://github.com/openclaw/gogcli), authorize read-only Gmail access, and export complete message JSON inside the private source directory:

```bash
gog auth add you@example.com \
  --services gmail --readonly --gmail-scope readonly

(
  umask 077
  cd ~/.local/share/personal-voice/sources
  GOG_ACCOUNT=you@example.com gog gmail messages search 'in:anywhere' \
    --all --full --json --gmail-no-send --no-input > gmail-page.json
)
```

Use `in:sent` instead when you want authored style examples without incoming-thread context. Put the same account address in `gmail-owners.txt`, one address per line, and keep that file mode `0600`.

## Build the private index

Keep every source and output path outside the repository:

```bash
uv run personal-voice build \
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
uv run personal-voice doctor --json
uv run personal-voice stats --json
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

Install that file in whichever runtimes should draft with the corpus:

```bash
for root in ~/.codex/skills ~/.claude/skills ~/.hermes/skills; do
  install -d "$root/personal-voice"
  install -m 644 examples/personal-voice-skill/SKILL.md \
    "$root/personal-voice/SKILL.md"
done
```

Start a new agent session after installation. Other agent runtimes can use the same `SKILL.md` as long as they can run local commands and write private input through process stdin.

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
