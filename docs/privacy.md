# Privacy and threat model

Personal message archives are sensitive even after names and account IDs are replaced. Treat every source export, index, dataset, adapter, manifest, and evaluation pack as private.

## What the code enforces

- Corpus sources must be regular files, not symlinks.
- Sources and generated artifacts must live outside Git worktrees.
- Private directories use mode `0700`; private files use mode `0600`.
- Build, health, training, and evaluation reports expose aggregate values by default.
- Local model endpoints accept loopback HTTP only, with proxies and redirects disabled.
- Private drafts and retrieval queries can travel through process stdin instead of argv.
- The renderer has no sending capability and falls back to the original draft when a guard fails.

## What it cannot guarantee

Pseudonymization is not anonymization. Message text can contain names, addresses, employers, handles, secrets, relationship details, and recognizable events. The built-in normalizer redacts obvious email addresses, phone-number-shaped strings, and URLs. It does not remove every identifier.

The corpus also includes messages written by other people because their incoming turns provide reply context. Make sure your use complies with the consent, employment, confidentiality, and data-protection rules that apply to you.

Deletion is not guaranteed to erase every copy. Backups, synced folders, filesystem snapshots, terminal recordings, and SSD behavior can retain old data.

## Hosted models

The index is local. Retrieval results are not necessarily local.

When a hosted agent drafts a reply, the incoming message and retrieved examples may be sent to that provider. Review its data controls before enabling this workflow. If that boundary is unacceptable, use a local drafting model and local renderer.

Do not paste retrieval output into shared chats, issue trackers, telemetry systems, or logs. Retrieval intentionally prints private examples for the active drafting process.

## Shell and process exposure

Use stdin for private query text and renderer drafts. Command-line arguments may appear in shell history and process listings. Use `--gmail-owner-file` for real account addresses rather than `--gmail-owner`.

Terminal scrollback can still retain retrieval output, previews, and blind comparisons. Run private sessions without transcript capture when that matters.

## Fine-tuning risk

A LoRA adapter is derived private data. It may reproduce or partially memorize training text. Never publish an adapter trained on private communication.

The included privacy probe looks for several forms of exact overlap and compares tuned output with the untouched base model. A passing result lowers measured risk. It does not prove that memorization is impossible.

## Storage checklist

Before processing real data:

1. Use an encrypted local disk.
2. Keep source exports and generated files outside Git and cloud-synced folders.
3. Set directories to `0700` and files to `0600`.
4. Confirm the drafting model's data-retention policy.
5. Disable terminal recording, debug logs, and training telemetry.
6. Keep adapters and evaluation packs private after the experiment ends.

Before publishing a code change:

```bash
uv run python scripts/release_privacy_check.py
git status --short
git log --format='%an <%ae>' -1
```

The release check reports only the file path and issue category. It does not print matched content.
