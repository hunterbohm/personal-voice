# Contributing

Contributions are welcome, but private communication must never enter the repository.

## Rules for fixtures

- Use invented conversations only.
- Use `example.com` for email addresses.
- Use reserved `555` phone numbers when a formatted number is necessary.
- Do not paste a redacted real message. Write a new synthetic sentence instead.
- Do not attach archives, databases, JSONL corpora, screenshots, adapters, weights, or evaluation packs.

## Local checks

```bash
uv sync --extra renderer
uv run pytest -q
uv run ruff check .
uv run python scripts/release_privacy_check.py
uv build
```

Pull requests that change a data contract, renderer protocol, manifest schema, privacy report, or activation gate must include focused tests and documentation.
