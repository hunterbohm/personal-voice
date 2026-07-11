# Experimental local renderer

The renderer is an optional style-transfer stage for short replies. It does not replace the capable model that decides what to say.

```text
complete draft from the calling model
  -> local 4-bit base model plus private LoRA
  -> literal and structure guards
  -> semantic comparison by the caller
  -> rewritten draft or original fallback
```

## Requirements

- Apple Silicon Mac
- Python 3.11 or newer
- `mlx-lm==0.31.1` and `transformers==5.5.0`
- A pinned local 4-bit MLX instruct model snapshot
- Enough private disk space for datasets, checkpoints, and the base model
- 24 GiB unified memory recommended for the supplied training configuration

Use a source checkout for this workflow. The wheel contains the runtime package but not the repository-level training scripts and guides.

The original development configuration used `mlx-community/Qwen3-4B-Instruct-2507-4bit` as the renderer base. Download and pin a model revision yourself. Model licenses are separate from this repository's Apache-2.0 license, and no weights are redistributed here.

## 1. Prepare private pairs

```bash
uv sync --extra renderer
uv run personal-voice renderer-prepare
uv run personal-voice renderer-data-stats
```

The workspace samples reply targets by medium, caps single-conversation dominance, assigns whole conversations to train/validation/test splits, and removes duplicate-reply leakage across splits.

## 2. Create neutral drafts

The target is the writer's real reply. The input must express the same meaning in plain wording so the model learns style rather than how to answer.

Use either the loopback-only Ollama path:

```bash
uv run personal-voice renderer-neutralize --model qwen3.5:9b
```

or a pinned local MLX teacher:

```bash
uv run python scripts/neutralize_renderer_mlx.py \
  --model /private/local/model/snapshot
```

Then run the semantic audit:

```bash
uv run personal-voice renderer-audit --model qwen3.5:9b --batch-size 8
```

The audit checks facts, questions, negation, uncertainty, permission, deadlines, and commitments. An unavailable judge leaves rows pending rather than approving them.

## 3. Export and train

```bash
uv run personal-voice renderer-export \
  --model /private/local/model/snapshot

uv run python scripts/train_renderer.py \
  --model /private/local/model/snapshot \
  --data ~/.local/share/personal-voice/renderer/data/mlx \
  --adapter ~/.local/share/personal-voice/renderer/runs/my-version/adapter
```

Training uses prompt masking so loss applies only to the target reply. Telemetry and network model loading are disabled. Review `training/renderer-lora.yaml` for the supplied starting configuration; it is not a universal optimum.

## 4. Stage and evaluate

```bash
uv run personal-voice renderer-install \
  --model-path /private/local/model/snapshot \
  --adapter-path ~/.local/share/personal-voice/renderer/runs/my-version/adapter \
  --version my-version \
  --base-model <model-id> \
  --base-revision <pinned-revision>

uv run personal-voice renderer-doctor \
  --manifest ~/.local/share/personal-voice/renderer/versions/my-version.json

uv run personal-voice renderer-evaluate \
  --manifest ~/.local/share/personal-voice/renderer/versions/my-version.json

uv run python scripts/renderer_semantic_gates.py \
  --manifest ~/.local/share/personal-voice/renderer/versions/my-version.json

uv run python scripts/benchmark_renderer.py \
  --manifest ~/.local/share/personal-voice/renderer/versions/my-version.json

uv run python scripts/renderer_privacy_probe.py \
  --manifest ~/.local/share/personal-voice/renderer/versions/my-version.json \
  --data ~/.local/share/personal-voice/renderer/data/mlx \
  --count 1000
```

Use `uv run personal-voice try` from the renderer-enabled checkout for a local manual comparison (`preview` remains an alias). The command validates the renderer before accepting a draft. It does not run the caller's semantic check and never activates a model.

## 5. Blind evaluation and activation

Blind evaluation randomizes the retrieval-augmented draft and rendered draft. It records preference and whether either draft is sendable without revealing labels during voting.

The current experimental promotion policy expects qualifying cases for text, DM, email, and public replies. Text-only users can run retrieval and manual previews, but they cannot satisfy automatic renderer activation without adapting that policy. This limitation is intentional in the first public release.

Never copy a private adapter, dataset, manifest, privacy report, runtime report, or blind pack into this repository.
