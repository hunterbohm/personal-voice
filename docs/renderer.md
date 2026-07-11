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
- A local teacher model; the documented path uses Ollama with `qwen3.5:9b`
- Enough private disk space for datasets, checkpoints, and the base model
- 24 GiB unified memory recommended for the supplied training configuration

Use a source checkout for this workflow. The wheel contains the runtime package but not the repository-level training scripts and guides.

The original development configuration used `mlx-community/Qwen3-4B-Instruct-2507-4bit` as the renderer base. Download and pin a model revision yourself. Model licenses are separate from this repository's Apache-2.0 license, and no weights are redistributed here.

## 0. Preflight and local models

The copy/paste agent setup treats the renderer as supported only on macOS with `arm64`, at least 24 GiB of unified memory, and at least 20 GiB of free local disk before model downloads. Those are conservative setup gates, not proof that every training run will fit. A user can explicitly approve an experiment below them; an unattended agent must skip it.

Check the machine before downloading anything:

```bash
test "$(uname -s)" = Darwin
test "$(uname -m)" = arm64
test "$(sysctl -n hw.memsize)" -ge 25769803776
df -h ~/.local/share
```

Install the renderer dependencies from the source checkout:

```bash
uv sync --extra renderer
```

After the user approves the model license, network download, and disk use, fetch the exact public base revision with the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli):

```bash
MODEL_ID=mlx-community/Qwen3-4B-Instruct-2507-4bit
MODEL_REVISION=50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b
BASE_MODEL_PATH="$(uvx hf download "$MODEL_ID" \
  --revision "$MODEL_REVISION" --quiet)"
test -f "$BASE_MODEL_PATH/config.json"
test -f "$BASE_MODEL_PATH/model.safetensors"
```

Keep `MODEL_ID`, `MODEL_REVISION`, and `BASE_MODEL_PATH` in the same shell while you run the later steps. Set them again if you resume in a new shell.

The documented neutralization and semantic-audit path uses the 6.6 GB `qwen3.5:9b` Ollama model. After the user approves that download, install and start [Ollama for macOS](https://docs.ollama.com/macos), then run:

```bash
ollama pull qwen3.5:9b
ollama show qwen3.5:9b >/dev/null
```

Both model downloads are public artifacts. The corpus, renderer workspace, adapter, and evaluation evidence remain private local data.

## 1. Prepare private pairs

```bash
uv run personal-voice renderer-prepare
uv run personal-voice renderer-data-stats
```

The workspace samples reply targets by medium, caps single-conversation dominance, assigns whole conversations to train/validation/test splits, and removes duplicate-reply leakage across splits.

## 2. Create neutral drafts

The target is the writer's real reply. The input must express the same meaning in plain wording so the model learns style rather than how to answer.

Use the loopback-only Ollama path prepared above:

```bash
uv run personal-voice renderer-neutralize --model qwen3.5:9b
```

Alternatively, use a separate pinned local MLX teacher that the user has approved:

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
  --model "$BASE_MODEL_PATH"

VERSION=my-version
uv run python scripts/train_renderer.py \
  --model "$BASE_MODEL_PATH" \
  --data ~/.local/share/personal-voice/renderer/data/mlx \
  --adapter ~/.local/share/personal-voice/renderer/runs/$VERSION/adapter
```

Keep `VERSION` in the same shell as the staging and evaluation commands below, or set it again before resuming.

Training uses prompt masking so loss applies only to the target reply. Telemetry and network model loading are disabled. Review `training/renderer-lora.yaml` for the supplied starting configuration; it is not a universal optimum.

## 4. Stage and evaluate

```bash
uv run personal-voice renderer-install \
  --model-path "$BASE_MODEL_PATH" \
  --adapter-path ~/.local/share/personal-voice/renderer/runs/$VERSION/adapter \
  --version "$VERSION" \
  --base-model "$MODEL_ID" \
  --base-revision "$MODEL_REVISION"

uv run personal-voice renderer-doctor \
  --manifest ~/.local/share/personal-voice/renderer/versions/$VERSION.json

uv run personal-voice renderer-evaluate \
  --manifest ~/.local/share/personal-voice/renderer/versions/$VERSION.json \
  --base-only

uv run personal-voice renderer-evaluate \
  --manifest ~/.local/share/personal-voice/renderer/versions/$VERSION.json

uv run python scripts/renderer_semantic_gates.py \
  --manifest ~/.local/share/personal-voice/renderer/versions/$VERSION.json

uv run python scripts/benchmark_renderer.py \
  --manifest ~/.local/share/personal-voice/renderer/versions/$VERSION.json

uv run python scripts/renderer_privacy_probe.py \
  --manifest ~/.local/share/personal-voice/renderer/versions/$VERSION.json \
  --data ~/.local/share/personal-voice/renderer/data/mlx \
  --count 1000
```

Use `uv run personal-voice try` from the renderer-enabled checkout for a local manual comparison (`preview` remains an alias). The command validates the renderer before accepting a draft. It does not run the caller's semantic check and never activates a model.

## 5. Blind evaluation and later activation

Blind evaluation randomizes the retrieval-augmented draft and rendered draft. It records preference and whether either draft is sendable without revealing labels during voting.

The current experimental promotion policy expects qualifying cases for text, DM, email, and public replies. Text-only users can run retrieval and manual previews, but they cannot satisfy automatic renderer activation without adapting that policy. This limitation is intentional in the first public release.

The copy/paste setup runbook ends with a staged, inactive candidate. It does not automate blind voting or call `renderer-activate`. Real blind cases require the user's frontier-plus-retrieval drafts and direct human ratings over time; setup cannot manufacture that evidence. Use the `renderer-blind-* --help` commands only in a later, user-supervised evaluation task.

Never copy a private adapter, dataset, manifest, privacy report, runtime report, or blind pack into this repository.
