# DFlash Draft Model Training Guide

This document describes the two-phase training pipeline for DFlash draft models used in speculative decoding with an autoregressive target model.

## Overview

DFlash is a diffusion-based draft model that predicts multiple tokens in parallel for speculative decoding. Training happens in two phases:

1. **Phase 1 — Pretraining (PT):** Train the draft model from scratch on a broad, diverse dataset.
2. **Phase 2 — Supervised Fine-Tuning (SFT):** Specialize the draft model for a target domain (e.g., agent conversations) using a Phase 1 checkpoint.

The target model (e.g., Llama-3.1-8B-Instruct) remains frozen throughout. Only the draft model is trained.

---

## Architecture

### Draft Model

The DFlash draft model is a small transformer that takes:
- **Target hidden states** from selected layers of the frozen target model
- **Noise embeddings** where only the anchor token (position 0 of each block) is real; remaining positions are MASK tokens

It predicts `block_size - 1` tokens per block in parallel. With the default `block_size=16`, each block predicts 15 tokens.

**Example config** (`configs/llama3-8b-dflash.json`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_hidden_layers` | 5 | Draft transformer layers |
| `num_target_layers` | 32 | Target model layers (for feature extraction) |
| `hidden_size` | 4096 | Hidden dimension (matches target) |
| `block_size` | 16 | Tokens per speculative block |
| `target_layer_ids` | [1, 7, 15, 23, 31] | Target layers to extract hidden states from |
| `mask_token_id` | 128002 | MASK token for noise input |

### Target Model Backend

Two backends are supported for generating target hidden states during training:

- **SGLang** (`--target-model-backend sglang`): Production-grade, supports tensor parallelism. Recommended for multi-GPU training.
- **HuggingFace** (`--target-model-backend hf`): Simpler setup, useful for development and debugging.

---

## Training Mechanics

### Block-Parallel Training

During each forward pass:

1. **Anchor sampling:** Randomly sample up to `num_anchors` (default: 512) positions from the sequence where `loss_mask > 0`.
2. **Noise embedding:** For each anchor, create a block of `block_size` tokens. Position 0 is the real anchor token; positions 1 through `block_size-1` are MASK tokens.
3. **Attention mask:** Each block attends to context tokens strictly before its anchor position (causal w.r.t. context) and has bidirectional attention within the block. Different blocks cannot attend to each other.
4. **Draft forward:** All blocks are processed in a single forward pass (parallel prediction).
5. **Loss:** Cross-entropy between predicted tokens and actual tokens at each position, excluding position 0 (the anchor).

### Anchor Token

The anchor token (position 0 in each block) is the seed — it's the last token the target model has committed to. The draft model uses it to predict the next `block_size - 1` tokens. This mirrors inference, where the last accepted token becomes the anchor for the next speculative block.

### Loss Decay Weighting

An optional exponential decay weights the loss across block positions:

```
weight[k] = exp(-(k-1) / gamma)    for k = 1, 2, ..., block_size-1
```

This prioritizes earlier positions (which matter more for acceptance length). Recommended gamma values:
- `block_size=16`: gamma = 7.0
- `block_size=10`: gamma = 5.0
- `block_size=8`: gamma = 4.0

### Optimizer

- **AdamW** with FP32 optimizer state and BF16 model parameters for numerical stability
- **LR schedule:** Linear warmup followed by cosine annealing decay to 0
- **Gradient clipping:** `max_grad_norm=1.0`

### Metrics

Three metrics are tracked during training and reported to wandb:

| Metric | Description |
|--------|-------------|
| `loss` | Weighted cross-entropy loss |
| `accuracy` | Fraction of correctly predicted tokens across all block positions |
| `acceptance_length` | Average consecutive correct predictions per block from position 1 onward, +1 for the bonus token. Simulates speculative decoding acceptance. Range: 1 (all rejected) to `block_size` (all accepted). |

---

## Phase 1 — Pretraining (PT)

Train the draft model from scratch on a broad dataset to learn general token prediction patterns of the target model.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| `--num-epochs` | 6 |
| `--batch-size` | 4 |
| `--learning-rate` | 6e-4 |
| `--warmup-ratio` | 0.04 |
| `--max-grad-norm` | 1.0 |
| `--max-length` | 3072 |
| `--block-size` | 16 |
| `--num-anchors` | 512 |
| `--loss-decay-gamma` | 7.0 |
| `--log-interval` | 50 |
| `--save-interval` | 1000 |

### Data

A single broad-distribution dataset (e.g., `perfectblend_llama3.1-8b_regen.jsonl`) covering diverse text domains. Sequences shorter than `2 * block_size` loss tokens are filtered out.

### Running

**Single node:**
```bash
bash examples/run_llama3.1_8b_dflash_online.sh [NUM_GPUS]
```

**Multi-node:**
```bash
# On each node:
bash examples/run_llama3.1_8b_dflash_online_multinode.sh \
    [NUM_GPUS] [NUM_NODES] [NODE_RANK] [MASTER_ADDR] [MASTER_PORT]
```

### What to look for

- **Loss** should decrease steadily across epochs
- **Accuracy** should increase, typically reaching 0.50–0.70 depending on the dataset
- **Acceptance length** should increase from ~1 toward 3–5+ as training progresses
- Select the best checkpoint based on eval loss/acceptance length if eval data is provided

---

## Phase 2 — Supervised Fine-Tuning (SFT)

Specialize the draft model for a target domain by fine-tuning from a Phase 1 checkpoint.

### Key Differences from Phase 1

| Parameter | Phase 1 (PT) | Phase 2 (SFT) |
|-----------|-------------|----------------|
| `--num-epochs` | 6 | 3 |
| `--learning-rate` | 6e-4 | 1e-4 (6x lower) |
| `--warmup-ratio` | 0.04 | 0.06 (slightly higher) |
| `--log-interval` | 50 | 25 (more frequent) |
| `--save-interval` | 1000 | 500 (more frequent) |
| Initialization | Random | Phase 1 checkpoint |
| `--reset-scheduler` | N/A | Enabled |

### Checkpoint Loading

Phase 2 loads from a Phase 1 checkpoint via `--ckpt-dir`:

- **Draft model weights:** Loaded from the checkpoint's `pytorch_model.bin`
- **Optimizer Adam state (momentum):** Loaded from `training_state.pt` to preserve gradient history
- **LR scheduler:** Reset for a fresh warmup + cosine decay schedule over the new total steps (`--reset-scheduler`)

This means the optimizer "remembers" gradient statistics from Phase 1 but starts a fresh learning rate trajectory appropriate for the shorter Phase 2 training.

### Data

Phase 2 typically uses a mix of:
- **Domain-specific data** (e.g., agent conversations) — the primary training signal
- **Phase 1 data replay** at a reduced fraction — prevents catastrophic forgetting

The `::` syntax specifies sampling fractions:
```
--train-data-path \
    "pretraining_data.jsonl::0.1" \     # 10% of Phase 1 data
    "agent_data.jsonl"                   # 100% of domain data
```

### Running

```bash
export PHASE1_CKPT_DIR=/path/to/outputs/llama3.1-8b-dflash-perfectblend/epoch_6_step_XXXX

bash examples/run_llama3.1_8b_dflash_online_multinode_phase2.sh \
    [NUM_GPUS] [NUM_NODES] [NODE_RANK] [MASTER_ADDR] [MASTER_PORT]
```

### What to look for

- **Loss** may initially spike slightly then decrease below Phase 1 levels on domain data
- **Acceptance length on domain data** should exceed Phase 1's acceptance length, since the drafter is now specialized
- Monitor eval metrics if `--eval-data-path` is provided to detect overfitting

---

## Evaluation

Evaluation is optional and runs periodically during training when `--eval-data-path` is provided.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--eval-data-path` | None | Path to evaluation dataset (jsonl format) |
| `--eval-interval` | 1000 | Run eval every N training steps |

During eval, the draft model is set to `.eval()` mode and all metrics (loss, accuracy, acceptance length) are computed under `torch.no_grad()` across the full eval set, averaged, and logged to wandb as `eval/loss`, `eval/accuracy`, and `eval/acceptance_length`.

---

## Experiment Tracking

Tracking is configured via `--report-to`. Supported backends:

| Backend | Flag |
|---------|------|
| Weights & Biases | `--report-to wandb` |
| TensorBoard | `--report-to tensorboard` |
| SwanLab | `--report-to swanlab` |
| MLflow | `--report-to mlflow` |
| None | `--report-to none` |

For wandb, additionally set:
- `--wandb-project`: Project name
- `--wandb-name`: Run name
- `--wandb-key` or `WANDB_API_KEY` env var: API key (if not already logged in)

### Logged Metrics

| Metric | Mode | Description |
|--------|------|-------------|
| `train/loss` | Train | Weighted cross-entropy loss |
| `train/accuracy` | Train | Token prediction accuracy |
| `train/acceptance_length` | Train | Simulated speculative acceptance length |
| `train/lr` | Train | Current learning rate |
| `eval/loss` | Eval | Eval loss (if eval data provided) |
| `eval/accuracy` | Eval | Eval accuracy (if eval data provided) |
| `eval/acceptance_length` | Eval | Eval acceptance length (if eval data provided) |

---

## Checkpointing

Checkpoints are saved every `--save-interval` steps and at the end of training. Each checkpoint directory (`epoch_{N}_step_{M}/`) contains:

| File | Contents |
|------|----------|
| `pytorch_model.bin` | Draft model weights |
| `config.json` | Draft model configuration |
| `training_state.pt` | Optimizer state, scheduler state, epoch, global step |
| `dflash.py` | Copy of the model class code for reproducibility |

### Resuming Training

To resume from the last checkpoint in the output directory:
```bash
--resume
```

To resume from a specific checkpoint:
```bash
--ckpt-dir /path/to/checkpoint/epoch_3_step_6000
```

---

## Distributed Training

Training uses PyTorch FSDP (Fully Sharded Data Parallel) with `SHARD_GRAD_OP` strategy and BF16 mixed precision. The target model can optionally use tensor parallelism via `--tp-size`.

Launch with `torchrun`:
```bash
torchrun \
    --nnodes $NUM_NODES \
    --node-rank $NODE_RANK \
    --nproc_per_node $NUM_GPUS \
    --master-addr $MASTER_ADDR \
    --master-port $MASTER_PORT \
    scripts/train_dflash.py \
    [args...]
```

---

## Quick Reference: All Arguments

### Model
| Argument | Default | Description |
|----------|---------|-------------|
| `--target-model-path` | required | HuggingFace model path for target model |
| `--target-model-backend` | `hf` | `sglang` or `hf` |
| `--draft-config-path` | None | Path to draft model config JSON |
| `--block-size` | 16 | Tokens per speculative block |
| `--num-draft-layers` | 1 | Draft transformer layers (if no config provided) |
| `--mask-token-id` | None | MASK token ID (auto-detect if not set) |
| `--attention-backend` | `flex_attention` | `eager`, `sdpa`, or `flex_attention` |
| `--num-anchors` | 512 | Anchor positions per sequence |
| `--loss-decay-gamma` | None | Exponential loss decay gamma |
| `--trust-remote-code` | False | Trust remote code for model loading |

### Dataset
| Argument | Default | Description |
|----------|---------|-------------|
| `--train-data-path` | required | Training data paths (supports `path::fraction` syntax) |
| `--eval-data-path` | None | Evaluation data path |
| `--chat-template` | `qwen` | Chat template name |
| `--is-preformatted` | False | Skip chat template formatting |
| `--dataloader-num-workers` | 8 | DataLoader workers |
| `--build-dataset-num-proc` | 8 | Dataset preprocessing workers |

### Training
| Argument | Default | Description |
|----------|---------|-------------|
| `--num-epochs` | 6 | Training epochs |
| `--batch-size` | 1 | Batch size per GPU |
| `--learning-rate` | 6e-4 | Peak learning rate |
| `--max-length` | 3072 | Max sequence length |
| `--warmup-ratio` | 0.04 | Fraction of steps for LR warmup |
| `--max-grad-norm` | 1.0 | Gradient clipping threshold |
| `--accumulation-steps` | 1 | Gradient accumulation steps |
| `--seed` | 42 | Random seed |
| `--resume` | False | Auto-resume from last checkpoint |
| `--ckpt-dir` | None | Resume from specific checkpoint |
| `--reset-scheduler` | False | Reset LR schedule (keep optimizer state) |

### Output
| Argument | Default | Description |
|----------|---------|-------------|
| `--output-dir` | required | Output directory for checkpoints |
| `--cache-dir` | `./cache` | Cache directory for processed datasets |
| `--log-interval` | 50 | Log metrics every N steps |
| `--eval-interval` | 1000 | Run eval every N steps |
| `--save-interval` | 1000 | Save checkpoint every N steps |

### Distributed
| Argument | Default | Description |
|----------|---------|-------------|
| `--tp-size` | 1 | Tensor parallel size for target model |
| `--dist-timeout` | 30 | Distributed timeout in minutes |
