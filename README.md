# NTIRE 2026 — Robust AI-Generated Image Detection

Deep learning solution for detecting AI-generated images with robustness to real-world post-processing transformations (JPEG compression, blur, resize).

**Course project for IMSECU 2026 (EURECOM).**

---

## Challenge Overview

| | |
|---|---|
| **Task** | Binary classification — Real vs. AI-generated images |
| **Primary Metric** | Robust ROC AUC (images with post-processing) |
| **Secondary Metric** | Clean ROC AUC (original images) |
| **Training Set** | ~277k images across 6 shards |
| **Validation Set** | 10k clean images + 2.5k hard variants |
| **Platform** | [CodaBench Competition #12761](https://www.codabench.org/competitions/12761/) |

---

## Approach

### Models

This repository implements three architectures:

- **CLIPDetector** — CLIP ViT-B/16 backbone with a linear classifier head (baseline).
- **DINOv2Detector** — Frozen DINOv2 backbone with a lightweight classifier head.
- **TransFusionNet** — Region-aligned CLIP + DINOv2 token fusion with spatial transformer classifier (our main model).

### TransFusionNet Architecture

Rather than combining final logits from each backbone, TransFusionNet fuses patch-level representations from both models:

1. **Token extraction** — Extract patch tokens (not CLS) from CLIP and DINOv2 last hidden states.
2. **Shared projection** — Two linear layers project both token sets into a common 512-d space (`proj_clip`, `proj_dino`).
3. **Spatial alignment** — Patch grids differ in resolution (different patch sizes), so tokens are reshaped into `(B, F, H, W)` grids and aligned to a shared `(Ht, Wt)` via bicubic interpolation. This ensures the same spatial index maps to the same image region.
4. **Per-location stream fusion** — For each location `p`, fuse the pair `[d_tok[p], c_tok[p]]` with multi-head attention over the stream axis (length 2). Intuition (per location):
   - `new_dino = α * dino + (1-α) * clip`
   - `new_clip = β * clip + (1-β) * dino`

   Then residual + LayerNorm and a mean over the 2 streams produce a single fused token per location.
5. **Spatial transformer + classification** — A learnable `[CLS]` token is prepended to the fused tokens. A Transformer encoder models global context, and the final prediction comes from the transformed CLS embedding through a linear head.

---

## Repository Structure

```
ntire2026-detection/
├── data/                        # Dataset (not tracked)
│   ├── train/                   # 6 shards (~277k images)
│   └── val/                     # Validation sets (clean + hard)
├── src/                         # Source code
│   ├── dataset.py               # Data loading & augmentation
│   ├── models.py                # Model architectures
│   ├── train.py                 # Training utilities
│   └── inference.py             # Prediction utilities
├── scripts/                     # CLI entry points
│   ├── train_cli.py             # Training script
│   └── inference_cli.py         # Inference & submission script
├── notebooks/                   # Early experiments (CLIP-only, single-backbone)
├── checkpoints/                 # Trained models (not tracked)
├── submissions/                 # Submission CSVs (not tracked)
├── requirements.txt
└── README.md
```
> **Note:** The `notebooks/` folder contains early-stage experiments where we prototyped and trained the CLIP-only baseline. They are kept for reference and reproducibility of the initial approach, but the final training and inference pipeline lives in `scripts/`.

### Dataset Download

| Split | Link |
|---|---|
| **Training** (6 shards, ~277k images) | [HuggingFace — NTIRE-RobustAIGenDetection-train](https://huggingface.co/datasets/deepfakesMSU/NTIRE-RobustAIGenDetection-train/tree/main) |
| **Validation** (clean + hard) | [HuggingFace — NTIRE-RobustAIGenDetection-val](https://huggingface.co/datasets/deepfakesMSU/NTIRE-RobustAIGenDetection-val/tree/main) |

Download and place the data under `data/train/` and `data/val/` respectively.

## Reproducibility Guide

This section walks through the full pipeline to reproduce our results from scratch.

### Step 0 — Environment setup

Create and activate a virtual environment:

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Step 1 — Train from scratch

Train TransFusionNet on all 6 shards for N epochs:

```bash
python scripts/train_cli.py \
    --model transfusion \
    --shards all \
    --epochs N \
    --batch-size 16 \
    --lr 1e-4 \
    --data-root "data/train" \
    --wandb \
    --wandb-tags transfusion,allshards
```

This produces checkpoints in `checkpoints/transfusion_epoch{1..N}.pth` and `transfusion_best.pth`.

> **Optional — Weights & Biases:** The `--wandb` flag enables logging to [Weights & Biases](https://wandb.ai/), a platform for tracking training metrics (loss, AUC, learning rate) in real-time with interactive charts. Requires a free account. Remove the flag to train without it.

### Step 2 — Resume training

To continue training from a checkpoint (e.g. extend from epoch 6 to epoch 10):

```bash
python scripts/train_cli.py \
    --model transfusion \
    --resume checkpoints/transfusion_epoch6.pth \
    --shards all \
    --epochs 10 \
    --batch-size 16 \
    --lr 1e-4 \
    --data-root "data/train" \
    --wandb \
    --wandb-tags transfusion,allshards
```

> **Note on the learning rate scheduler:** When resuming, the script recreates `CosineAnnealingLR` with `T_max = total_epochs - completed_epochs` (e.g. `T_max=4` when going from epoch 6 to 10). This ensures the learning rate decays properly over the remaining epochs rather than staying at ~0 from the previous cosine cycle.

### Step 3 — Inference

Generate submission CSVs for CodaBench evaluation:

```bash
# Clean validation set
python scripts/inference_cli.py \
    --model transfusion \
    --model-path checkpoints/transfusion_best.pth \
    --images-dir data/val/official/images \
    --output-csv submissions/transfusion_clean.csv

# Hard (post-processed) validation set
python scripts/inference_cli.py \
    --model transfusion \
    --model-path checkpoints/transfusion_best.pth \
    --images-dir data/val/official/val_images_hard \
    --output-csv submissions/transfusion_hard.csv
```

The script outputs a CodaBench-formatted CSV ready for submission.

### Step 4 — Submit to CodaBench

Package the inference CSVs into a ZIP archive for submission:
```bash
cd submissions
cp codabench_transfusion_clean.csv submission.csv
cp codabench_transfusion_hard.csv submission_hard.csv
zip submission.zip submission.csv submission_hard.csv
```

Upload `submission.zip` to the [CodaBench competition page](https://www.codabench.org/competitions/12761/).



### CLI Reference

#### `train_cli.py`

| Flag | Default | Description |
|---|---|---|
| `--model` | `dino` | Model: `clip`, `dino`, `transfusion` |
| `--data-root` | `../data/train` | Path to training data |
| `--shards` | `all` | Shard selection: `all`, `0`, `0,1,2` |
| `--epochs` | `N` | Total number of epochs |
| `--batch-size` | `16` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--accumulation-steps` | `2` | Gradient accumulation steps |
| `--val-split` | `0.1` | Validation split ratio |
| `--weight-decay` | `0.01` | AdamW weight decay |
| `--resume` | — | Checkpoint path to resume from |
| `--checkpoint-dir` | `checkpoints` | Output directory for checkpoints |
| `--save-prefix` | model name | Checkpoint file prefix |
| `--seed` | `42` | Random seed |
| `--device` | `auto` | `auto`, `cuda`, `mps`, `cpu` |
| `--wandb` | off | Enable W&B logging |

#### `inference_cli.py`

| Flag | Default | Description |
|---|---|---|
| `--model` | required | Model: `clip`, `dino`, `transfusion` |
| `--model-path` | required | Path to `.pth` checkpoint |
| `--images-dir` | required | Directory containing test images |
| `--output-csv` | required | Output CSV path |
| `--device` | `auto` | `auto`, `cuda`, `mps`, `cpu` |

---

## Results

### CodaBench Leaderboard (TransFusionNet)

| Clean ROC AUC | Robust ROC AUC | Clean Hard ROC AUC | Robust Hard ROC AUC |
|:---:|:---:|:---:|:---:|
| **0.9343** | **0.9149** | **0.8717** | **0.8367** |

### Training Progression

| Epoch | Train Loss | Val AUC |
|:---:|:---:|:---:|
| 1 | 0.4379 | 0.9213 |
| 2 | 0.3873 | 0.9433 |
| 3 | 0.3612 | 0.9487 |
| 4 | 0.3383 | 0.9562 |
| 5 | 0.3200 | 0.9671 |
| 6 | 0.3071 | 0.9692 |
| 7 | 0.3034 | 0.9684 |
| 8 | 0.3013 | 0.9701 |
| 9 | 0.2980 | 0.9701 |
| 10 | 0.2969 | **0.9703** |

---

## Hardware

Developed and trained on:

- **MacBook Pro M4 Pro** — 14-core CPU, 20-core GPU, 16-core Neural Engine, 24 GB unified memory
- ~12 hours per epoch on full dataset (all 6 shards, ~277k images)

---

## References

- "Raising the Bar of AI-generated Image Detection with CLIP" (CVPR 2024W)
- "A Bias-Free Training Paradigm for More General AI-generated Image Detection" (CVPR 2025)

---

## Authors

### GroupID on Codabench: ImSecuGroup3DipGosGauKim

Antonello Di Pede · Dario Gosmar · Andrea Gaudino · Sean Kim

## License

This project is licensed under the [MIT License](LICENSE).