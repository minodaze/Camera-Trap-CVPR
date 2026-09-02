# ICICLE: Camera Trap Continual Learning Benchmark

This repository contains the official implementation for the ECCV 2026 paper **"StreamTrap: A Temporal Camera Trap Benchmark for Continual Wildlife Monitoring"**. ICICLE provides a unified training and evaluation pipeline for studying continual learning on the **StreamTrap** benchmark — 546 camera traps spanning 17 LILA BC datasets across Africa, North America, South America, and Oceania.

## Overview

Camera trap deployments produce data streams where species distributions, habitats, and visual conditions change over time. StreamTrap captures this **temporal variability** by partitioning images into chronological intervals (15, 30, or 60 days), enabling realistic evaluation of continual adaptation.

Key features of this codebase:

- **Continual Learning (CL) strategies**: Zero-shot, Oracle, Accumulative, Sequential, Replay variants
- **Adaptation Recipe** (our proposal): BioCLIP-2 + LoRA + Balanced Softmax (BSM) — best overall
- **PEFT methods**: LoRA, Adapter (Pfeiffer, Houlsby), Visual Prompt Tuning (VPT), ConvPass, RepadapterV
- **Loss functions**: Cross-Entropy (CE), Balanced Softmax (BSM), Class-Balanced Focal (CB-Focal), CDT
- **Model backbones**: BioCLIP-2 (primary), BioCLIP, CLIP (OpenAI ViT-L/14)
- **WildIng** (optional): Geographic domain adaptation baseline used in rebuttal

---

## Quick Start

Reproduce the paper's **adaptation recipe** (accum⋆, Table 3) in three steps:

```bash
# 1 — Install
conda create -n ICICLE python=3.11 && conda activate ICICLE
pip install -r requirements.txt

# 2 — Symlink data
cd config/data && ln -s /path/to/camera-trap-benchmark/dataset_15_60 dataset && cd -

# 3 — Train (30-day intervals, single camera trap)
python run_pipeline.py \
  --c config/pipeline/<CAMERA_TRAP>/best_accum_30.yaml \
  --pretrained_weights bioclip2 \
  --lora_bottleneck 8 \
  --loss_type bsm \
  --text_template bioclip \
  --resume \
  --eval_per_epoch \
  --save_best_model
```

For batch submission across many camera traps, use the pre-built SLURM scripts in `script3/`.

---

## Table of Contents

1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [Configuration Files](#configuration-files)
4. [Running Experiments](#running-experiments)
   - [Zero-shot Baseline](#1-zero-shot-baseline)
   - [Oracle Upper Bound](#2-oracle-upper-bound)
   - [Accumulative (Our Recipe)](#3-accumulative-our-adaptation-recipe)
   - [Sequential Fine-tuning](#4-sequential-fine-tuning)
   - [Replay Baselines](#5-replay-baselines)
5. [PEFT Methods](#peft-methods)
6. [Loss Functions](#loss-functions)
7. [Using Different Models](#using-different-models)
8. [Batch Evaluation (Eval-Only Mode)](#batch-evaluation-eval-only-mode)
9. [WildIng (Optional)](#wilding-optional)
10. [Repository Structure](#repository-structure)
11. [Citation](#citation)

---

## Installation

**Prerequisites**: Python 3.11, CUDA-capable GPU (≥16 GB VRAM recommended)

```bash
# Create conda environment
conda create -n ICICLE python=3.11
conda activate ICICLE

# Install dependencies
pip install -r requirements.txt
```

**Pretrained weights**: Link the directory containing your pretrained weights:

```bash
ln -s /path/to/pretrained_weights pretrained_weights
```

BioCLIP-2 weights are downloaded automatically via `pybioclip`. For CLIP (OpenAI ViT-L/14), weights are fetched through `open_clip_torch`. SigLIP-2 weights must be downloaded manually from HuggingFace and placed at `pretrained_weights/siglip2-base-patch16-224`.

---

## Data Preparation

StreamTrap data follows the [LILA BC COCO camera trap format](https://lila.science/). Each camera trap's images are pre-cropped using MegaDetector v5a and stored as:

```
<dataset_root>/
  <CAMERA_TRAP_ID>/
    <interval>/          # e.g., 15, 30, or 60 (days per interval)
      train.json         # per-interval training splits
      test.json          # per-interval test splits
      train-all.json     # all training data combined (used for oracle)
```

**Symlink your data directory:**

```bash
cd config/data
ln -s /path/to/camera-trap-benchmark/dataset ./
```

**JSON format**: Each JSON follows the COCO camera trap schema with two additions:
- `ckp_<N>` keys per interval index (e.g., `ckp_0`, `ckp_1`, …, `ckp_K`)
- `common` and `scientific` species name fields per annotation

---

## Configuration Files

All experiments are controlled by a YAML config file. The basic structure is:

```yaml
module_name: <experiment_name>
log_path: <output_directory>

common_config:
  model: bioclip2                              # backbone choice
  train_data_config_path: <path/to/train.json>
  eval_data_config_path:  <path/to/test.json>
  all_data_config_path:   <path/to/train-all.json>  # used by oracle
  train_batch_size: 32
  eval_batch_size: 512
  optimizer_name: AdamW
  optimizer_params:
    lr: 2.5e-5
    weight_decay: 1e-4
  scheduler: CosineAnnealingLR
  scheduler_params:
    T_max: 60
    eta_min: 4.17e-7

pretrain_config:
  pretrain: false        # set true for oracle (see below)

ood_config:
  method: none

al_config:
  method: none

cl_config:
  method: none           # CL strategy — see options below
  epochs: 30
```

**`cl_config.method` options:**

| Method key | Description |
|---|---|
| `none` | Zero-shot — no training, direct evaluation |
| `accumulative-scratch` | Accumulate all data up to current interval, retrain from scratch each time (used for accum⋆) |
| `accumulative` | Accumulate data and continue fine-tuning from previous checkpoint |
| `naive-ft` | Sequential — train only on current interval's data |
| `replay` | CLEAR-style — 50:50 mix of current and buffered past data |
| `er1` | Experience Replay variant 1 |
| `er2` | Experience Replay variant 2 |
| `lwf` | Learning Without Forgetting |
| `mir` | Maximally Interfered Retrieval |
| `derpp` | Dark Experience Replay++ |
| `co2l` | Contrastive Continual Learning |

---

## Running Experiments

All experiments call `run_pipeline.py` with a config YAML and CLI flags. The SLURM sbatch scripts in `script3/` automate this for batch runs over many camera traps.

### 1. Zero-shot Baseline

No training — evaluate the pretrained BioCLIP-2 directly.

**Config** (`cl_config.method: none`):
```yaml
cl_config:
  method: none
```

**Command:**
```bash
python run_pipeline.py \
  --c config/pipeline/<CAMERA_TRAP>/zs.yaml \
  --pretrained_weights bioclip2 \
  --eval_per_epoch \
  --save_best_model
```

**Batch (SLURM):**
```bash
sbatch script3/sbatch_run_30_zs.sh 'DATASET1 DATASET2' 0.000025
```

---

### 2. Oracle Upper Bound

Train on **all** data from all intervals at once — the best achievable result with full data access.

**Config** (set `pretrain_config.pretrain: true` with `all_data_config_path`):
```yaml
pretrain_config:
  pretrain: true
  pretrain_data_config_path: <path/to/train-all.json>
  epochs: 30
  loss_type: bsm
cl_config:
  method: none
```

**Command:**
```bash
python run_pipeline.py \
  --c config/pipeline/<CAMERA_TRAP>/oracle.yaml \
  --pretrained_weights bioclip2 \
  --lora_bottleneck 8 \
  --loss_type bsm \
  --text_template bioclip \
  --eval_per_epoch \
  --save_best_model
```

**Batch (SLURM):**
```bash
sbatch script3/sbatch_run_best_oracle_30.sh 'DATASET1 DATASET2' 0.000025
```

---

### 3. Accumulative (Our Adaptation Recipe)

The **accum⋆** model in the paper: at each interval, retrain from scratch on all data accumulated so far, using LoRA + BSM.

**Config** (`cl_config.method: accumulative-scratch`):
```yaml
cl_config:
  method: accumulative-scratch
  epochs: 30
  loss_type: bsm
```

**Command (primary recipe — LoRA + BSM + BioCLIP-2):**
```bash
python run_pipeline.py \
  --c config/pipeline/<CAMERA_TRAP>/best_a.yaml \
  --pretrained_weights bioclip2 \
  --lora_bottleneck 8 \
  --loss_type bsm \
  --text_template bioclip \
  --resume \
  --eval_per_epoch \
  --save_best_model
```

**Batch (SLURM, 30-day intervals):**
```bash
sbatch script3/sbatch_run_best_accum_30_bio.sh 'DATASET1 DATASET2' 0.000025
```

For 15-day or 60-day intervals:
```bash
sbatch script3/sbatch_run_best_accum_15_bio.sh 'DATASET1 DATASET2' 0.000025
sbatch script3/sbatch_run_best_accum_60_bio.sh 'DATASET1 DATASET2' 0.000025
```

---

### 4. Sequential Fine-tuning

Train independently on each interval without retaining past data.

**Config** (`cl_config.method: naive-ft`):
```yaml
cl_config:
  method: naive-ft
  epochs: 30
```

**Command:**
```bash
python run_pipeline.py \
  --c config/pipeline/<CAMERA_TRAP>/seq.yaml \
  --pretrained_weights bioclip2 \
  --lora_bottleneck 8 \
  --loss_type bsm \
  --text_template bioclip \
  --eval_per_epoch \
  --save_best_model
```

**Batch (SLURM):**
```bash
sbatch script3/sbatch_run_best_seq.sh 'DATASET1 DATASET2' 0.000025
```

---

### 5. Replay Baselines

CLEAR-style replay mixes 50% current and 50% buffered past data.

**Config** (`cl_config.method: replay`):
```yaml
cl_config:
  method: replay
  buffer_size: 500     # number of samples in memory buffer
  epochs: 30
```

**Command:**
```bash
python run_pipeline.py \
  --c config/pipeline/<CAMERA_TRAP>/replay.yaml \
  --pretrained_weights bioclip2 \
  --lora_bottleneck 8 \
  --loss_type bsm \
  --text_template bioclip \
  --eval_per_epoch \
  --save_best_model
```

**Batch (SLURM):**
```bash
sbatch script3/sbatch_run_best_replay.sh   'DATASET1 DATASET2' 0.000025
sbatch script3/sbatch_run_best_replay-er1.sh 'DATASET1 DATASET2' 0.000025
sbatch script3/sbatch_run_best_replay-er2.sh 'DATASET1 DATASET2' 0.000025
```

---

## PEFT Methods

The paper ablates LoRA, Adapter, and VPT. All are compatible with any CL strategy above. Only the relevant CLI flags change.

### LoRA (best PEFT method, used in adaptation recipe)

```bash
python run_pipeline.py --c <config.yaml> \
  --pretrained_weights bioclip2 \
  --lora_bottleneck 8 \
  --loss_type bsm
```

### Full Fine-tuning (FFT)

```bash
python run_pipeline.py --c <config.yaml> \
  --pretrained_weights bioclip2 \
  --full \
  --loss_type ce
```

### Adapter — Pfeiffer (MLP only, sequential-after)

```bash
python run_pipeline.py --c <config.yaml> \
  --pretrained_weights bioclip2 \
  --ft_mlp_module adapter --ft_mlp_mode sequential_after \
  --adapter_bottleneck 8 --adapter_init zero --adapter_scaler 1
```

### Adapter — Houlsby (Attention + MLP)

```bash
python run_pipeline.py --c <config.yaml> \
  --pretrained_weights bioclip2 \
  --ft_attn_module adapter --ft_attn_mode sequential_after \
  --ft_mlp_module adapter  --ft_mlp_mode sequential_after \
  --adapter_bottleneck 8 --adapter_init xavier --adapter_scaler 1
```

### Visual Prompt Tuning (VPT-Deep)

```bash
python run_pipeline.py --c <config.yaml> \
  --pretrained_weights bioclip2 \
  --vpt_mode deep --vpt_num 10
```

### Text Encoder Fine-tuning

By default only the visual encoder and classification head are updated. To also fine-tune the text encoder:

| Mode | Flag |
|---|---|
| Head only (default) | `--text head` |
| Full text encoder | `--text full` |
| LoRA on text encoder | `--text lora --lora_bottleneck 8` |

---

## Loss Functions

Swap the `--loss_type` flag to compare imbalance-aware objectives:

| Loss | Flag | Notes |
|---|---|---|
| Cross-Entropy (CE) | `--loss_type ce` | Default baseline |
| Balanced Softmax (BSM) | `--loss_type bsm` | **Best** — no hyperparameters needed |
| Class-Balanced Focal | `--loss_type cb-focal` | `--loss_beta 0.999 --loss_gamma 0.5` |
| CDT | `--loss_type cdt` | `--loss_gamma 0.3` |
| Focal | `--loss_type focal` | `--loss_gamma <value>` |

Example (CB-Focal):
```bash
python run_pipeline.py --c <config.yaml> \
  --pretrained_weights bioclip2 \
  --lora_bottleneck 8 \
  --loss_type cb-focal \
  --loss_beta 0.999 --loss_gamma 0.5
```

---

## Using Different Models

Switch backbone with `--pretrained_weights`:

| Model | Flag | Notes |
|---|---|---|
| BioCLIP-2 | `--pretrained_weights bioclip2` | Default, best zero-shot (84.3%) |
| BioCLIP | `--pretrained_weights bioclip` | Older BioCLIP version |
| CLIP (OpenAI ViT-L/14) | `--pretrained_weights openai-ViT-L-14` | General-purpose baseline (72.3%) |

Use `--text_template bioclip` with BioCLIP models and `--text_template openai` with CLIP.

---

## Batch Evaluation (Eval-Only Mode)

Load saved `.pth` checkpoint files and evaluate them on test splits without retraining.
The pipeline auto-detects whether the saved run was an Oracle (`pretrain_best_model.pth`)
or an Accumulative run (`ckp_X_best_model.pth` files).

```bash
python run_pipeline.py \
  --c <config.yaml> \
  --eval_only \
  --model_dir <path/to/saved/log> \
  --pretrained_weights bioclip2 \
  --lora_bottleneck 8
```

**Optional flags for eval-only:**

| Flag | Effect |
|------|--------|
| `--accu_eval` | For each checkpoint's model, also evaluate it on all _future_ test intervals |
| `--save_predictions` | Dump raw `(preds, labels)` arrays to `.pkl` files |
| `--checkpoint_list 2 4 6` | Evaluate only specific interval indices |
| `--weight_sanity` | Cross-check balanced accuracy against `final_training_summary.json` (±0.03 tolerance) |
| `--skip_head` | Keep original head weights when loading (ignores saved head parameters) |

Results are saved to `eval_only_summary.json` in the same directory.

---

## WildIng (Optional)

WildIng is a geographic domain adaptation baseline used in the paper's rebuttal. It trains a lightweight MLP on top of BioCLIP image + VLM description embeddings.

**Step 1 — Precompute class description embeddings:**
```bash
python precompute_class_descriptions.py \
  --data_json <path/to/train.json> \
  --output config/LLM_description/class_centroids.pt
```

**Step 2 — Precompute per-image VLM description embeddings:**
```bash
python precompute_vlm_descriptions.py \
  --data_json <path/to/train.json> \
  --desc_json config/LLM_description/species_vlm_descriptions.json \
  --output config/LLM_description/vlm_desc_cache.pt
```

**Step 3 — Train WildIng MLP (standalone script):**
```bash
python train_wilding.py \
  --c <config.yaml> \
  --wilding_class_desc config/LLM_description/class_centroids.pt \
  --wilding_desc_cache config/LLM_description/vlm_desc_cache.pt \
  --wilding_alpha 0.5 \
  --wilding_epochs 30
```

> **Note:** WildIng is implemented as a separate standalone script (`train_wilding.py`).
> It is **not** part of the main `run_pipeline.py` dispatch path.

---

## Repository Structure

```
ICICLE/
├── run_pipeline.py              # Main training and evaluation entry point
├── train_wilding.py             # WildIng MLP training (optional, rebuttal)
├── precompute_class_descriptions.py  # Pre-encode class names for WildIng
├── precompute_vlm_descriptions.py    # Pre-encode VLM captions for WildIng
│
├── core/                        # Core implementation
│   ├── data.py                  # Dataset classes and transforms
│   ├── model.py                 # Model factory (CLIP/BioCLIP classifiers)
│   ├── common.py                # Training/eval loops, optimizers, schedulers
│   ├── loss.py                  # Loss functions (CE, BSM, CB-Focal, CDT, …)
│   ├── calibration.py           # Temperature calibration utilities
│   ├── wilding.py               # WildIng classifier (optional)
│   ├── wilding_dataset.py       # WildIng dataset (optional)
│   ├── speciesnet_model.py      # SpeciesNet zero-shot wrapper (optional)
│   ├── module/
│   │   ├── cl.py                # Continual Learning strategies
│   │   ├── al.py                # Active Learning modules
│   │   └── ood.py               # Out-of-Distribution detection
│   ├── open_clip/               # OpenCLIP library with LoRA/Adapter extensions
│   └── petl_model/              # PETL implementations (LoRA, Adapter, VPT, …)
│
├── config/
│   ├── pipeline/                # Per-camera-trap YAML configs
│   ├── data/                    # Symlinks to dataset JSON directories
│   └── LLM_description/        # Pre-encoded class/image embeddings (WildIng)
│
├── script3/                     # SLURM sbatch scripts for batch experiments
│   ├── sbatch_run_*_zs.sh       # Zero-shot evaluation (15/30/60-day intervals)
│   ├── sbatch_run_best_oracle_*.sh  # Oracle upper bound
│   ├── sbatch_run_best_accum_*_bio.sh  # Accumulative (our recipe)
│   ├── sbatch_run_best_replay*.sh  # Replay baselines
│   └── sbatch_run_best_seq.sh   # Sequential baseline
│
├── uselist/                     # Dataset lists for batch job submission
├── utils/                       # Logging, GPU monitoring, misc utilities
├── Long-CLIP/                   # Long-CLIP submodule (for WildIng VLM captions)
├── requirements.txt
└── environment.yml
```

---

## Hyperparameter Reference

The following hyperparameters reproduce the main paper results (Table 3):

| Hyperparameter | Value |
|---|---|
| Model | BioCLIP-2 |
| Batch size (train) | 32 |
| Optimizer | AdamW |
| Learning rate | 2.5 × 10⁻⁵ |
| Weight decay | 1 × 10⁻⁴ |
| LR scheduler | CosineAnnealingLR |
| Scheduler T_max | 60 |
| Min LR (η_min) | 4.17 × 10⁻⁷ |
| LoRA rank (r) | 8 |
| Loss | BSM |
| Epochs per interval | 30 |
| Text template | bioclip |

---

## Citation

```bibtex
@inproceedings{streamtrap2026,
  title     = {StreamTrap: A Temporal Camera Trap Benchmark for Continual Wildlife Monitoring},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026},
}
```
