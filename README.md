# DINOv3 PyTorch Lightning Training

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red.svg)
![Lightning](https://img.shields.io/badge/Lightning-2.5.0-purple.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A comprehensive PyTorch Lightning implementation of DINOv3 self-supervised learning with support for prototype analysis, continued pretraining, GRAM loss, and multi-dataset training pipelines.

> **Built upon the original [DINOv3](https://github.com/facebookresearch/dinov3) by Meta AI Research** - This implementation extends DINOv3 with PyTorch Lightning integration, prototype analysis sweeps, GRAM loss support, and automated training pipelines while maintaining full compatibility with official pretrained weights.

## Features

- **PyTorch Lightning Integration**: Clean, modular code with automatic multi-GPU support
- **Prototype Analysis Pipeline**: Systematic sweep over prototype counts with automated result aggregation
- **Continued Pretraining**: Initialize from DINOv3 official weights for domain adaptation
- **GRAM Loss Support**: Gradient-based Regularization with Auxiliary Model
- **Multi-Dataset Support**: DTD, EuroSAT, Oxford Pets, NCTCRCHE100K, Tissue, ImageNet-1K
- **SLURM Integration**: Ready-to-use scripts for HPC cluster deployment
- **Flexible Sampling**: Infinite, distributed, epoch-based, and sharded-infinite samplers
- **Comprehensive Logging**: TensorBoard, CSV, and real-time progress tracking

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supported Datasets](#supported-datasets)
- [Training Modes](#training-modes)
- [Prototype Analysis Pipeline](#prototype-analysis-pipeline)
- [SLURM Cluster Usage](#slurm-cluster-usage)
- [Configuration Reference](#configuration-reference)
- [Output Structure](#output-structure)
- [Troubleshooting](#troubleshooting)

## Installation

### 1. Clone Repository

```bash
git clone --recurse-submodules https://github.com/marjanstoimchev/DinoV3LightningTraining.git
cd DinoV3LightningTraining

# If you didn't use --recurse-submodules:
git submodule update --init --recursive
```

### 2. Create Environment

```bash
conda env create -f environment.yml
conda activate dinov3_lightning
```

### 3. Download DINOv3 Weights (Optional)

For continued pretraining, download the official DINOv3 weights:

```bash
mkdir -p dinov3_official_weights
wget -O dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
```

## Quick Start

### Single Dataset SSL Pretraining

```bash
# From scratch
python src/training/ssl/train.py \
    --config-file configs/eurosat/config_ssl_pretraining.yaml \
    --output-dir ./output/eurosat \
    --gpus 2 \
    --precision bf16-mixed

# Continued pretraining from DINOv3 weights
python src/training/ssl/train.py \
    --config-file configs/eurosat/config_ssl_continued_pretraining.yaml \
    --output-dir ./output/eurosat_continued \
    --gpus 2 \
    --precision bf16-mixed
```

### Classification/Fine-tuning

```bash
python scripts/classification/train.py \
    --config configs/eurosat/config_classification.yaml \
    --pretrained_path ./output/eurosat/checkpoints/pretraining/last.ckpt \
    --max_epochs 30 \
    --devices 0,1
```

## Supported Datasets

| Dataset | Config Directory | HuggingFace Path | Classes | Images |
|---------|-----------------|------------------|---------|--------|
| **DTD** | `configs/DTD/` | `cansa/Describable-Textures-Dataset-DTD` | 47 | ~5.6K |
| **EuroSAT** | `configs/eurosat/` | `blanchon/EuroSAT_RGB` | 10 | ~27K |
| **Oxford Pets** | `configs/oxford_pets/` | `timm/oxford-iiit-pet` | 37 | ~7.3K |
| **NCTCRCHE100K** | `configs/NCTCRCHE100K/` | `DykeF/NCTCRCHE100K` | 9 | ~100K |
| **Tissue** | `configs/tissue/` | Custom TIFF | 10 | Custom |
| **ImageNet-1K** | `configs/imagenet1k/` | `ILSVRC/imagenet-1k` | 1000 | ~1.28M |

Each dataset has three configuration files:
- `config_ssl_pretraining.yaml` - Training from scratch
- `config_ssl_continued_pretraining.yaml` - Continued pretraining from DINOv3
- `config_classification.yaml` - Downstream classification

## Training Modes

### 1. SSL Pretraining from Scratch

Train a DINOv3 model from random initialization:

```bash
python src/training/ssl/train.py \
    --config-file configs/DTD/config_ssl_pretraining.yaml \
    --output-dir ./output/dtd_scratch \
    --gpus 3 \
    --num-prototypes 4096 \
    --max-epochs 100 \
    --precision bf16-mixed \
    --sampler-type distributed
```

### 2. Continued Pretraining from DINOv3

Initialize from official DINOv3 weights for domain adaptation:

```bash
python src/training/ssl/train.py \
    --config-file configs/DTD/config_ssl_continued_pretraining.yaml \
    --output-dir ./output/dtd_continued \
    --gpus 3 \
    --max-epochs 100 \
    --precision bf16-mixed
```

### 3. SSL Pretraining with GRAM Loss

Enable GRAM loss for enhanced regularization:

```bash
python src/training/ssl/train.py \
    --config-file configs/DTD/config_ssl_pretraining.yaml \
    --output-dir ./output/dtd_gram \
    --gpus 3 \
    --enable-gram \
    --gram-weight 1.0 \
    --precision bf16-mixed
```

### 4. Classification Fine-tuning

Fine-tune a pretrained model for classification:

```bash
python scripts/classification/train.py \
    --config configs/DTD/config_classification.yaml \
    --pretrained_path ./output/dtd_scratch/checkpoints/pretraining/last.ckpt \
    --max_epochs 30 \
    --learning_rate 0.0001 \
    --devices 0,1 \
    --encoder_type teacher \
    --precision bf16-mixed
```

### 5. Linear Evaluation

Evaluate with frozen backbone:

```bash
python scripts/classification/train.py \
    --config configs/DTD/config_classification.yaml \
    --pretrained_path ./output/dtd_scratch/checkpoints/pretraining/last.ckpt \
    --max_epochs 30 \
    --devices 0 \
    --freeze_backbone
```

## Prototype Analysis Pipeline

The prototype analysis pipeline systematically evaluates the effect of prototype count on model performance.

### Non-SLURM Usage

#### Basic Sweep

```bash
./scripts/prototype_analysis/run_sweep.sh \
    --dataset dtd \
    --gpus 0,1,2 \
    --prototypes "128 256 512 1024" \
    --seeds "0 1 42" \
    --pretrain-epochs 100 \
    --classify-epochs 30
```

#### Multi-Dataset Sweep

```bash
./scripts/prototype_analysis/run_sweep.sh \
    --datasets "dtd eurosat oxford_pets" \
    --gpus 0,1,2,3 \
    --prototypes "128 256 512 1024 2048 4096" \
    --seeds "0 1 42" \
    --classify-mode finetune \
    --precision bf16-mixed
```

#### Continued Pretraining Sweep

```bash
./scripts/prototype_analysis/run_sweep.sh \
    --dataset dtd \
    --gpus 0,1,2 \
    --prototypes "128 256 512 1024" \
    --seeds "0 1 42" \
    --pretrain-checkpoint dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --output-dir prototype_analysis_continued
```

#### With GRAM Loss

```bash
./scripts/prototype_analysis/run_sweep.sh \
    --dataset dtd \
    --gpus 0,1,2 \
    --prototypes "128 256 512" \
    --seeds "0 1 42" \
    --enable-gram \
    --gram-weight 0.5
```

### run_sweep.sh Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--datasets` | Space-separated dataset names | Required |
| `--dataset` | Single dataset (backward compatible) | - |
| `--gpus` | Comma-separated GPU indices | Required |
| `--prototypes` | Space-separated prototype counts | `"128 256 512 1024 2048 4096"` |
| `--seeds` | Classification seeds | `"0 1 42"` |
| `--pretrain-seed` | Fixed pretraining seed | `42` |
| `--pretrain-epochs` | Pretraining epochs | Config default |
| `--pretrain-checkpoint` | DINOv3 checkpoint for continued pretraining | - |
| `--skip-pretraining` | Skip pretraining, use checkpoint directly | `false` |
| `--classify-epochs` | Classification epochs | `30` |
| `--classify-lr` | Classification learning rate | `0.0001` |
| `--classify-mode` | `finetune` or `lineareval` | `finetune` |
| `--batch-size` | Batch size | `128` |
| `--koleo-weight` | KoLeo loss weight | `0.1` |
| `--enable-gram` | Enable GRAM loss | `false` |
| `--gram-weight` | GRAM loss weight | `1.0` |
| `--output-dir` | Output directory | `prototype_analysis_dinov3` |
| `--precision` | Training precision | `bf16-mixed` |
| `--compile` | Enable PyTorch 2.0 compilation | `false` |

## SLURM Cluster Usage

### Basic SLURM Submission

```bash
sbatch scripts/prototype_analysis/run_sweep_slurm.sh
```

### With Environment Variables

```bash
# Single dataset
DATASETS="dtd" GPUS="0,1,2" PROTOTYPES="128 256 512" \
    sbatch scripts/prototype_analysis/run_sweep_slurm.sh

# Multiple datasets
DATASETS="dtd eurosat oxford_pets" GPUS="0,1,2,3" \
    sbatch scripts/prototype_analysis/run_sweep_slurm.sh

# Continued pretraining
CONTINUED_PRETRAINING=1 DATASETS="dtd" PROTOTYPES="128 256 512" \
    sbatch scripts/prototype_analysis/run_sweep_slurm.sh

# With GRAM loss
ENABLE_GRAM=1 GRAM_WEIGHT=0.5 DATASETS="dtd" \
    sbatch scripts/prototype_analysis/run_sweep_slurm.sh

# Full customization
DATASETS="eurosat dtd" \
GPUS="0,1,2" \
PROTOTYPES="128 256 512 1024" \
SEEDS="0 1 42" \
PRETRAIN_EPOCHS=100 \
CLASSIFY_EPOCHS=30 \
CLASSIFY_MODE=finetune \
BATCH_SIZE=128 \
PRECISION=bf16-mixed \
CONTINUED_PRETRAINING=1 \
ENABLE_GRAM=1 \
GRAM_WEIGHT=0.5 \
OUTPUT_DIR=my_experiment \
    sbatch scripts/prototype_analysis/run_sweep_slurm.sh
```

### SLURM Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATASETS` | Space-separated dataset names | `dtd` |
| `GPUS` | Comma-separated GPU indices | `0,1,2` |
| `PROTOTYPES` | Space-separated prototype counts | `"128 256 512 1024"` |
| `SEEDS` | Classification seeds | `"0 1 42"` |
| `PRETRAIN_SEED` | Pretraining seed | `42` |
| `PRETRAIN_EPOCHS` | Pretraining epochs | `100` |
| `CLASSIFY_EPOCHS` | Classification epochs | `30` |
| `CLASSIFY_LR` | Classification learning rate | `0.0001` |
| `CLASSIFY_MODE` | `finetune` or `lineareval` | `finetune` |
| `BATCH_SIZE` | Batch size | `128` |
| `KOLEO_WEIGHT` | KoLeo loss weight | `0.1` |
| `CONTINUED_PRETRAINING` | Enable continued pretraining | - |
| `PRETRAIN_CHECKPOINT` | Custom checkpoint path | Auto-set if `CONTINUED_PRETRAINING` |
| `ENABLE_GRAM` | Enable GRAM loss | - |
| `GRAM_WEIGHT` | GRAM loss weight | - |
| `OUTPUT_DIR` | Output directory | `prototype_analysis_dinov3` |
| `PRECISION` | Training precision | `bf16-mixed` |
| `COMPILE` | Enable compilation | - |
| `SIF_IMAGE` | Singularity image path | `$HOME/deeplearning.sif` |

### SLURM Configuration

The default SLURM configuration in `run_sweep_slurm.sh`:

```bash
#SBATCH --job-name=dinov3-proto-sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:3
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --partition=e8
```

Modify these directly in the script for your cluster.

## Configuration Reference

### SSL Pretraining Config Structure

```yaml
# Model architecture
student:
  arch: vit_small              # vit_small, vit_base, vit_large
  patch_size: 16               # Patch size
  drop_path_rate: 0.3          # Drop path rate
  pretrained_weights: ''       # Path to pretrained weights (for continued pretraining)
  n_storage_tokens: 4          # Number of register tokens

# DINO head
dino:
  head_n_prototypes: 65536     # Number of prototypes
  koleo_loss_weight: 0.1       # KoLeo loss weight
  head_bottleneck_dim: 256     # Bottleneck dimension
  head_hidden_dim: 2048        # Hidden dimension

# iBOT (masked image modeling)
ibot:
  loss_weight: 1.0             # iBOT loss weight
  mask_ratio_min_max: [0.1, 0.5]  # Masking ratio range

# GRAM loss (optional)
gram:
  use_loss: false              # Enable GRAM loss
  loss_weight: 1.0             # GRAM loss weight

# Optimization
optim:
  epochs: 100                  # Training epochs
  lr: 0.001                    # Learning rate
  weight_decay: 0.04           # Weight decay
  warmup_epochs: 10            # Warmup epochs

# Data augmentation
crops:
  global_crops_size: 256       # Global crop size
  local_crops_size: 112        # Local crop size
  local_crops_number: 8        # Number of local crops

# Training
train:
  batch_size_per_gpu: 128      # Batch size per GPU
  dataset_path: HuggingFace:name=dataset-name
```

## Output Structure

### Prototype Analysis Output

```
prototype_analysis_dinov3/
├── pretraining/
│   └── {dataset}/
│       └── proto_{N}/
│           ├── checkpoints/
│           │   └── last.ckpt
│           └── logs/
│
├── classification/
│   └── {dataset}/
│       └── proto_{N}/
│           └── seed_{S}/
│               ├── checkpoints/
│               ├── logs/
│               └── results/
│                   ├── predictions.npy
│                   ├── confusion_matrix.csv
│                   └── per_class_metrics.csv
│
results/
└── prototype_analysis_dinov3/
    └── {dataset}/
        ├── classification_results.csv    # Per-run results
        └── classification_stats.csv      # Aggregated statistics
```

### Result Files

- **classification_results.csv**: Per-run results with columns for prototype count, seed, accuracy, top5_accuracy, f1_macro, and other metrics
- **classification_stats.csv**: Aggregated statistics with mean, std, and count across seeds for each prototype configuration

## Repository Structure

```
DinoV3LightningTraining/
├── configs/                           # Dataset configurations
│   ├── DTD/
│   │   ├── config_ssl_pretraining.yaml
│   │   ├── config_ssl_continued_pretraining.yaml
│   │   └── config_classification.yaml
│   ├── eurosat/
│   ├── oxford_pets/
│   ├── NCTCRCHE100K/
│   ├── tissue/
│   └── imagenet1k/
│
├── scripts/
│   ├── pretraining/
│   │   └── train.py                   # Pretraining wrapper
│   ├── classification/
│   │   ├── train.py                   # Classification training
│   │   └── eval.py                    # Evaluation
│   └── prototype_analysis/
│       ├── run_sweep.sh               # Main sweep script
│       ├── run_sweep_slurm.sh         # SLURM submission
│       ├── sweep_pretraining.py       # Pretraining sweep
│       └── sweep_classification.py    # Classification sweep
│
├── src/
│   ├── training/
│   │   └── ssl/
│   │       └── train.py               # SSL training script
│   ├── ssl/
│   │   ├── models/
│   │   │   └── ssl_learner.py         # PyTorch Lightning module
│   │   └── data/
│   │       └── datamodule.py          # Data loading
│   ├── classification/
│   ├── callbacks/
│   └── checkpointing/
│
├── dinov3/                            # DINOv3 submodule
├── dinov3_official_weights/           # Pretrained weights
├── notebooks/                         # Analysis notebooks
├── environment.yml                    # Conda environment
└── requirements.txt                   # Python dependencies
```

## Troubleshooting

### Common Issues

**ImportError: No module named 'dinov3'**
```bash
git submodule update --init --recursive
```

**CUDA out of memory**
```bash
# Reduce batch size
--batch-size 64

# Or use gradient accumulation
--accumulate-grad-batches 2
```

**Weights not loading (NaN values)**
```bash
# Ensure pretrained_weights path is correct in config
# For continued pretraining configs, verify the path:
student:
  pretrained_weights: dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
```

**SLURM job fails immediately**
```bash
# Check Singularity image exists
ls -la $HOME/deeplearning.sif

# Or specify custom path
SIF_IMAGE=/path/to/your/container.sif sbatch run_sweep_slurm.sh
```

**DataLoader hangs**
```bash
# Reduce num_workers in config
train:
  num_workers: 4  # Instead of 10
```

### Verifying Pretrained Weights

```python
import torch
weights = torch.load('dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth')
print(f"Keys: {list(weights.keys())[:5]}")
print(f"CLS token shape: {weights['cls_token'].shape}")  # Should be [1, 1, 384]
```

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir ./output/tensorboard_logs
```

### Real-time Progress

Training displays real-time metrics:

```
Epoch 1/100 | Step 384/832 (46.2%) | ETA: 2.1h | Speed: 5.72 it/s
DINO_L: 9.000 | DINO_G: 9.000 | KOLEO: -0.274 | IBOT: 2.254 | GRAM: 0.845 | total_loss: 12.095
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [DINOv3 original implementation](https://github.com/facebookresearch/dinov3) by Meta AI
- [PyTorch Lightning](https://lightning.ai/) framework
- The self-supervised learning community

---

**Happy Training!**
