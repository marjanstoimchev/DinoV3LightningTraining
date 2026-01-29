# Usage Examples

This document provides comprehensive examples for training DINOv3 models with PyTorch Lightning.

## Basic Examples

### 1. Single GPU Training (Testing)

Fastest way to test the framework:

```bash
python src/training/ssl/train.py \
    --config-file configs/eurosat/config_ssl_pretraining.yaml \
    --output-dir ./output_test \
    --gpus 1 \
    --max-epochs 5 \
    --sampler-type infinite \
    --limit-train-batches 0.1
```

### 2. Multi-GPU Training (Production)

Full-scale training on 3 GPUs:

```bash
python src/training/ssl/train.py \
    --config-file configs/eurosat/config_ssl_pretraining.yaml \
    --output-dir ./output_eurosat \
    --gpus 3 \
    --strategy ddp \
    --sampler-type distributed \
    --batch-size 384 \
    --max-epochs 100 \
    --precision bf16-mixed
```

### 3. Resume from Checkpoint

Continue training from a saved checkpoint:

```bash
python src/training/ssl/train.py \
    --config-file configs/eurosat/config_ssl_pretraining.yaml \
    --resume-from-checkpoint ./output_eurosat/checkpoints/last.ckpt \
    --output-dir ./output_resumed \
    --gpus 3 \
    --strategy ddp
```

### 4. Continued Pretraining from DINOv3

Initialize from official DINOv3 weights for domain adaptation:

```bash
python src/training/ssl/train.py \
    --config-file configs/eurosat/config_ssl_continued_pretraining.yaml \
    --output-dir ./output_eurosat_continued \
    --gpus 3 \
    --precision bf16-mixed
```

## Dataset-Specific Examples

### EuroSAT (Satellite Imagery)

```bash
# From scratch
python src/training/ssl/train.py \
    --config-file configs/eurosat/config_ssl_pretraining.yaml \
    --output-dir ./output_eurosat \
    --gpus 3 \
    --strategy ddp \
    --sampler-type distributed \
    --batch-size 384 \
    --max-epochs 100

# Continued pretraining
python src/training/ssl/train.py \
    --config-file configs/eurosat/config_ssl_continued_pretraining.yaml \
    --output-dir ./output_eurosat_continued \
    --gpus 3 \
    --precision bf16-mixed
```

### DTD (Describable Textures)

```bash
# From scratch
python src/training/ssl/train.py \
    --config-file configs/DTD/config_ssl_pretraining.yaml \
    --output-dir ./output_dtd \
    --gpus 3 \
    --sampler-type distributed \
    --max-epochs 100

# With custom prototype count
python src/training/ssl/train.py \
    --config-file configs/DTD/config_ssl_pretraining.yaml \
    --output-dir ./output_dtd_custom \
    --gpus 3 \
    --num-prototypes 4096 \
    --koleo-weight 0.1
```

### Oxford Pets

```bash
python src/training/ssl/train.py \
    --config-file configs/oxford_pets/config_ssl_pretraining.yaml \
    --output-dir ./output_pets \
    --gpus 2 \
    --sampler-type distributed \
    --max-epochs 100
```

### NCTCRCHE100K (Histopathology)

```bash
# Large dataset - use distributed sampler
python src/training/ssl/train.py \
    --config-file configs/NCTCRCHE100K/config_ssl_pretraining.yaml \
    --output-dir ./output_nctcrche \
    --gpus 4 \
    --strategy ddp \
    --sampler-type distributed \
    --batch-size 512 \
    --max-epochs 100 \
    --precision bf16-mixed
```

### Tissue (Custom TIFF Dataset)

```bash
python src/training/ssl/train.py \
    --config-file configs/tissue/config_ssl_pretraining.yaml \
    --output-dir ./output_tissue \
    --gpus 3 \
    --sampler-type epoch \
    --max-epochs 200
```

### ImageNet-1K (Large Scale)

```bash
# Large-scale pretraining
python src/training/ssl/train.py \
    --config-file configs/imagenet1k/config_ssl_pretraining.yaml \
    --output-dir ./output_imagenet \
    --gpus 8 \
    --strategy ddp \
    --sampler-type distributed \
    --batch-size 1024 \
    --precision bf16-mixed \
    --compile
```

## Architecture-Specific Examples

### ViT-Small (Default)

```bash
# Fastest training, good for experimentation
python src/training/ssl/train.py \
    --config-file configs/eurosat/config_ssl_pretraining.yaml \
    --output-dir ./output_vit_small \
    --gpus 3 \
    --batch-size 384 \
    --sampler-type distributed
```

### ViT-Base (Higher Capacity)

Modify the config file to use ViT-Base:

```yaml
# In config file:
student:
  arch: vit_base
  patch_size: 14
```

```bash
python src/training/ssl/train.py \
    --config-file configs/eurosat/config_ssl_pretraining_vitbase.yaml \
    --output-dir ./output_vit_base \
    --gpus 4 \
    --batch-size 256 \  # Reduced for larger model
    --sampler-type distributed
```

### ViT-Large (Maximum Performance)

```yaml
# In config file:
student:
  arch: vit_large
  patch_size: 14
```

```bash
python src/training/ssl/train.py \
    --config-file configs/eurosat/config_ssl_pretraining_vitlarge.yaml \
    --output-dir ./output_vit_large \
    --gpus 8 \
    --batch-size 128 \  # Small batch for large model
    --accumulate-grad-batches 2 \
    --sampler-type distributed
```

## Sampler Comparison Examples

### Infinite Sampler (SSL Training)

```bash
# Best for continuous streaming, self-supervised learning
python src/training/ssl/train.py \
    --config-file configs/eurosat/config_ssl_pretraining.yaml \
    --output-dir ./output_infinite \
    --gpus 3 \
    --sampler-type infinite \
    --batch-size 384

# Progress shown as: Step 384/70 (epoch length from config)
# Uses OFFICIAL_EPOCH_LENGTH from config
```

### Distributed Sampler (Multi-GPU Efficiency)

```bash
# Best for multi-GPU training efficiency
python src/training/ssl/train.py \
    --config-file configs/eurosat/config_ssl_pretraining.yaml \
    --output-dir ./output_distributed \
    --gpus 3 \
    --strategy ddp \
    --sampler-type distributed \
    --batch-size 384

# Dataset automatically split across GPUs
```

### Epoch Sampler (Traditional Training)

```bash
# Traditional epoch-based training
python src/training/ssl/train.py \
    --config-file configs/eurosat/config_ssl_pretraining.yaml \
    --output-dir ./output_epoch \
    --gpus 3 \
    --sampler-type epoch \
    --batch-size 384

# Each GPU sees the full dataset per epoch
```

## GRAM Loss Examples

### Enable GRAM Loss

```bash
python src/training/ssl/train.py \
    --config-file configs/eurosat/config_ssl_pretraining.yaml \
    --output-dir ./output_gram \
    --gpus 3 \
    --enable-gram \
    --gram-weight 1.0 \
    --precision bf16-mixed
```

### GRAM with Custom Weight

```bash
python src/training/ssl/train.py \
    --config-file configs/eurosat/config_ssl_pretraining.yaml \
    --output-dir ./output_gram_custom \
    --gpus 3 \
    --enable-gram \
    --gram-weight 0.5 \
    --precision bf16-mixed
```

## Prototype Analysis Examples

### Basic Sweep

```bash
./scripts/prototype_analysis/run_sweep.sh \
    --dataset eurosat \
    --gpus 0,1,2 \
    --prototypes "128 256 512 1024" \
    --seeds "0 1 42" \
    --pretrain-epochs 100 \
    --classify-epochs 30
```

### Multi-Dataset Sweep

```bash
./scripts/prototype_analysis/run_sweep.sh \
    --datasets "dtd eurosat oxford_pets" \
    --gpus 0,1,2,3 \
    --prototypes "128 256 512 1024 2048 4096" \
    --seeds "0 1 42" \
    --classify-mode finetune \
    --precision bf16-mixed
```

### Continued Pretraining Sweep

```bash
./scripts/prototype_analysis/run_sweep.sh \
    --dataset dtd \
    --gpus 0,1,2 \
    --prototypes "128 256 512 1024" \
    --seeds "0 1 42" \
    --pretrain-checkpoint dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --output-dir prototype_analysis_continued
```

### Sweep with GRAM Loss

```bash
./scripts/prototype_analysis/run_sweep.sh \
    --dataset dtd \
    --gpus 0,1,2 \
    --prototypes "128 256 512" \
    --seeds "0 1 42" \
    --enable-gram \
    --gram-weight 0.5
```

## Classification Examples

### Fine-tuning

```bash
python scripts/classification/train.py \
    --config configs/eurosat/config_classification.yaml \
    --pretrained_path ./output_eurosat/checkpoints/pretraining/last.ckpt \
    --max_epochs 30 \
    --learning_rate 0.0001 \
    --devices 0,1 \
    --encoder_type teacher \
    --precision bf16-mixed
```

### Linear Evaluation (Frozen Backbone)

```bash
python scripts/classification/train.py \
    --config configs/eurosat/config_classification.yaml \
    --pretrained_path ./output_eurosat/checkpoints/pretraining/last.ckpt \
    --max_epochs 30 \
    --devices 0 \
    --freeze_backbone
```

### Evaluation Only

```bash
python scripts/classification/eval.py \
    --config configs/eurosat/config_classification.yaml \
    --checkpoint_path ./output_classification/checkpoints/best.ckpt \
    --devices 0
```

## Performance Optimization Examples

### Memory-Optimized Training

```bash
# For limited GPU memory
python src/training/ssl/train.py \
    --config-file configs/eurosat/config_ssl_pretraining.yaml \
    --output-dir ./output_memory_opt \
    --gpus 3 \
    --batch-size 192 \  # Smaller total batch
    --accumulate-grad-batches 2 \  # Simulate larger batch
    --precision bf16-mixed \
    --sampler-type distributed
```

### Speed-Optimized Training

```bash
# For maximum speed with PyTorch 2.0 compilation
python src/training/ssl/train.py \
    --config-file configs/eurosat/config_ssl_pretraining.yaml \
    --output-dir ./output_speed_opt \
    --gpus 3 \
    --batch-size 384 \
    --compile \
    --sampler-type distributed \
    --precision bf16-mixed
```

### Large-Scale Training

```bash
# Multi-node, high-throughput training
python src/training/ssl/train.py \
    --config-file configs/imagenet1k/config_ssl_pretraining.yaml \
    --output-dir ./output_largescale \
    --gpus 8 \
    --num-nodes 2 \  # 16 GPUs total
    --batch-size 2048 \
    --strategy ddp \
    --sampler-type distributed \
    --precision bf16-mixed \
    --save-every-n-steps 1000
```

## Debugging and Development

### Fast Development Iteration

```bash
# Quick training for code testing
python src/training/ssl/train.py \
    --config-file configs/eurosat/config_ssl_pretraining.yaml \
    --output-dir ./output_debug \
    --gpus 1 \
    --fast-dev-run \
    --sampler-type infinite
```

### Limited Training for Testing

```bash
# Train on subset of data
python src/training/ssl/train.py \
    --config-file configs/eurosat/config_ssl_pretraining.yaml \
    --output-dir ./output_subset \
    --gpus 1 \
    --limit-train-batches 0.01 \  # Only 1% of data
    --max-epochs 3 \
    --sampler-type epoch
```

### Profiling Training

```bash
# With detailed logging
python src/training/ssl/train.py \
    --config-file configs/eurosat/config_ssl_pretraining.yaml \
    --output-dir ./output_profile \
    --gpus 1 \
    --max-epochs 1 \
    --limit-train-batches 100 \
    --log-every-n-steps 1
```

## Monitoring Examples

### TensorBoard Monitoring

```bash
# Start training
python src/training/ssl/train.py \
    --config-file configs/eurosat/config_ssl_pretraining.yaml \
    --output-dir ./output

# In another terminal, start TensorBoard
tensorboard --logdir ./output/tensorboard_logs --port 6006

# Open browser: http://localhost:6006
```

### CSV Logging Analysis

```python
# Analyze training logs
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV logs
logs = pd.read_csv('./output/csv_logs/metrics.csv')

# Plot training losses
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(logs['step'], logs['total_loss'])
plt.title('Total Loss')

plt.subplot(1, 2, 2)
plt.plot(logs['step'], logs['dino_local_loss'], label='DINO Local')
plt.plot(logs['step'], logs['ibot_loss'], label='iBOT')
plt.legend()
plt.title('Component Losses')
plt.show()
```

## Common Workflow Examples

### Experiment Pipeline

```bash
# 1. Quick test on small dataset
python src/training/ssl/train.py \
    --config-file configs/dtd/config_ssl_pretraining.yaml \
    --output-dir ./output_test \
    --gpus 1 --max-epochs 5 --limit-train-batches 0.1

# 2. Medium-scale validation
python src/training/ssl/train.py \
    --config-file configs/dtd/config_ssl_pretraining.yaml \
    --output-dir ./output_validation \
    --gpus 2 --max-epochs 20 --batch-size 256

# 3. Full production training
python src/training/ssl/train.py \
    --config-file configs/dtd/config_ssl_pretraining.yaml \
    --output-dir ./output_production \
    --gpus 3 --max-epochs 100 --batch-size 384 --strategy ddp
```

### Hyperparameter Search

```bash
# Grid search over prototype counts
for proto in 128 256 512 1024 2048 4096; do
    python src/training/ssl/train.py \
        --config-file configs/eurosat/config_ssl_pretraining.yaml \
        --output-dir ./output_proto_${proto} \
        --gpus 3 \
        --max-epochs 100 \
        --num-prototypes $proto
done
```

### Domain Adaptation Pipeline

```bash
# 1. Start from DINOv3 pretrained weights
python src/training/ssl/train.py \
    --config-file configs/tissue/config_ssl_continued_pretraining.yaml \
    --output-dir ./output_tissue_adapted \
    --gpus 3 \
    --max-epochs 50

# 2. Fine-tune for classification
python scripts/classification/train.py \
    --config configs/tissue/config_classification.yaml \
    --pretrained_path ./output_tissue_adapted/checkpoints/pretraining/last.ckpt \
    --max_epochs 30 \
    --devices 0,1

# 3. Evaluate
python scripts/classification/eval.py \
    --config configs/tissue/config_classification.yaml \
    --checkpoint_path ./output_classification/checkpoints/best.ckpt
```

This comprehensive examples guide covers most use cases for training DINOv3 models with PyTorch Lightning.
