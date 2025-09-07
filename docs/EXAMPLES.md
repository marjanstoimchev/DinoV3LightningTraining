# Usage Examples

This document provides comprehensive examples for training DINOv3 models with PyTorch Lightning.

## Basic Examples

### 1. Single GPU Training (Testing)

Fastest way to test the framework:

```bash
python src/training/train_dinov3_lightning.py \
    --config-file configs/config_lightning_finetuning.yaml \
    --checkpoint-path dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --output-dir ./output_test \
    --gpus 1 \
    --max-epochs 5 \
    --sampler-type infinite \
    --limit-train-batches 0.1
```

### 2. Multi-GPU Training (Production)

Full-scale training on 4 GPUs:

```bash
python src/training/train_dinov3_lightning.py \
    --config-file configs/config_lightning_finetuning.yaml \
    --checkpoint-path dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --output-dir ./output_multigpu \
    --gpus 4 \
    --strategy ddp \
    --sampler-type distributed \
    --batch-size 128 \
    --max-epochs 100 \
    --precision bf16-mixed
```

### 3. Resume from Checkpoint

Continue training from a saved checkpoint:

```bash
python src/training/train_dinov3_lightning.py \
    --config-file configs/config_lightning_finetuning.yaml \
    --checkpoint-path dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --resume-from-checkpoint ./output/checkpoints/last.ckpt \
    --output-dir ./output_resumed \
    --gpus 4 \
    --strategy ddp
```

## Dataset-Specific Examples

### HuggingFace Datasets

**Food-101 Dataset:**
```bash
# First, update config file:
# dataset_path: HuggingFace:name=food101:split=train

python src/training/train_dinov3_lightning.py \
    --config-file configs/config_lightning_finetuning.yaml \
    --checkpoint-path dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --output-dir ./output_food101 \
    --gpus 2 \
    --strategy ddp \
    --sampler-type distributed \
    --batch-size 64
```

**Oxford Pets Dataset:**
```bash
# Update config: dataset_path: HuggingFace:name=jonathancui/oxford-pets

python src/training/train_dinov3_lightning.py \
    --config-file configs/config_lightning_finetuning.yaml \
    --checkpoint-path dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --output-dir ./output_pets \
    --gpus 1 \
    --sampler-type infinite \
    --max-epochs 50
```

**Custom HuggingFace Dataset:**
```bash
# For dataset with custom column names:
# dataset_path: HuggingFace:name=your/dataset:image_key=img:label_key=target

python src/training/train_dinov3_lightning.py \
    --config-file configs/config_lightning_finetuning.yaml \
    --checkpoint-path dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --output-dir ./output_custom \
    --gpus 4 \
    --sampler-type distributed \
    --batch-size 128
```

### Custom TIFF Datasets

**Medical Images:**
```bash
# Update config: dataset_path: CustomTIFF:root=/data/medical_images/

python src/training/train_dinov3_lightning.py \
    --config-file configs/config_lightning_finetuning.yaml \
    --checkpoint-path dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --output-dir ./output_medical \
    --gpus 4 \
    --strategy ddp \
    --sampler-type epoch \
    --batch-size 96 \
    --max-epochs 200
```

**Satellite Images:**
```bash
# For high-resolution satellite images
# Update config to use larger crop sizes:
# global_crops_size: 512
# local_crops_size: 256

python src/training/train_dinov3_lightning.py \
    --config-file configs/config_lightning_satellite.yaml \
    --checkpoint-path dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --output-dir ./output_satellite \
    --gpus 8 \
    --strategy ddp \
    --sampler-type distributed \
    --batch-size 64 \
    --precision bf16-mixed
```

## Architecture-Specific Examples

### ViT-Small (Default)
```bash
# Fastest training, good for experimentation
python src/training/train_dinov3_lightning.py \
    --config-file configs/config_lightning_finetuning.yaml \
    --checkpoint-path dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --output-dir ./output_vit_small \
    --gpus 4 \
    --batch-size 128 \
    --sampler-type distributed
```

### ViT-Base (Higher Capacity)
```bash
# Update config: student.arch = vit_base
# Use corresponding checkpoint: dinov3_vitb16_pretrain_lvd1689m-*.pth

python src/training/train_dinov3_lightning.py \
    --config-file configs/config_lightning_vitbase.yaml \
    --checkpoint-path dinov3_official_weights/dinov3_vitb16_pretrain_lvd1689m-08c60483.pth \
    --output-dir ./output_vit_base \
    --gpus 4 \
    --batch-size 64 \  # Reduced batch size for larger model
    --sampler-type distributed
```

### ViT-Large (Maximum Performance)
```bash
# Update config: student.arch = vit_large  
# Use corresponding checkpoint: dinov3_vitl16_pretrain_lvd1689m-*.pth

python src/training/train_dinov3_lightning.py \
    --config-file configs/config_lightning_vitlarge.yaml \
    --checkpoint-path dinov3_official_weights/dinov3_vitl16_pretrain_lvd1689m-08c60483.pth \
    --output-dir ./output_vit_large \
    --gpus 8 \
    --batch-size 32 \  # Even smaller batch size
    --accumulate-grad-batches 2 \
    --sampler-type distributed
```

## Sampler Comparison Examples

### Infinite Sampler (SSL Training)
```bash
# Best for continuous streaming, self-supervised learning
python src/training/train_dinov3_lightning.py \
    --config-file configs/config_lightning_finetuning.yaml \
    --checkpoint-path dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --output-dir ./output_infinite \
    --gpus 4 \
    --sampler-type infinite \
    --batch-size 128
    
# Progress shown as: Step 384/600 (64.0%)
# Uses OFFICIAL_EPOCH_LENGTH from config
```

### Distributed Sampler (Multi-GPU Efficiency)
```bash
# Best for multi-GPU training efficiency
python src/training/train_dinov3_lightning.py \
    --config-file configs/config_lightning_finetuning.yaml \
    --checkpoint-path dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --output-dir ./output_distributed \
    --gpus 4 \
    --strategy ddp \
    --sampler-type distributed \
    --batch-size 128
    
# Progress shown as: Step 207/207 (100.0%) per GPU
# Dataset automatically split across GPUs
```

### Epoch Sampler (Traditional Training)
```bash
# Traditional epoch-based training
python src/training/train_dinov3_lightning.py \
    --config-file configs/config_lightning_finetuning.yaml \
    --checkpoint-path dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --output-dir ./output_epoch \
    --gpus 4 \
    --sampler-type epoch \
    --batch-size 128
    
# Progress shown as: Step 832/832 (100.0%) per GPU  
# Each GPU sees the full dataset
```

## Performance Optimization Examples

### Memory-Optimized Training
```bash
# For limited GPU memory
python src/training/train_dinov3_lightning.py \
    --config-file configs/config_lightning_finetuning.yaml \
    --checkpoint-path dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --output-dir ./output_memory_opt \
    --gpus 4 \
    --batch-size 32 \  # Smaller total batch
    --accumulate-grad-batches 4 \  # Simulate larger batch
    --precision bf16-mixed \  # Mixed precision
    --sampler-type distributed
```

### Speed-Optimized Training
```bash
# For maximum speed
python src/training/train_dinov3_lightning.py \
    --config-file configs/config_lightning_finetuning.yaml \
    --checkpoint-path dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --output-dir ./output_speed_opt \
    --gpus 4 \
    --batch-size 128 \
    --compile \  # PyTorch 2.0 compilation
    --sampler-type distributed \
    --precision bf16-mixed \
    --num-nodes 1
```

### Large-Scale Training
```bash
# Multi-node, high-throughput training
python src/training/train_dinov3_lightning.py \
    --config-file configs/config_lightning_largescale.yaml \
    --checkpoint-path dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --output-dir ./output_largescale \
    --gpus 8 \
    --num-nodes 4 \  # 32 GPUs total
    --batch-size 512 \  # Large batch size
    --strategy ddp \
    --sampler-type distributed \
    --precision bf16-mixed \
    --save-every-n-steps 1000
```

## Advanced Configuration Examples

### Custom Learning Rate Schedule
```yaml
# In config file:
optim:
  lr: 0.0005              # Higher initial LR
  warmup_epochs: 10       # Longer warmup
  weight_decay: 0.01      # Lower weight decay
  min_lr: 1.0e-06        # Lower minimum LR
  
# Run with:
python src/training/train_dinov3_lightning.py \
    --config-file configs/config_custom_lr.yaml \
    --max-epochs 200 \
    --gpus 4
```

### Data Augmentation Tuning
```yaml
# In config file - for high-resolution images:
crops:
  global_crops_size: 384       # Larger global crops
  local_crops_size: 196        # Larger local crops  
  global_crops_scale: [0.6, 1.0]  # Less aggressive scaling
  local_crops_number: 8        # More local crops
  
# Run with:
python src/training/train_dinov3_lightning.py \
    --config-file configs/config_hires_augment.yaml \
    --gpus 4 \
    --batch-size 64  # Reduced for larger images
```

### Loss Weight Tuning
```yaml
# In config file - for noisy datasets:
dino:
  loss_weight: 0.8           # Reduce DINO loss
  koleo_loss_weight: 0.1     # Increase regularization
  
ibot:
  loss_weight: 1.2           # Increase iBOT loss
  mask_sample_probability: 0.4  # More aggressive masking
```

## Debugging and Development

### Fast Development Iteration
```bash
# Quick training for code testing
python src/training/train_dinov3_lightning.py \
    --config-file configs/config_lightning_finetuning.yaml \
    --checkpoint-path dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --output-dir ./output_debug \
    --gpus 1 \
    --fast-dev-run \  # Only 2 batches
    --sampler-type infinite
```

### Limited Training for Testing
```bash
# Train on subset of data
python src/training/train_dinov3_lightning.py \
    --config-file configs/config_lightning_finetuning.yaml \
    --checkpoint-path dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --output-dir ./output_subset \
    --gpus 1 \
    --limit-train-batches 0.01 \  # Only 1% of data
    --max-epochs 3 \
    --sampler-type epoch
```

### Profiling Training
```bash
# With detailed profiling
python src/training/train_dinov3_lightning.py \
    --config-file configs/config_lightning_finetuning.yaml \
    --checkpoint-path dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --output-dir ./output_profile \
    --gpus 1 \
    --max-epochs 1 \
    --limit-train-batches 100 \  # Fixed number of batches
    --log-every-n-steps 1  # Detailed logging
```

## Monitoring Examples

### TensorBoard Monitoring
```bash
# Start training
python src/training/train_dinov3_lightning.py ... --output-dir ./output

# In another terminal, start TensorBoard  
tensorboard --logdir ./output/tensorboard_logs --port 6006

# Open browser: http://localhost:6006
```

### WandB Integration
```yaml
# In config file:
logging:
  wandb:
    project: "dinov3-training"
    entity: "your-username" 
    name: "vit-small-food101"

# Run training normally, metrics will appear in WandB
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
plt.plot(logs['step'], logs['dino_local_loss'], label='DINO')
plt.plot(logs['step'], logs['ibot_loss'], label='iBOT')
plt.legend()
plt.title('Component Losses')
plt.show()
```

## Common Workflow Examples

### Experiment Pipeline
```bash
# 1. Quick test on small dataset
python src/training/train_dinov3_lightning.py \
    --config-file configs/config_lightning_finetuning.yaml \
    --checkpoint-path dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --output-dir ./output_test \
    --gpus 1 --max-epochs 5 --limit-train-batches 0.1

# 2. Medium-scale validation  
python src/training/train_dinov3_lightning.py \
    --config-file configs/config_lightning_finetuning.yaml \
    --checkpoint-path dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --output-dir ./output_validation \
    --gpus 2 --max-epochs 20 --batch-size 64

# 3. Full production training
python src/training/train_dinov3_lightning.py \
    --config-file configs/config_lightning_finetuning.yaml \
    --checkpoint-path dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --output-dir ./output_production \
    --gpus 4 --max-epochs 100 --batch-size 128 --strategy ddp
```

### Hyperparameter Search
```bash
# Grid search over learning rates
for lr in 0.0001 0.0005 0.001; do
    python src/training/train_dinov3_lightning.py \
        --config-file configs/config_lightning_finetuning.yaml \
        --checkpoint-path dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
        --output-dir ./output_lr_${lr} \
        --gpus 2 \
        --max-epochs 20 \
        --batch-size 64
    # Update lr in config or use config override
done
```

This comprehensive examples guide should cover most use cases for training DINOv3 models with PyTorch Lightning.