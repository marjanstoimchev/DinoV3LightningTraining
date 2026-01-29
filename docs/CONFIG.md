# Configuration Reference

This document provides detailed information about all configuration options available in DINOv3 Lightning.

## Configuration File Structure

Configuration files are organized by dataset in the `configs/` directory:

```
configs/
├── DTD/
│   ├── config_ssl_pretraining.yaml
│   ├── config_ssl_continued_pretraining.yaml
│   └── config_classification.yaml
├── eurosat/
├── oxford_pets/
├── NCTCRCHE100K/
├── tissue/
└── imagenet1k/
```

Each dataset has three configuration files:
- **config_ssl_pretraining.yaml** - Training DINOv3 from scratch
- **config_ssl_continued_pretraining.yaml** - Continued pretraining from official DINOv3 weights
- **config_classification.yaml** - Downstream classification/fine-tuning

## Core Configuration Sections

### MODEL
```yaml
MODEL:
  META_ARCHITECTURE: SSLMetaArch  # Meta architecture type
  DEVICE: cuda                    # Device to use (cuda/cpu)
  WEIGHTS: ''                     # Path to model weights (if any)
  DTYPE: float32                  # Base data type
```

### Compute Precision
```yaml
compute_precision:
  param_dtype: bf16      # Parameter dtype (fp32/fp16/bf16)
  reduce_dtype: fp32     # Reduction operations dtype
  sharding_strategy: SHARD_GRAD_OP  # Sharding strategy for distributed training
```

### Training Configuration
```yaml
train:
  batch_size_per_gpu: 128                     # Batch size per GPU
  dataset_path: HuggingFace:name=blanchon/EuroSAT_RGB  # Dataset specification
  output_dir: .                               # Output directory
  saveckp_freq: 1000                          # Legacy checkpoint frequency
  seed: 0                                     # Random seed
  num_workers: 10                             # DataLoader workers
  OFFICIAL_EPOCH_LENGTH: 70                   # Steps per epoch for infinite samplers
  monitor_gradient_norm: false                # Enable gradient norm monitoring
  cache_dataset: false                        # Enable dataset caching
  use_teacher_head: true                      # Use teacher head in training
  compile: true                               # Enable PyTorch 2.0 compilation
  cudagraphs: false                           # Enable CUDA graphs
  centering: "sinkhorn_knopp"                 # Centering method
```

### GRAM Loss Configuration
```yaml
gram:
  use_loss: false                 # Enable GRAM loss training
  loss_weight: 1.0                # GRAM loss weight
  ema_teacher: false              # Use EMA teacher for GRAM
  update_frequency: 50000         # Update frequency for GRAM
  normalized: true                # Normalize GRAM loss
  tokens_used: all                # Which tokens to use
```

**GRAM Loss Details:**
- **use_loss**: Enables gradient-based regularization with auxiliary teacher model
- **loss_weight**: Weight for the GRAM loss term
- **update_frequency**: How often to update GRAM teacher

### Dataset Path Formats

**HuggingFace Datasets:**
```yaml
# Basic dataset
dataset_path: HuggingFace:name=blanchon/EuroSAT_RGB

# With specific split
dataset_path: HuggingFace:name=food101:split=train

# With custom keys
dataset_path: HuggingFace:name=your-dataset:image_key=img:label_key=target

# With streaming
dataset_path: HuggingFace:name=large-dataset:streaming=true
```

**Custom TIFF Datasets:**
```yaml
# Basic path
dataset_path: CustomTIFF:root=/path/to/images/

# The path should contain TIFF/PNG/JPG images
# Supports recursive directory scanning
```

### Student Model Configuration
```yaml
student:
  arch: vit_small              # Architecture (vit_small/vit_base/vit_large)
  patch_size: 16               # Vision Transformer patch size
  drop_path_rate: 0.3          # DropPath rate for regularization
  layerscale: 1.0e-05          # LayerScale initialization
  pretrained_weights: ''       # Path to pretrained weights (for continued pretraining)
  ffn_layer: mlp               # Feed-forward network type
  ffn_ratio: 4.0               # FFN hidden dimension ratio
  qkv_bias: true               # Use bias in QKV projections
  proj_bias: true              # Use bias in projections
  ffn_bias: true               # Use bias in FFN
  norm_layer: layernorm        # Normalization layer type
  n_storage_tokens: 4          # Number of register tokens
  pos_embed_type: rope         # Position embedding type (rope recommended)
```

### Teacher Model Configuration
```yaml
teacher:
  momentum_teacher: 0.994          # EMA momentum for teacher updates
  final_momentum_teacher: 0.994    # Final momentum value
  warmup_teacher_temp: 0.04        # Initial teacher temperature
  teacher_temp: 0.07               # Final teacher temperature
  warmup_teacher_temp_epochs: 30   # Temperature warmup epochs
```

### Optimization Configuration
```yaml
optim:
  epochs: 100                  # Total training epochs
  optimizer: adamw             # Optimizer type
  weight_decay: 0.04           # Weight decay
  weight_decay_end: 0.04       # Final weight decay (cosine schedule)
  lr: 0.001                    # Learning rate
  warmup_epochs: 10            # Warmup epochs
  min_lr: 0.001                # Minimum learning rate
  clip_grad: 3.0               # Gradient clipping threshold
  freeze_last_layer_epochs: 1  # Freeze last layer for N epochs
  scaling_rule: sqrt_wrt_1024  # LR scaling rule
  layerwise_decay: 0.9         # Layer-wise learning rate decay
  adamw_beta1: 0.9             # Adam beta1
  adamw_beta2: 0.999           # Adam beta2
```

### Data Augmentation (Crops)
```yaml
crops:
  global_crops_scale: [0.32, 1.0]  # Global crop scale range
  local_crops_number: 8             # Number of local crops
  local_crops_scale: [0.05, 0.32]  # Local crop scale range
  global_crops_size: 256            # Global crop size
  local_crops_size: 112             # Local crop size
  horizontal_flips: true            # Enable horizontal flips
  # ImageNet normalization
  rgb_mean: [0.485, 0.456, 0.406]   # RGB mean values
  rgb_std: [0.229, 0.224, 0.225]    # RGB std values
```

### Loss Configuration

**DINO Loss:**
```yaml
dino:
  loss_weight: 1.0              # Loss weight
  head_n_prototypes: 65536      # Number of prototypes
  head_bottleneck_dim: 256      # Bottleneck dimension
  head_nlayers: 3               # Number of head layers
  head_hidden_dim: 2048         # Hidden dimension
  koleo_loss_weight: 0.1        # KoLeo regularization weight
```

**iBOT Loss:**
```yaml
ibot:
  loss_weight: 1.0                    # Loss weight
  mask_sample_probability: 0.5        # Probability of masking
  mask_ratio_min_max: [0.1, 0.5]      # Masking ratio range
  separate_head: true                  # Use separate head
  head_n_prototypes: 65536            # Number of prototypes
  head_bottleneck_dim: 256            # Bottleneck dimension
```

### Checkpointing
```yaml
checkpointing:
  period: 350          # Checkpoint period (steps), ~5 epochs
  max_to_keep: 3       # Maximum checkpoints to keep
```

## Command Line Arguments

All configuration options can be overridden via command line arguments:

### Basic Arguments
```bash
--config-file CONFIG_FILE        # Path to config file (required)
--checkpoint-path PATH           # Path to pretrained DINOv3 checkpoint
--output-dir DIR                 # Output directory
--seed SEED                      # Random seed (default: 42)
--resume-from-checkpoint PATH    # Resume from Lightning checkpoint
```

### Lightning-Specific Arguments
```bash
--gpus GPUS                      # Number of GPUs (default: 1)
--num-nodes NODES                # Number of nodes (default: 1)
--precision PRECISION            # Training precision (bf16-mixed/16-mixed/32)
--strategy STRATEGY              # Training strategy (auto/ddp/ddp_sharded)
--accumulate-grad-batches N      # Gradient accumulation batches
--max-epochs EPOCHS              # Override max epochs
```

### Data Loading Arguments
```bash
--sampler-type TYPE              # Sampler type (infinite/distributed/epoch/sharded_infinite)
--batch-size BATCH_SIZE          # Total batch size across all GPUs
--compile                        # Enable PyTorch 2.0 compilation
```

### Model Configuration Arguments
```bash
--num-prototypes N               # Override number of prototypes in DINO head
--koleo-weight WEIGHT            # Override KoLeo loss weight
--enable-gram                    # Enable GRAM loss
--gram-weight WEIGHT             # GRAM loss weight
```

### Logging Arguments
```bash
--log-every-n-steps N            # Log every N steps (default: 10)
--save-every-n-steps N           # Save checkpoint every N steps (default: 100)
--progress-log-every-n-steps N   # Progress log frequency (default: 10)
```

## Sampler Types Comparison

| Sampler Type | Best For | Multi-GPU | Performance | Epoch Length |
|-------------|----------|-----------|-------------|--------------|
| `infinite` | SSL training, continuous streaming | Yes | Fastest | Uses `OFFICIAL_EPOCH_LENGTH` |
| `distributed` | Standard multi-GPU training | Yes | Very Fast | Uses actual dataset size / GPUs |
| `epoch` | Traditional training | Yes | Fast | Uses actual dataset size |
| `sharded_infinite` | Memory-efficient streaming | Yes | Good | Uses `OFFICIAL_EPOCH_LENGTH` |

## Performance Tuning

### Batch Size Guidelines
- **ViT-Small**: Start with `batch_size_per_gpu=128`
- **ViT-Base**: Start with `batch_size_per_gpu=64`
- **ViT-Large**: Start with `batch_size_per_gpu=32`

Scale total batch size with number of GPUs:
```bash
# 3 GPUs, 128 per GPU = 384 total
--gpus 3 --batch-size 384
```

### Memory Optimization
```yaml
# In config file:
train:
  batch_size_per_gpu: 64       # Reduce if OOM
  num_workers: 8               # Adjust based on CPU cores
  compile: true                # Enable for PyTorch 2.0 speedup

compute_precision:
  param_dtype: bf16            # Use bf16 for memory efficiency
```

### Distributed Training Best Practices
```bash
# Use distributed sampler for multi-GPU
--sampler-type distributed --strategy ddp

# Enable sync batch norm for multi-GPU
# (automatically enabled in multi-GPU setups)
```

## Environment Variables

```bash
# CUDA settings
export CUDA_VISIBLE_DEVICES=0,1,2     # Specify GPU devices
export OMP_NUM_THREADS=8               # OpenMP threads

# NCCL settings for distributed training
export NCCL_DEBUG=INFO                 # NCCL debug level
export NCCL_IB_DISABLE=1              # Disable InfiniBand if needed
```

## Example Configurations

### Quick Testing (Single GPU)
```bash
python src/training/ssl/train.py \
    --config-file configs/eurosat/config_ssl_pretraining.yaml \
    --output-dir ./output_test \
    --gpus 1 \
    --max-epochs 5 \
    --limit-train-batches 0.1
```

### Production Training (Multi-GPU)
```bash
python src/training/ssl/train.py \
    --config-file configs/eurosat/config_ssl_pretraining.yaml \
    --output-dir ./output_eurosat \
    --gpus 3 \
    --strategy ddp \
    --precision bf16-mixed \
    --sampler-type distributed
```

### Continued Pretraining from DINOv3
```bash
python src/training/ssl/train.py \
    --config-file configs/eurosat/config_ssl_continued_pretraining.yaml \
    --output-dir ./output_eurosat_continued \
    --gpus 3 \
    --precision bf16-mixed
```

### Custom Dataset Training
```yaml
# In config file for high-resolution images:
train:
  batch_size_per_gpu: 64
  dataset_path: CustomTIFF:root=/data/my_images/

crops:
  global_crops_size: 384  # Larger crops for high-res images
  local_crops_size: 196
```

## Dataset-Specific Configurations

### EuroSAT (Satellite Imagery)
- ~27K images, 10 classes
- OFFICIAL_EPOCH_LENGTH: 70 (with batch_size 128 on 3 GPUs)
- Global crops: 256px, Local crops: 112px

### DTD (Describable Textures)
- ~5.6K images, 47 classes
- OFFICIAL_EPOCH_LENGTH: 15 (with batch_size 128 on 3 GPUs)
- Good for texture representation learning

### Oxford Pets
- ~7.3K images, 37 classes
- OFFICIAL_EPOCH_LENGTH: 20 (with batch_size 128 on 3 GPUs)
- Fine-grained animal classification

### NCTCRCHE100K (Histopathology)
- ~100K images, 9 classes
- OFFICIAL_EPOCH_LENGTH: 260 (with batch_size 128 on 3 GPUs)
- Medical imaging domain

### ImageNet-1K
- ~1.28M images, 1000 classes
- OFFICIAL_EPOCH_LENGTH: 3333 (with batch_size 128 on 3 GPUs)
- Large-scale pretraining
