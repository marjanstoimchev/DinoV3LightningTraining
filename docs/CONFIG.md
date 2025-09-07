# Configuration Reference

This document provides detailed information about all configuration options available in DINOv3 Lightning.

## Configuration File Structure

The main configuration file is `configs/config_lightning_finetuning.yaml`. It follows the original DINOv3 configuration structure with additional Lightning-specific options.

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
  batch_size_per_gpu: 8                    # Batch size per GPU
  dataset_path: HuggingFace:name=food101   # Dataset specification
  output_dir: ./lightning_output           # Output directory
  saveckp_freq: 20                         # Legacy checkpoint frequency
  seed: 42                                 # Random seed
  num_workers: 8                           # DataLoader workers
  OFFICIAL_EPOCH_LENGTH: 600               # Steps per epoch for infinite samplers
  monitor_gradient_norm: false             # Enable gradient norm monitoring
  cache_dataset: false                     # Enable dataset caching
  use_teacher_head: true                   # Use teacher head in training
  compile: false                           # Enable PyTorch 2.0 compilation
  cudagraphs: false                        # Enable CUDA graphs
```

### GRAM Loss Configuration
```yaml
gram:
  use_loss: true                          # Enable GRAM loss training
  teacher_momentum: 0.999                 # Teacher EMA momentum (default: 0.999)
  warmup_teacher_temp: 0.04              # Initial teacher temperature (default: 0.04)
  teacher_temp: 0.05                      # Final teacher temperature (default: 0.05)
  warmup_teacher_temp_epochs: 30         # Epochs for teacher temperature warmup
```

**GRAM Loss Details:**
- **use_loss**: Enables gradient-based regularization with auxiliary teacher model
- **teacher_momentum**: EMA momentum for teacher model updates (higher = slower updates)
- **warmup_teacher_temp**: Starting temperature for teacher softmax (lower = sharper)
- **teacher_temp**: Final temperature after warmup period
- **warmup_teacher_temp_epochs**: Number of epochs for temperature scheduling

#### Dataset Path Formats

**HuggingFace Datasets:**
```yaml
# Basic dataset
dataset_path: HuggingFace:name=food101

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
  drop_path_rate: 0.1          # DropPath rate for regularization
  layerscale: 1.0e-05          # LayerScale initialization
  pretrained_weights: 'path'   # Path to pretrained weights
  ffn_layer: mlp               # Feed-forward network type
  ffn_ratio: 4.0               # FFN hidden dimension ratio
  qkv_bias: true               # Use bias in QKV projections
  proj_bias: true              # Use bias in projections
  ffn_bias: true               # Use bias in FFN
  norm_layer: layernorm        # Normalization layer type
  pos_embed_type: rope         # Position embedding type
```

### Teacher Model Configuration
```yaml
teacher:
  momentum_teacher: 0.999          # EMA momentum for teacher updates
  final_momentum_teacher: 1.0      # Final momentum value
  warmup_teacher_temp: 0.04        # Initial teacher temperature
  teacher_temp: 0.04               # Final teacher temperature  
  warmup_teacher_temp_epochs: 5    # Temperature warmup epochs
```

### Optimization Configuration
```yaml
optim:
  epochs: 100                  # Total training epochs
  optimizer: adamw             # Optimizer type
  weight_decay: 0.02           # Weight decay
  weight_decay_end: 0.1        # Final weight decay (cosine schedule)
  lr: 0.0001                   # Learning rate
  warmup_epochs: 2             # Warmup epochs
  min_lr: 1.0e-07              # Minimum learning rate
  clip_grad: 3.0               # Gradient clipping threshold
  freeze_last_layer_epochs: 3  # Freeze last layer for N epochs
  scaling_rule: sqrt_wrt_1024  # LR scaling rule
  layerwise_decay: 0.95        # Layer-wise learning rate decay
  adamw_beta1: 0.9             # Adam beta1
  adamw_beta2: 0.999           # Adam beta2
```

### Data Augmentation (Crops)
```yaml
crops:
  global_crops_scale: [0.5, 1.0]    # Global crop scale range
  local_crops_number: 6              # Number of local crops
  local_crops_scale: [0.1, 0.5]     # Local crop scale range
  global_crops_size: 224             # Global crop size
  local_crops_size: 96               # Local crop size
  horizontal_flips: true             # Enable horizontal flips
  # ImageNet normalization
  rgb_mean: [0.485, 0.456, 0.406]    # RGB mean values
  rgb_std: [0.229, 0.224, 0.225]     # RGB std values
```

### Loss Configuration

**DINO Loss:**
```yaml
dino:
  loss_weight: 1.0              # Loss weight
  head_n_prototypes: 8192       # Number of prototypes
  head_bottleneck_dim: 256      # Bottleneck dimension
  head_nlayers: 3               # Number of head layers
  head_hidden_dim: 2048         # Hidden dimension
  koleo_loss_weight: 0.05       # KoLeo regularization weight
```

**iBOT Loss:**
```yaml
ibot:
  loss_weight: 1.0                    # Loss weight
  mask_sample_probability: 0.3         # Probability of masking
  mask_ratio_min_max: [0.1, 0.4]     # Masking ratio range
  separate_head: true                  # Use separate head
  head_n_prototypes: 8192             # Number of prototypes
  head_bottleneck_dim: 256            # Bottleneck dimension
```

### Checkpointing
```yaml
checkpointing:
  period: 1000          # Checkpoint period (steps)
  max_to_keep: 3        # Maximum checkpoints to keep
  max_eval_to_keep: 5   # Maximum evaluation checkpoints
```

## Command Line Arguments

All configuration options can be overridden via command line arguments:

### Basic Arguments
```bash
--config-file CONFIG_FILE        # Path to config file (required)
--checkpoint-path PATH           # Path to pretrained checkpoint
--output-dir DIR                 # Output directory
--seed SEED                      # Random seed (default: 42)
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

### Logging Arguments
```bash
--log-every-n-steps N            # Log every N steps (default: 10)
--save-every-n-steps N           # Save checkpoint every N steps (default: 100)
--progress-log-every-n-steps N   # Progress log frequency (default: 10)
```

## Sampler Types Comparison

| Sampler Type | Best For | Multi-GPU | Performance | Epoch Length |
|-------------|----------|-----------|-------------|--------------|
| `infinite` | SSL training, continuous streaming | ✓ | Fastest | Uses `OFFICIAL_EPOCH_LENGTH` |
| `distributed` | Standard multi-GPU training | ✓ | Very Fast | Uses actual dataset size ÷ GPUs |
| `epoch` | Traditional training | ✓ | Fast | Uses actual dataset size |
| `sharded_infinite` | Memory-efficient streaming | ✓ | Good | Uses `OFFICIAL_EPOCH_LENGTH` |

## Performance Tuning

### Batch Size Guidelines
- **ViT-Small**: Start with `batch_size_per_gpu=8`
- **ViT-Base**: Start with `batch_size_per_gpu=4`  
- **ViT-Large**: Start with `batch_size_per_gpu=2`

Scale total batch size with number of GPUs:
```bash
# 4 GPUs, 8 per GPU = 32 total
--gpus 4 --batch-size 32
```

### Memory Optimization
```yaml
# In config file:
train:
  batch_size_per_gpu: 4      # Reduce if OOM
  num_workers: 8             # Adjust based on CPU cores
  compile: true              # Enable for PyTorch 2.0 speedup
  
compute_precision:
  param_dtype: bf16          # Use bf16 for memory efficiency
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
export CUDA_VISIBLE_DEVICES=0,1,2,3    # Specify GPU devices
export OMP_NUM_THREADS=8                # OpenMP threads

# NCCL settings for distributed training
export NCCL_DEBUG=INFO                  # NCCL debug level
export NCCL_IB_DISABLE=1               # Disable InfiniBand if needed
```

## Example Configurations

### Quick Testing (Single GPU)
```yaml
train:
  batch_size_per_gpu: 2
  OFFICIAL_EPOCH_LENGTH: 100
  dataset_path: HuggingFace:name=jonathancui/oxford-pets

optim:
  epochs: 5
  lr: 0.0005
```

### Production Training (Multi-GPU)
```yaml
train:
  batch_size_per_gpu: 8
  OFFICIAL_EPOCH_LENGTH: 1000
  dataset_path: HuggingFace:name=imagenet-1k
  num_workers: 16

optim:
  epochs: 100
  lr: 0.0001
```

### Custom Dataset Training
```yaml
train:
  batch_size_per_gpu: 8
  dataset_path: CustomTIFF:root=/data/my_images/
  OFFICIAL_EPOCH_LENGTH: 500

crops:
  global_crops_size: 384  # Larger crops for high-res images
  local_crops_size: 196
```