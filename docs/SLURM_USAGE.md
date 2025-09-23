# SLURM Usage Guide for DINOv3 Lightning Fine-tuning

This guide explains how to run the DINOv3 fine-tuning on SLURM clusters using the provided `slurm_train.sh` script.

## Quick Start

```bash
# Submit the job
sbatch slurm_train.sh

# Check job status
squeue -u $USER

# View logs (replace JOBID with actual job ID)
tail -f logs/slurm-JOBID.out
```

## SLURM Script Configuration

### Core Parameters

```bash
#SBATCH --job-name=dinov3-finetune    # Job name in queue
#SBATCH --nodes=1                     # Number of compute nodes
#SBATCH --ntasks-per-node=1          # MPI tasks per node (keep at 1)
#SBATCH --cpus-per-task=32           # CPU cores (8 × num_gpus)
#SBATCH --gres=gpu:4                 # Number of GPUs
#SBATCH --time=24:00:00              # Max runtime (HH:MM:SS)
#SBATCH --mem=64G                    # Memory per node
#SBATCH --partition=gpu              # Queue/partition name
```

### Parameter Matching Requirements

| SLURM Parameter | Training Script Variable | Purpose | Notes |
|----------------|-------------------------|---------|--------|
| `--gres=gpu:4` | `GPUS=4` | GPU allocation | Must match exactly |
| `--nodes=1` | `--num-nodes 1` | Node allocation vs Lightning config | **Different but must match** |
| `--cpus-per-task=32` | `OMP_NUM_THREADS` | CPU allocation | ~8 CPUs per GPU |
| `--ntasks-per-node=1` | N/A | MPI task count | Keep at 1 for PyTorch Lightning |

#### Critical Node Parameter Difference

**`#SBATCH --nodes=1`** vs **`--num-nodes 1`** are **different parameters** that must match:

- **`#SBATCH --nodes=1`**: SLURM directive that **requests** 1 compute node from the cluster
- **`--num-nodes 1`**: PyTorch Lightning argument that **configures** distributed training for 1 node

**How they work together**:
1. SLURM allocates the physical compute nodes based on `--nodes`
2. Lightning sets up distributed training coordination based on `--num-nodes` 
3. If they don't match, training will fail or hang waiting for nodes

**Example for multi-node training**:
```bash
#SBATCH --nodes=2        # Request 2 nodes from SLURM
--num-nodes 2            # Tell Lightning to expect 2 nodes
```

**In the training code** (`train_dinov3_lightning.py:285,296`):
- Lightning uses `num_nodes` in the `Trainer` configuration
- Enables batch normalization sync when `num_nodes > 1` or `gpus > 1`

## Customizing the Training Script

The `slurm_train.sh` script includes many configurable parameters. Edit these variables in the script to customize your training:

### Core Training Parameters

```bash
# Model and data configuration
CONFIG_FILE="configs/config_lightning_finetuning.yaml"
CHECKPOINT_PATH="dinov3/dinov3/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
OUTPUT_DIR="./output_slurm_${SLURM_JOB_ID}"
SEED=42

# GPU and distributed training
GPUS=4                          # Number of GPUs to use
STRATEGY="ddp"                  # Training strategy (auto, ddp, ddp_sharded, etc.)
PRECISION="bf16-mixed"          # Training precision (32, 16, bf16-mixed, 16-mixed)

# Training duration and batching
MAX_EPOCHS=100                  # Maximum training epochs
BATCH_SIZE=32                   # Total batch size across all GPUs
ACCUMULATE_GRAD_BATCHES=1       # Gradient accumulation steps
LIMIT_TRAIN_BATCHES=1.0         # Fraction of dataset to use (1.0 = full dataset)

# Data loading and sampling
SAMPLER_TYPE="infinite"         # Options: infinite, distributed, sharded_infinite, epoch

# Checkpointing and logging
SAVE_EVERY_N_STEPS=5000         # Save checkpoint every N steps
PROGRESS_LOG_STEPS=50           # Log progress every N steps

# Advanced options
RESUME_FROM_CHECKPOINT=""       # Path to Lightning checkpoint to resume from
COMPILE=false                   # Enable PyTorch 2.0 compilation
```

### Data Sampler Types

Choose the appropriate sampler for your training:

| Sampler Type | Use Case | Dataset Behavior | Progress Bar |
|--------------|----------|------------------|--------------|
| `infinite` | Standard DINOv3 training | Infinite sampling, no epochs | Enhanced with step estimates |
| `distributed` | Multi-GPU with finite epochs | Distributed sampling across GPUs | Standard with epoch progress |
| `sharded_infinite` | Large-scale infinite sampling | Sharded infinite sampling | Enhanced with step estimates |
| `epoch` | Traditional epoch-based training | Standard epoch boundaries | Standard with epoch progress |

### Performance Optimization Examples

#### For Different GPU Configurations

```bash
# Single GPU setup
GPUS=1
STRATEGY="auto"
BATCH_SIZE=8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# Dual GPU setup
GPUS=2
STRATEGY="ddp"
BATCH_SIZE=16
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16

# High-end 8 GPU setup
GPUS=8
STRATEGY="ddp"
BATCH_SIZE=64
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
```

#### For Different Training Scenarios

```bash
# Quick testing/debugging
MAX_EPOCHS=1
LIMIT_TRAIN_BATCHES=0.1         # Use only 10% of dataset
SAVE_EVERY_N_STEPS=100
PROGRESS_LOG_STEPS=10

# Long production training
MAX_EPOCHS=200
LIMIT_TRAIN_BATCHES=1.0
SAVE_EVERY_N_STEPS=5000
PROGRESS_LOG_STEPS=50

# Memory-constrained training
BATCH_SIZE=16
ACCUMULATE_GRAD_BATCHES=2       # Effective batch size = 16 * 2 = 32
PRECISION="16-mixed"            # Use 16-bit precision
```

#### Compilation and Performance

```bash
# Enable PyTorch 2.0 compilation for faster training
COMPILE=true

# Note: Compilation adds startup time but can significantly speed up training
# Best for longer training runs (>1000 steps)
```

## Customizing for Your Cluster

### 1. Check Available Resources

```bash
# View partitions and limits
sinfo

# Check GPU types available
sinfo -o "%P %G %N" | grep gpu

# View your current jobs
squeue -u $USER
```

### 2. Adjust SLURM Resource Requests

Edit the SLURM directives in `slurm_train.sh` based on your cluster:

```bash
# For different GPU counts (must match GPUS variable)
#SBATCH --gres=gpu:2        # For 2 GPUs
#SBATCH --cpus-per-task=16  # Adjust CPUs accordingly (8 per GPU)
# Then change: GPUS=2 in script

# For different partitions
#SBATCH --partition=v100    # If your cluster uses GPU-specific partitions

# For longer/shorter jobs
#SBATCH --time=48:00:00     # 48 hours
#SBATCH --time=4:00:00      # 4 hours for testing
```

### 3. Memory Requirements

```bash
# Adjust based on your model and batch size
#SBATCH --mem=32G          # For smaller models/batch sizes
#SBATCH --mem=128G         # For larger models/batch sizes

# Rule of thumb: 16GB per GPU minimum
# For batch_size=32 with 4 GPUs: 64-128GB recommended
```

## Monitoring and Debugging

### Job Status Commands

```bash
# Check job queue status
squeue -u $USER

# Detailed job info
scontrol show job JOBID

# Cancel a job
scancel JOBID

# View job history
sacct -u $USER --format=JobID,JobName,State,ExitCode,Start,End
```

### Log Files

The script creates logs in the `logs/` directory:

- `logs/slurm-JOBID.out` - Standard output and training logs
- `logs/slurm-JOBID.err` - Error messages and warnings

```bash
# Monitor training progress
tail -f logs/slurm-JOBID.out

# Check for errors
tail -f logs/slurm-JOBID.err

# Search for specific metrics
grep "val_loss" logs/slurm-JOBID.out
```

## Common Issues and Solutions

### 1. Job Won't Start

**Problem**: Job stays in pending state
```bash
# Check why job is pending
squeue -u $USER -l
```

**Solutions**:
- Reduce resource requests (GPUs, memory, time)
- Check if partition name is correct
- Verify you have access to requested resources

### 2. Out of Memory Errors

**Problem**: `CUDA out of memory` or similar errors

**Solutions**:
```bash
# Reduce batch size in script
BATCH_SIZE=16  # Instead of 32

# Use gradient accumulation to maintain effective batch size
BATCH_SIZE=16
ACCUMULATE_GRAD_BATCHES=2  # Effective batch size = 16 * 2 = 32

# Use lower precision
PRECISION="16-mixed"  # Instead of bf16-mixed

# Reduce training data for testing
LIMIT_TRAIN_BATCHES=0.5  # Use only 50% of dataset
```

### 3. GPU Allocation Mismatch

**Problem**: Training uses wrong number of GPUs

**Solution**: Ensure these match:
```bash
#SBATCH --gres=gpu:4  # SLURM request
GPUS=4                # Training script
```

### 4. Compilation Issues

**Problem**: PyTorch compilation fails or slows down training

**Solutions**:
```bash
# Disable compilation for debugging
COMPILE=false

# Or use compilation only for longer runs
# (compilation overhead is amortized over many steps)
```

### 5. Sampler-Related Issues

**Problem**: Training hangs or progress bar behaves unexpectedly

**Solutions**:
```bash
# Try different sampler types:
SAMPLER_TYPE="epoch"        # For traditional epoch-based training
SAMPLER_TYPE="distributed"  # For distributed finite-length training
SAMPLER_TYPE="infinite"     # For DINOv3-style infinite sampling (default)
```

### 6. Node Communication Issues

**Problem**: Multi-node training fails (if using multiple nodes)

**Solutions**:
- Check network connectivity between nodes
- Verify NCCL configuration
- Use InfiniBand if available

## Performance Optimization

### 1. Optimal Resource Allocation

```bash
# Rule of thumb for CPU allocation
CPUs = GPUs × 8  # 8 CPUs per GPU

# Memory allocation
Memory = GPUs × 16GB  # 16GB per GPU minimum
```

### 2. Batch Size Tuning

Start with smaller batch sizes and increase:

```bash
# Conservative starting point
BATCH_SIZE=8   # 2 per GPU for 4 GPUs

# Increase if memory allows
BATCH_SIZE=16  # 4 per GPU
BATCH_SIZE=32  # 8 per GPU
```

### 3. Checkpoint Strategy

```bash
# Frequent checkpointing for long jobs
SAVE_EVERY_N_STEPS=1000  # More frequent saves

# Less frequent for stable training
SAVE_EVERY_N_STEPS=5000  # Default
```

## Advanced Usage

### Interactive Development

For debugging, you can request an interactive session:

```bash
# Request interactive GPU node
salloc --gres=gpu:1 --cpus-per-task=8 --mem=16G --time=2:00:00

# Then run training interactively
python train_dinov3_lightning.py [args...]
```

### Array Jobs

For hyperparameter sweeps:

```bash
#SBATCH --array=1-10  # Run 10 different configurations

# Use $SLURM_ARRAY_TASK_ID in script to vary parameters
SEED=$((42 + SLURM_ARRAY_TASK_ID))
```

### Multi-Node Training

For very large models, edit the script to use multiple nodes:

```bash
#SBATCH --nodes=2          # Request 2 compute nodes
#SBATCH --ntasks-per-node=1 # Keep at 1 per node for Lightning

# Update training command - MUST match SLURM nodes
--num-nodes 2              # Tell Lightning to expect 2 nodes
```

**Important multi-node considerations**:
- Both `#SBATCH --nodes` and `--num-nodes` must be identical
- Lightning automatically handles cross-node communication
- Network latency between nodes affects training speed
- Use InfiniBand networks for optimal performance

**Multi-node batch size calculation**:
```bash
# With 2 nodes, 4 GPUs each = 8 total GPUs
BATCH_SIZE=64  # 8 samples per GPU across 8 GPUs
```

## Example Workflows

### 1. Quick Test Run

```bash
# Edit script for short test
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:1
MAX_EPOCHS=1

sbatch slurm_train.sh
```

### 2. Production Training

```bash
# Use default settings
sbatch slurm_train.sh

# Monitor progress
watch -n 10 'squeue -u $USER'
```

### 3. Resume from Checkpoint

The script now supports resuming from Lightning checkpoints. Edit the script to resume:

```bash
# Resume from a specific Lightning checkpoint
RESUME_FROM_CHECKPOINT="./output_slurm_12345/checkpoints/last.ckpt"

# Or resume from a step-specific checkpoint
RESUME_FROM_CHECKPOINT="./output_slurm_12345/checkpoints/model_epoch_05_step_010000_loss_0.234567.ckpt"
```

**Important**: This is different from the initial pretrained checkpoint:
- `CHECKPOINT_PATH`: DINOv3 pretrained model (.pth file) - used for initialization
- `RESUME_FROM_CHECKPOINT`: Lightning training checkpoint (.ckpt file) - used to resume training

### 4. Fast Development and Testing

```bash
# Quick test with minimal resources
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:1
MAX_EPOCHS=1
LIMIT_TRAIN_BATCHES=0.1
BATCH_SIZE=8
COMPILE=false
```

## Getting Help

- **Cluster Documentation**: Check your cluster's specific SLURM documentation
- **SLURM Manual**: `man sbatch`, `man squeue`, `man scancel`
- **PyTorch Lightning**: [Lightning Multi-GPU Guide](https://lightning.ai/docs/pytorch/stable/accelerators/gpu.html)