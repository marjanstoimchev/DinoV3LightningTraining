# SLURM Usage Guide for DINOv3 Lightning Training

This guide explains how to run DINOv3 training on SLURM clusters using the provided scripts.

## Quick Start

### Prototype Analysis Sweep (Recommended)

```bash
# Basic sweep on single dataset
sbatch scripts/prototype_analysis/run_sweep_slurm.sh

# Multi-dataset sweep
DATASETS="dtd eurosat oxford_pets" sbatch scripts/prototype_analysis/run_sweep_slurm.sh

# Continued pretraining from DINOv3 weights
CONTINUED_PRETRAINING=1 DATASETS="dtd" sbatch scripts/prototype_analysis/run_sweep_slurm.sh
```

### Check Job Status

```bash
# Check job status
squeue -u $USER

# View logs (replace JOBID with actual job ID)
tail -f logs/slurm-JOBID.out
```

## SLURM Script Configuration

### Default SLURM Parameters

The default configuration in `scripts/prototype_analysis/run_sweep_slurm.sh`:

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

### Parameter Matching Requirements

| SLURM Parameter | Environment Variable | Purpose | Notes |
|----------------|---------------------|---------|-------|
| `--gres=gpu:3` | `GPUS=0,1,2` | GPU allocation | Number must match |
| `--nodes=1` | N/A | Node allocation | Single node for standard runs |
| `--cpus-per-task=16` | N/A | CPU allocation | ~5 CPUs per GPU recommended |
| `--ntasks-per-node=3` | N/A | Task count | Match GPU count |

## Environment Variables

### Core Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `DATASETS` | Space-separated dataset names | `dtd` |
| `GPUS` | Comma-separated GPU indices | `0,1,2` |
| `PROTOTYPES` | Space-separated prototype counts | `"128 256 512 1024"` |
| `SEEDS` | Classification seeds | `"0 1 42"` |
| `PRETRAIN_SEED` | Pretraining seed | `42` |
| `OUTPUT_DIR` | Output directory | `prototype_analysis_dinov3` |

### Training Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `PRETRAIN_EPOCHS` | Pretraining epochs | `100` |
| `CLASSIFY_EPOCHS` | Classification epochs | `30` |
| `CLASSIFY_LR` | Classification learning rate | `0.0001` |
| `CLASSIFY_MODE` | `finetune` or `lineareval` | `finetune` |
| `BATCH_SIZE` | Batch size per GPU | `128` |
| `KOLEO_WEIGHT` | KoLeo loss weight | `0.1` |
| `PRECISION` | Training precision | `bf16-mixed` |
| `COMPILE` | Enable PyTorch compilation | - |

### Continued Pretraining

| Variable | Description | Default |
|----------|-------------|---------|
| `CONTINUED_PRETRAINING` | Enable continued pretraining | - |
| `PRETRAIN_CHECKPOINT` | Checkpoint path | Auto-set when `CONTINUED_PRETRAINING` enabled |

### GRAM Loss

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_GRAM` | Enable GRAM loss | - |
| `GRAM_WEIGHT` | GRAM loss weight | - |

### Container Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `SIF_IMAGE` | Singularity image path | `$HOME/deeplearning.sif` |

## Usage Examples

### Basic Prototype Analysis

```bash
# Single dataset with default prototypes
DATASETS="dtd" \
    sbatch scripts/prototype_analysis/run_sweep_slurm.sh
```

### Multi-Dataset Sweep

```bash
# Multiple datasets
DATASETS="dtd eurosat oxford_pets" \
GPUS="0,1,2,3" \
PROTOTYPES="128 256 512 1024" \
    sbatch scripts/prototype_analysis/run_sweep_slurm.sh
```

### Continued Pretraining

```bash
# From official DINOv3 weights
CONTINUED_PRETRAINING=1 \
DATASETS="dtd eurosat" \
PROTOTYPES="128 256 512" \
    sbatch scripts/prototype_analysis/run_sweep_slurm.sh

# With custom checkpoint
CONTINUED_PRETRAINING=1 \
PRETRAIN_CHECKPOINT="/path/to/custom/checkpoint.pth" \
DATASETS="dtd" \
    sbatch scripts/prototype_analysis/run_sweep_slurm.sh
```

### With GRAM Loss

```bash
ENABLE_GRAM=1 \
GRAM_WEIGHT=0.5 \
DATASETS="dtd eurosat" \
    sbatch scripts/prototype_analysis/run_sweep_slurm.sh
```

### Full Customization

```bash
DATASETS="eurosat dtd" \
GPUS="0,1,2" \
PROTOTYPES="128 256 512 1024 2048" \
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

### Quick Testing

```bash
# Short test run
DATASETS="dtd" \
PROTOTYPES="128" \
SEEDS="0" \
PRETRAIN_EPOCHS=5 \
CLASSIFY_EPOCHS=5 \
    sbatch scripts/prototype_analysis/run_sweep_slurm.sh
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

### 2. Modify SLURM Directives

Edit `scripts/prototype_analysis/run_sweep_slurm.sh`:

```bash
# For different GPU counts
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16

# For different partitions
#SBATCH --partition=v100

# For longer/shorter jobs
#SBATCH --time=48:00:00  # 48 hours
#SBATCH --time=4:00:00   # 4 hours for testing

# For specific nodes
#SBATCH --nodelist=gpu-node-01
```

### 3. Memory Requirements

```bash
# Adjust based on your model and batch size
#SBATCH --mem=32G   # For smaller models/batch sizes
#SBATCH --mem=64G   # For larger models/batch sizes
#SBATCH --mem=128G  # For very large experiments

# Rule of thumb: 16GB per GPU minimum
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
grep "accuracy" logs/slurm-JOBID.out
grep "loss" logs/slurm-JOBID.out
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

### 2. Singularity Container Not Found

**Problem**: `ERROR: Singularity image not found`

**Solutions**:
```bash
# Verify container exists
ls -la $HOME/deeplearning.sif

# Or specify custom path
SIF_IMAGE=/path/to/your/container.sif sbatch scripts/prototype_analysis/run_sweep_slurm.sh
```

### 3. Out of Memory Errors

**Problem**: `CUDA out of memory` or similar errors

**Solutions**:
```bash
# Reduce batch size
BATCH_SIZE=64 sbatch scripts/prototype_analysis/run_sweep_slurm.sh

# Use lower precision
PRECISION=16-mixed sbatch scripts/prototype_analysis/run_sweep_slurm.sh

# Reduce number of prototypes
PROTOTYPES="128 256" sbatch scripts/prototype_analysis/run_sweep_slurm.sh
```

### 4. GPU Allocation Mismatch

**Problem**: Training uses wrong number of GPUs

**Solution**: Ensure SLURM and script match:
```bash
#SBATCH --gres=gpu:4      # Request 4 GPUs
#SBATCH --ntasks-per-node=4

# Then use:
GPUS="0,1,2,3" sbatch scripts/prototype_analysis/run_sweep_slurm.sh
```

### 5. Dataset Not Found

**Problem**: HuggingFace dataset download fails

**Solutions**:
- Ensure internet access from compute nodes
- Pre-download datasets to local storage
- Use cached datasets

### 6. Import Errors

**Problem**: `ImportError: No module named 'dinov3'`

**Solutions**:
```bash
# Ensure submodules are initialized
git submodule update --init --recursive
```

## Performance Optimization

### 1. Optimal Resource Allocation

```bash
# Rule of thumb for CPU allocation
CPUs = GPUs * 5  # 5 CPUs per GPU

# Memory allocation
Memory = GPUs * 16GB  # 16GB per GPU minimum
```

### 2. Batch Size Tuning

Start with smaller batch sizes and increase:

```bash
# Conservative starting point
BATCH_SIZE=64

# Increase if memory allows
BATCH_SIZE=128
BATCH_SIZE=256
```

### 3. PyTorch Compilation

```bash
# Enable compilation for faster training (longer startup)
COMPILE=1 sbatch scripts/prototype_analysis/run_sweep_slurm.sh
```

## Advanced Usage

### Interactive Development

For debugging, request an interactive session:

```bash
# Request interactive GPU node
salloc --gres=gpu:1 --cpus-per-task=8 --mem=16G --time=2:00:00

# Then run training interactively
python src/training/ssl/train.py \
    --config-file configs/eurosat/config_ssl_pretraining.yaml \
    --output-dir ./output_debug \
    --gpus 1 \
    --max-epochs 5
```

### Array Jobs

For multiple independent runs:

```bash
#SBATCH --array=1-10

# Use $SLURM_ARRAY_TASK_ID in script to vary parameters
SEED=$((42 + SLURM_ARRAY_TASK_ID))
```

### Multi-Node Training

For very large models (edit SLURM directives):

```bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4

# Lightning will automatically handle multi-node coordination
```

## Output Structure

After running prototype analysis sweeps:

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
│
results/
└── prototype_analysis_dinov3/
    └── {dataset}/
        ├── classification_results.csv
        └── classification_stats.csv
```

## Getting Help

- **Cluster Documentation**: Check your cluster's specific SLURM documentation
- **SLURM Manual**: `man sbatch`, `man squeue`, `man scancel`
- **PyTorch Lightning**: [Lightning Multi-GPU Guide](https://lightning.ai/docs/pytorch/stable/accelerators/gpu.html)
