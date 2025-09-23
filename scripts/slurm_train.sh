#!/bin/bash

#SBATCH --job-name=dinov3-finetune
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpu

# DINOv3 Lightning Fine-tuning SLURM Script
# Usage: sbatch slurm_train.sh

echo "Starting DINOv3 Multi-GPU PyTorch Lightning Fine-tuning on SLURM..."
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# Create logs directory
mkdir -p logs

# Set environment variables for multi-GPU training
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG="$NCCL_DEBUG_LEVEL"
export CUDA_LAUNCH_BLOCKING="$CUDA_LAUNCH_BLOCKING"

# Calculate per-GPU batch size
PER_GPU_BATCH_SIZE=$((BATCH_SIZE / GPUS))
echo "Calculated per-GPU batch size: $PER_GPU_BATCH_SIZE"

# Configuration - customize these as needed
CONFIG_FILE="configs/config_lightning_finetuning.yaml"
CHECKPOINT_PATH="dinov3/dinov3/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
OUTPUT_DIR="./output_slurm_${SLURM_JOB_ID}"
SEED=42

# Training parameters for multi-GPU
GPUS=4  # Number of GPUs requested
PRECISION="bf16-mixed"
MAX_EPOCHS=100
STRATEGY="ddp"  # Distributed Data Parallel
BATCH_SIZE=32  # Total effective batch size
PROGRESS_LOG_STEPS=50

# Checkpoint saving parameters
SAVE_EVERY_N_STEPS=5000
MAX_TO_KEEP=2

# Additional training parameters
SAMPLER_TYPE="infinite"  # Options: infinite, distributed, sharded_infinite, epoch
LIMIT_TRAIN_BATCHES=1.0  # Use full dataset (1.0) or fraction for testing
ACCUMULATE_GRAD_BATCHES=1  # Gradient accumulation steps
RESUME_FROM_CHECKPOINT=""  # Path to Lightning checkpoint to resume from (leave empty for fresh start)
COMPILE=false  # Enable PyTorch 2.0 compilation for faster training

# Performance and System Configuration
NUM_WORKERS=32           # Number of data loading workers per GPU
NCCL_DEBUG_LEVEL="WARN"  # NCCL debug level: OFF, WARN, INFO
CUDA_LAUNCH_BLOCKING=0   # Set to 1 for debugging CUDA errors

# Development and Testing Options
FAST_DEV_RUN=false       # Quick test run with minimal data
LEARNING_RATE=0.0001     # Learning rate (can override config)
DATASET_PATH=""          # Dataset path (can override config, leave empty to use config default)

# Create output directory
mkdir -p $OUTPUT_DIR

echo "=============================================="
echo "DINOv3 SLURM Multi-GPU Lightning Training"
echo "=============================================="
echo "SLURM Configuration:"
echo "  Job ID:            $SLURM_JOB_ID"
echo "  Node:              $SLURM_NODELIST"
echo "  GPUs available:    $CUDA_VISIBLE_DEVICES"
echo ""
echo "Training Configuration:"
echo "  Config file:       $CONFIG_FILE"
echo "  Checkpoint:        $CHECKPOINT_PATH"
echo "  Output dir:        $OUTPUT_DIR"
echo "  Precision:         $PRECISION"
echo "  Max epochs:        $MAX_EPOCHS"
echo "  Sampler type:      $SAMPLER_TYPE"
echo "  Learning rate:     $LEARNING_RATE"
echo "  Dataset override:  ${DATASET_PATH:-'Using config default'}"
echo ""
echo "Performance Settings:"
echo "  GPU Count:         $GPUS"
echo "  Strategy:          $STRATEGY"
echo "  Total batch size:  $BATCH_SIZE"
echo "  Per-GPU batch:     $PER_GPU_BATCH_SIZE"
echo "  Workers per GPU:   $NUM_WORKERS"
echo "  OMP threads:       $OMP_NUM_THREADS"
echo "  Accumulate grad:   $ACCUMULATE_GRAD_BATCHES"
echo "  Compile enabled:   $COMPILE"
echo ""
echo "Logging & Checkpoints:"
echo "  Progress log every: $PROGRESS_LOG_STEPS steps"
echo "  Save every:         $SAVE_EVERY_N_STEPS steps"
echo "  Keep models:        $MAX_TO_KEEP (+ last)"
echo ""
echo "Development Options:"
echo "  Fast dev run:      $FAST_DEV_RUN"
echo "  Limit batches:     $LIMIT_TRAIN_BATCHES"
echo "  CUDA blocking:     $CUDA_LAUNCH_BLOCKING"
echo "  NCCL debug:        $NCCL_DEBUG_LEVEL"
echo "  Resume from:       ${RESUME_FROM_CHECKPOINT:-'Fresh start'}"
echo "  Seed:              $SEED"
echo "=============================================="
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file not found at $CHECKPOINT_PATH"
    exit 1
fi

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at $CONFIG_FILE"
    exit 1
fi

# ============================================================================
# DYNAMIC CONFIG MODIFICATION
# ============================================================================

# Create temporary config with runtime parameters if overrides are specified
if [ -n "$LEARNING_RATE" ] || [ -n "$DATASET_PATH" ]; then
    TEMP_CONFIG_FILE="${OUTPUT_DIR}/temp_config.yaml"
    mkdir -p "$(dirname "$TEMP_CONFIG_FILE")"

    echo "Modifying config with runtime parameters..."
    if [ -n "$LEARNING_RATE" ]; then
        echo "  Learning rate: $LEARNING_RATE"
    fi
    if [ -n "$DATASET_PATH" ]; then
        echo "  Dataset path:  $DATASET_PATH"
    fi

    # Copy original config and modify parameters
    cp "$CONFIG_FILE" "$TEMP_CONFIG_FILE"

    # Modify learning rate using sed if specified
    if [ -n "$LEARNING_RATE" ]; then
        sed -i "s/lr: [0-9]*\.[0-9]*/lr: $LEARNING_RATE/g" "$TEMP_CONFIG_FILE"
    fi

    # Modify dataset path using sed if specified
    if [ -n "$DATASET_PATH" ]; then
        sed -i "s|dataset_path: .*|dataset_path: $DATASET_PATH|g" "$TEMP_CONFIG_FILE"
    fi

    # Update CONFIG_FILE to point to temporary config
    CONFIG_FILE="$TEMP_CONFIG_FILE"

    echo "Temporary config created: $TEMP_CONFIG_FILE"
    echo ""
fi

echo "Starting training..."
echo "Press Ctrl+C to stop training"
echo "========================================="

# Build training arguments using array for better handling
TRAINING_ARGS=(
    --config-file "$CONFIG_FILE"
    --checkpoint-path "$CHECKPOINT_PATH"
    --output-dir "$OUTPUT_DIR"
    --seed "$SEED"
    --gpus "$GPUS"
    --num-nodes 1
    --precision "$PRECISION"
    --strategy "$STRATEGY"
    --sampler-type "$SAMPLER_TYPE"
    --batch-size "$BATCH_SIZE"
    --accumulate-grad-batches "$ACCUMULATE_GRAD_BATCHES"
    --max-epochs "$MAX_EPOCHS"
    --log-every-n-steps 10
    --save-every-n-steps "$SAVE_EVERY_N_STEPS"
    --progress-log-every-n-steps "$PROGRESS_LOG_STEPS"
)

# Add optional development flags
if [[ "$FAST_DEV_RUN" == "true" ]]; then
    TRAINING_ARGS+=(--fast-dev-run)
    echo "⚠️  Fast dev run enabled - minimal data will be used"
fi

if [[ "$LIMIT_TRAIN_BATCHES" != "1.0" ]]; then
    TRAINING_ARGS+=(--limit-train-batches "$LIMIT_TRAIN_BATCHES")
    echo "⚠️  Training limited to $LIMIT_TRAIN_BATCHES of dataset"
fi

# Add compile flag if enabled
if [[ "$COMPILE" == "true" ]]; then
    TRAINING_ARGS+=(--compile)
    echo "⚠️  PyTorch compilation enabled - first run will be slower due to compilation"
fi

# Add resume checkpoint if specified and exists
if [ -n "$RESUME_FROM_CHECKPOINT" ] && [ -f "$RESUME_FROM_CHECKPOINT" ]; then
    TRAINING_ARGS+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
    echo "⚠️  Resuming from checkpoint: $RESUME_FROM_CHECKPOINT"
fi

echo ""
echo "Command: python src/training/train_dinov3_lightning.py [args...]"
echo ""

# Run training
python src/training/train_dinov3_lightning.py "${TRAINING_ARGS[@]}"

echo ""
echo "Training completed! Job ID: $SLURM_JOB_ID"