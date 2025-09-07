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
export NCCL_DEBUG=INFO

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

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Checkpoint: $CHECKPOINT_PATH"  
echo "  Output dir: $OUTPUT_DIR"
echo "  GPUs: $GPUS"
echo "  Strategy: $STRATEGY"
echo "  Precision: $PRECISION"
echo "  Max epochs: $MAX_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Seed: $SEED"
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

echo "Starting training..."
echo "========================================="

# Run training
python src/training/train_dinov3_lightning.py \
    --config-file $CONFIG_FILE \
    --checkpoint-path $CHECKPOINT_PATH \
    --output-dir $OUTPUT_DIR \
    --seed $SEED \
    --gpus $GPUS \
    --precision $PRECISION \
    --max-epochs $MAX_EPOCHS \
    --strategy $STRATEGY \
    --log-every-n-steps 10 \
    --save-every-n-steps $SAVE_EVERY_N_STEPS \
    --accumulate-grad-batches 1 \
    --progress-log-every-n-steps $PROGRESS_LOG_STEPS \
    --num-nodes 1

echo ""
echo "Training completed! Job ID: $SLURM_JOB_ID"