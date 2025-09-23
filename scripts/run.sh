#!/bin/bash
#
# DINOv3 Multi-GPU PyTorch Lightning Training Script
# Enhanced version with full control over training parameters
#

set -e

# ============================================================================
# CONFIGURATION SECTION - Edit these parameters as needed
# ============================================================================

# GPU Configuration
CUDA_VISIBLE_DEVICES="0,1,2,4"  # Which physical GPUs to use
GPUS=4                           # Number of GPUs (should match CUDA_VISIBLE_DEVICES count)

# Training Configuration  
CONFIG_FILE="configs/config_lightning_finetuning_v2.yaml"
CHECKPOINT_PATH="dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
OUTPUT_DIR="./output_hf"
SEED=42

# Model and Training Parameters
PRECISION="bf16-mixed"           # Training precision: 32, 16, bf16-mixed, 16-mixed
MAX_EPOCHS=100                   # Number of training epochs
STRATEGY="ddp"                   # Training strategy: auto, ddp, ddp_sharded
BATCH_SIZE=32                    # Total effective batch size across all GPUs
SAMPLER_TYPE="infinite"             # Sampler type: infinite, distributed, sharded_infinite
LEARNING_RATE=0.0001            # Learning rate for optimizer
# DATASET_PATH="CustomTIFF:root=../Datasets/composite/"  # Dataset path and format

DATASET_PATH="HuggingFace:name=jonathancui/oxford-pets"

# Performance and System Configuration
NUM_WORKERS=32                    # Number of data loading workers per GPU
OMP_NUM_THREADS=8               # OpenMP threads for CPU operations
COMPILE_ENABLED=false           # Enable PyTorch 2.0 compilation (experimental)

# Logging and Progress
PROGRESS_LOG_STEPS=1          # Log progress every N training steps  
LOG_EVERY_N_STEPS=1           # Log metrics every N steps
NCCL_DEBUG_LEVEL="WARN"        # NCCL debug level: OFF, WARN, INFO

# Checkpoint Configuration
SAVE_EVERY_N_STEPS=1000        # Save checkpoint every N steps
MAX_TO_KEEP=2                 # Keep last N best models (plus always keep last)

# Development and Testing Options
FAST_DEV_RUN=false             # Quick test run with minimal data
LIMIT_TRAIN_BATCHES=1.0        # Limit training batches (1.0 = all, 0.1 = 10%)
CUDA_LAUNCH_BLOCKING=0         # Set to 1 for debugging CUDA errors

# Resume Training Options
RESUME_FROM_CHECKPOINT=""      # Path to Lightning checkpoint to resume from (leave empty for fresh start)

# ============================================================================
# DERIVED CONFIGURATION - Automatically calculated
# ============================================================================

# Calculate per-GPU batch size
PER_GPU_BATCH_SIZE=$((BATCH_SIZE / GPUS))
echo "Calculated per-GPU batch size: $PER_GPU_BATCH_SIZE"

# Auto-configure strategy and sampler for single vs multi-GPU
if [[ $GPUS -eq 1 ]]; then
    STRATEGY="auto"
    USE_TORCHRUN=false
    # Use epoch sampler for single GPU (official DINOv3 behavior)
    if [[ "$SAMPLER_TYPE" == "distributed" ]]; then
        SAMPLER_TYPE="epoch"
        echo "Single GPU detected: switching from distributed to epoch sampler"
    fi
else
    STRATEGY="ddp"
    USE_TORCHRUN=true
fi

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

# Set environment variables for multi-GPU training
export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export OMP_NUM_THREADS="$OMP_NUM_THREADS"
export NCCL_DEBUG="$NCCL_DEBUG_LEVEL"
export CUDA_LAUNCH_BLOCKING="$CUDA_LAUNCH_BLOCKING"

# PyTorch compilation settings
if [[ "$COMPILE_ENABLED" == "true" ]]; then
    export TORCH_COMPILE_DEBUG=1
    COMPILE_FLAG="--compile"
else
    COMPILE_FLAG=""
fi

# ============================================================================
# CONFIGURATION DISPLAY
# ============================================================================

echo "=============================================="
echo "DINOv3 Multi-GPU Lightning Training"
echo "=============================================="
echo "GPU Configuration:"
echo "  Physical GPUs:     $CUDA_VISIBLE_DEVICES"
echo "  GPU Count:         $GPUS"
echo "  Strategy:          $STRATEGY"
echo "  Use torchrun:      $USE_TORCHRUN"
echo ""
echo "Training Configuration:"
echo "  Config file:       $CONFIG_FILE"
echo "  Checkpoint:        $CHECKPOINT_PATH"  
echo "  Output dir:        $OUTPUT_DIR"
echo "  Precision:         $PRECISION"
echo "  Max epochs:        $MAX_EPOCHS"
echo "  Sampler type:      $SAMPLER_TYPE"
echo "  Learning rate:     $LEARNING_RATE"
echo "  Dataset path:      $DATASET_PATH"
echo ""
echo "Performance Settings:"
echo "  Total batch size:  $BATCH_SIZE"
echo "  Per-GPU batch:     $PER_GPU_BATCH_SIZE"
echo "  Workers per GPU:   $NUM_WORKERS"
echo "  OMP threads:       $OMP_NUM_THREADS"
echo "  Compile enabled:   $COMPILE_ENABLED"
echo ""
echo "Logging & Checkpoints:"
echo "  Progress log every: $PROGRESS_LOG_STEPS steps"
echo "  Metrics log every:  $LOG_EVERY_N_STEPS steps"  
echo "  Save every:         $SAVE_EVERY_N_STEPS steps"
echo "  Keep models:        $MAX_TO_KEEP (+ last)"
echo ""
echo "Development Options:"
echo "  Fast dev run:      $FAST_DEV_RUN"
echo "  Limit batches:     $LIMIT_TRAIN_BATCHES"
echo "  CUDA blocking:     $CUDA_LAUNCH_BLOCKING"
echo "  Resume from:       ${RESUME_FROM_CHECKPOINT:-'Fresh start'}"
echo "  Seed:              $SEED"
echo "=============================================="
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file not found at $CHECKPOINT_PATH"
    echo "Please make sure the DINOv3 checkpoint is available."
    exit 1
fi

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at $CONFIG_FILE"
    exit 1
fi

# ============================================================================
# VALIDATION AND EXECUTION
# ============================================================================

# ============================================================================
# DYNAMIC CONFIG MODIFICATION
# ============================================================================

# Create temporary config with runtime parameters
TEMP_CONFIG_FILE="${OUTPUT_DIR}/temp_config.yaml"
mkdir -p "$(dirname "$TEMP_CONFIG_FILE")"

echo "Modifying config with runtime parameters..."
echo "  Learning rate: $LEARNING_RATE"
echo "  Dataset path:  $DATASET_PATH"

# Copy original config and modify learning rate and dataset path
cp "$CONFIG_FILE" "$TEMP_CONFIG_FILE"

# Modify learning rate using sed
sed -i "s/lr: [0-9]*\.[0-9]*/lr: $LEARNING_RATE/g" "$TEMP_CONFIG_FILE"

# Modify dataset path using sed (handle various formats)
sed -i "s|dataset_path: .*|dataset_path: $DATASET_PATH|g" "$TEMP_CONFIG_FILE"

# Update CONFIG_FILE to point to temporary config
CONFIG_FILE="$TEMP_CONFIG_FILE"

echo "Temporary config created: $TEMP_CONFIG_FILE"
echo ""

echo "Starting training..."
echo "Press Ctrl+C to stop training"
echo ""

# Build training arguments
TRAINING_ARGS=(
    --config-file "$CONFIG_FILE"
    --checkpoint-path "$CHECKPOINT_PATH"
    --output-dir "$OUTPUT_DIR"
    --seed "$SEED"
    --gpus "$GPUS"
    --precision "$PRECISION"
    --max-epochs "$MAX_EPOCHS"
    --strategy "$STRATEGY"
    --sampler-type "$SAMPLER_TYPE"
    --batch-size "$BATCH_SIZE"
    --log-every-n-steps "$LOG_EVERY_N_STEPS"
    --save-every-n-steps "$SAVE_EVERY_N_STEPS"
    --progress-log-every-n-steps "$PROGRESS_LOG_STEPS"
    --accumulate-grad-batches 1
    --num-nodes 1
)

# Add optional development flags
if [[ "$FAST_DEV_RUN" == "true" ]]; then
    TRAINING_ARGS+=(--fast-dev-run)
fi

if [[ "$LIMIT_TRAIN_BATCHES" != "1.0" ]]; then
    TRAINING_ARGS+=(--limit-train-batches "$LIMIT_TRAIN_BATCHES")
fi

# Add compile flag if enabled (requires training script support)
if [[ "$COMPILE_ENABLED" == "true" ]]; then
    TRAINING_ARGS+=(--compile)
    echo "⚠️  PyTorch compilation enabled - first run will be slower due to compilation"
fi

# Add resume checkpoint if specified and exists
if [ -n "$RESUME_FROM_CHECKPOINT" ] && [ -f "$RESUME_FROM_CHECKPOINT" ]; then
    TRAINING_ARGS+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
    echo "⚠️  Resuming from checkpoint: $RESUME_FROM_CHECKPOINT"
fi

# Execute training with appropriate launcher
if [[ "$USE_TORCHRUN" == "true" ]]; then
    echo "Launching $GPUS processes with torchrun..."
    echo "Command: torchrun --nproc_per_node=$GPUS src/training/train_dinov3_lightning.py [args...]"
    echo ""
    
    exec torchrun --nproc_per_node="$GPUS" src/training/train_dinov3_lightning.py "${TRAINING_ARGS[@]}"
else
    echo "Launching single process training..."
    echo "Command: python src/training/train_dinov3_lightning.py [args...]"
    echo ""
    
    exec python src/training/train_dinov3_lightning.py "${TRAINING_ARGS[@]}"
fi

echo ""
echo "Training completed or interrupted!"