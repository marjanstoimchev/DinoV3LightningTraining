#!/bin/bash

#SBATCH --job-name=c-dtd-proto
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --partition=gpu

# =============================================================================
# DINOv3 Prototype Analysis Sweep - SLURM Script
# =============================================================================
# Runs the full prototype analysis sweep (pretrain + classify + eval)
# on a SLURM-managed node.
#
# Supports both training from scratch and continued pretraining from DINOv3.
#
# Usage:
#   sbatch scripts/prototype_analysis/run_sweep_slurm.sh
#
#   # Override defaults with environment variables:
#   DATASETS="eurosat dtd" GPUS=0,1,2 PROTOTYPES="128 256 512" \
#       sbatch scripts/prototype_analysis/run_sweep_slurm.sh
#
#   # Continued pretraining from DINOv3 official weights:
#   CONTINUED_PRETRAINING=1 DATASETS="dtd eurosat" PROTOTYPES="128 256 512" \
#       sbatch scripts/prototype_analysis/run_sweep_slurm.sh
# =============================================================================

# Navigate to repo root (where sbatch was submitted from)
cd "$SLURM_SUBMIT_DIR" || exit 1

# Create logs directory if needed
mkdir -p logs

# =============================================================================
# Singularity container setup
# =============================================================================
SIF_IMAGE="${SIF_IMAGE:-$HOME/deeplearning.sif}"

if [ ! -f "$SIF_IMAGE" ]; then
    echo "ERROR: Singularity image not found at $SIF_IMAGE"
    exit 1
fi

# Print job info
echo "============================================================"
echo "DINOv3 Prototype Analysis Sweep (Singularity)"
echo "============================================================"
echo "SLURM Job ID:    $SLURM_JOB_ID"
echo "Node:            $SLURM_NODELIST"
echo "GPUs:            $CUDA_VISIBLE_DEVICES"
echo "Working dir:     $(pwd)"
echo "Container:       $SIF_IMAGE"
echo "Start time:      $(date)"
echo "============================================================"
squeue -l
echo ""

# =============================================================================
# Configuration (override via environment variables)
# =============================================================================
DATASETS="${DATASETS:-dtd}"
GPUS="${GPUS:-0,1}"
PROTOTYPES="${PROTOTYPES:-128 256 512 1024}"
SEEDS="${SEEDS:-0 1 42}"
PRETRAIN_SEED="${PRETRAIN_SEED:-42}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-100}"
CLASSIFY_EPOCHS="${CLASSIFY_EPOCHS:-30}"
CLASSIFY_LR="${CLASSIFY_LR:-0.0001}"
CLASSIFY_MODE="${CLASSIFY_MODE:-finetune}"
BATCH_SIZE="${BATCH_SIZE:-128}"
KOLEO_WEIGHT="${KOLEO_WEIGHT:-0.1}"
OUTPUT_DIR="${OUTPUT_DIR:-prototype_analysis_dinov3}"
PRECISION="${PRECISION:-bf16-mixed}"
COMPILE="${COMPILE:-}"

# Continued pretraining support
CONTINUED_PRETRAINING="${CONTINUED_PRETRAINING:-}"
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-}"

# GRAM loss support (disabled by default)
ENABLE_GRAM="${ENABLE_GRAM:-}"
GRAM_WEIGHT="${GRAM_WEIGHT:-}"

# If CONTINUED_PRETRAINING is set, use the DINOv3 official weights
if [[ -n "$CONTINUED_PRETRAINING" ]]; then
    PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-dinov3_official_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth}"
    # Override OUTPUT_DIR only if user didn't explicitly set it
    if [[ "$OUTPUT_DIR" == "prototype_analysis_dinov3" ]]; then
        OUTPUT_DIR="prototype_analysis_dinov3_continued"
    fi
    echo "Continued pretraining enabled"
    echo "  Checkpoint: $PRETRAIN_CHECKPOINT"
fi

echo "Datasets:         $DATASETS"
echo "GPUs:             $GPUS"
echo "Prototypes:       $PROTOTYPES"
echo "Seeds:            $SEEDS"
echo "Classify mode:    $CLASSIFY_MODE"
echo "Output dir:       $OUTPUT_DIR"
echo "Precision:        $PRECISION"
[[ -n "$PRETRAIN_CHECKPOINT" ]] && echo "Pretrain ckpt:    $PRETRAIN_CHECKPOINT"
[[ -n "$COMPILE" ]] && echo "Compile:          enabled"
[[ -n "$ENABLE_GRAM" ]] && echo "GRAM loss:        enabled"
[[ -n "$GRAM_WEIGHT" ]] && echo "GRAM weight:      $GRAM_WEIGHT"
echo "============================================================"
echo ""

# =============================================================================
# Run sweep inside Singularity container
# =============================================================================

export SINGULARITYENV_TORCH_COMPILE_DISABLE=1 #

# --- Proxy settings (required for compute nodes to reach the internet) ---
# export https_proxy=http://www-proxy.ijs.si:8080
# export http_proxy=http://www-proxy.ijs.si:8080
# export no_proxy=127.0.0.0/8
# export SINGULARITYENV_https_proxy="$https_proxy"
# export SINGULARITYENV_http_proxy="$http_proxy"
# export SINGULARITYENV_no_proxy="$no_proxy"

# --- Writable cache directories ---
export HF_HOME="$HOME/.cache/huggingface"
export TORCH_HOME="$HOME/.cache/torch"
mkdir -p "$HF_HOME" "$TORCH_HOME"

export SINGULARITYENV_HF_HOME="$HF_HOME"
export SINGULARITYENV_TORCH_HOME="$TORCH_HOME"

# --- CUDA memory allocation ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"

# --- Prevent local packages from overriding container ---
export SINGULARITYENV_PYTHONNOUSERSITE=1

# --- Use random port for DDP to avoid conflicts with other jobs ---
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
export SINGULARITYENV_MASTER_PORT="$MASTER_PORT"
echo "Using MASTER_PORT: $MASTER_PORT"

# --- Multi-GPU via torchrun (single SLURM task, torchrun spawns processes) ---
# With --ntasks-per-node=1, we let torchrun handle multi-GPU process spawning
# This avoids conflicts between SLURM and Lightning's distributed launchers

# Build command arguments
SWEEP_ARGS=(
    --datasets "$DATASETS"
    --gpus "$GPUS"
    --prototypes "$PROTOTYPES"
    --seeds "$SEEDS"
    --pretrain-seed "$PRETRAIN_SEED"
    --pretrain-epochs "$PRETRAIN_EPOCHS"
    --classify-epochs "$CLASSIFY_EPOCHS"
    --classify-lr "$CLASSIFY_LR"
    --classify-mode "$CLASSIFY_MODE"
    --batch-size "$BATCH_SIZE"
    --koleo-weight "$KOLEO_WEIGHT"
    --output-dir "$OUTPUT_DIR"
    --precision "$PRECISION"
)

[[ -n "$PRETRAIN_CHECKPOINT" ]] && SWEEP_ARGS+=(--pretrain-checkpoint "$PRETRAIN_CHECKPOINT")
[[ -n "$COMPILE" ]] && SWEEP_ARGS+=(--compile)
[[ -n "$ENABLE_GRAM" ]] && SWEEP_ARGS+=(--enable-gram)
[[ -n "$GRAM_WEIGHT" ]] && SWEEP_ARGS+=(--gram-weight "$GRAM_WEIGHT")

# --- Shared memory for DataLoader workers ---
SHM_DIR="/dev/shm/${USER}_${SLURM_JOB_ID}"
mkdir -p "$SHM_DIR"

srun singularity exec --nv \
    --bind "$SLURM_SUBMIT_DIR":"$SLURM_SUBMIT_DIR" \
    --bind /tmp:/tmp \
    --bind "$HF_HOME":"$HF_HOME" \
    --bind "$TORCH_HOME":"$TORCH_HOME" \
    --bind "$SHM_DIR":/dev/shm \
    "$SIF_IMAGE" \
    ./scripts/prototype_analysis/run_sweep.sh "${SWEEP_ARGS[@]}"

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Sweep finished with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "============================================================"

# Clean up shared memory
rm -rf "$SHM_DIR"

exit $EXIT_CODE
