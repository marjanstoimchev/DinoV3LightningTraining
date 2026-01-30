#!/bin/bash
# =============================================================================
# DINOv3 Prototype Analysis Sweep
# =============================================================================
# Execution order:
#   For each dataset:           <- OUTERMOST
#     For each classification seed:
#       For each prototype:
#         - Pretrain with PRETRAIN_SEED (only once, skip if exists)
#         - Classify with current seed
#     Aggregate results for this dataset
#
# Example with --datasets "eurosat dtd" --prototypes "128 256" --seeds "0 1":
#
#   === eurosat ===
#   pretrain proto_128 (seed 42) → classify proto_128 (seed 0)
#   pretrain proto_256 (seed 42) → classify proto_256 (seed 0)
#   classify proto_128 (seed 1)
#   classify proto_256 (seed 1)
#   aggregate eurosat results
#
#   === dtd ===
#   pretrain proto_128 (seed 42) → classify proto_128 (seed 0)
#   pretrain proto_256 (seed 42) → classify proto_256 (seed 0)
#   classify proto_128 (seed 1)
#   classify proto_256 (seed 1)
#   aggregate dtd results
#
# Usage:
#   ./scripts/prototype_analysis/run_sweep.sh \
#       --datasets "eurosat dtd oxford_pets" \
#       --gpus 1,2,3 \
#       --prototypes "128 256 512 1024" \
#       --seeds "0 1 42"
#
#   # Single dataset (backward compatible):
#   ./scripts/prototype_analysis/run_sweep.sh \
#       --dataset dtd \
#       --gpus 1,2,3 \
#       --prototypes "128 256 512 1024" \
#       --seeds "0 1 42"
#
#   # Quick test:
#   ./scripts/prototype_analysis/run_sweep.sh \
#       --datasets "eurosat dtd" \
#       --gpus 0 \
#       --prototypes "128 256" \
#       --seeds "0" \
#       --pretrain-epochs 10 \
#       --classify-epochs 5
# =============================================================================

set -e  # Exit on error
set -o pipefail  # Catch errors in pipelines

# -----------------------------------------------------------------------------
# Signal handling for graceful shutdown (SLURM sends SIGTERM before killing)
# -----------------------------------------------------------------------------
cleanup() {
    local exit_code=$?
    # Don't print cleanup message on successful exit
    if [[ $exit_code -eq 0 ]]; then
        return 0
    fi
    echo ""
    log "=========================================="
    log "JOB INTERRUPTED (exit code: $exit_code)"
    log "=========================================="
    if [[ -n "${DATASET:-}" ]]; then
        log "Last working on: Dataset=${DATASET:-unknown}, Proto=${NUM_PROTO:-unknown}, Seed=${SEED:-unknown}"
    fi
    log "Checkpoints saved so far should be intact."
    log "Rerun the same command to resume from last checkpoint."
    echo ""
    # Print memory info for debugging
    if command -v nvidia-smi &> /dev/null; then
        log "GPU memory at exit:"
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv 2>/dev/null || true
    fi
    exit $exit_code
}
trap cleanup SIGTERM SIGINT SIGHUP EXIT

# -----------------------------------------------------------------------------
# Usage
# -----------------------------------------------------------------------------
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  --datasets \"D1 D2 ..\"    Dataset names, space-separated (dtd, eurosat, oxford_pets, nctcrche100k, tissue, imagenet1k)"
    echo "  --dataset DATASET        Single dataset (backward compatible, same as --datasets \"DATASET\")"
    echo "  --gpus GPUS              GPU indices, comma-separated (e.g., 0,1,2,3)"
    echo ""
    echo "Optional:"
    echo "  --prototypes \"N1 N2 ..\"  Prototype counts to sweep (default: 128 256 512 1024 2048 4096)"
    echo "  --seeds \"S1 S2 ..\"       Classification seeds (default: 0 1 42)"
    echo "  --pretrain-seed SEED     Fixed seed for pretraining (default: 42)"
    echo "  --pretrain-epochs N      Pretraining epochs (default: use config)"
    echo "  --pretrain-checkpoint P  Path to DINOv3 checkpoint for continued pretraining (default: from scratch)"
    echo "  --skip-pretraining       Skip pretraining, use checkpoint directly for classification"
    echo "  --classify-epochs N      Classification epochs (default: 30)"
    echo "  --classify-lr LR         Classification learning rate (default: 0.0001)"
    echo "  --classify-mode MODE     finetune or lineareval (default: finetune)"
    echo "  --batch-size N           Batch size (default: use config)"
    echo "  --koleo-weight W         KoLeo loss weight (default: 0.1)"
    echo "  --enable-gram            Enable GRAM loss (default: disabled)"
    echo "  --gram-weight W          GRAM loss weight (default: 1.0 when enabled)"
    echo "  --output-dir DIR         Output directory (default: prototype_analysis_dinov3)"
    echo "  --precision PREC         Training precision: 32, 16, bf16-mixed, 16-mixed (default: bf16-mixed)"
    echo "  --compile                Enable PyTorch 2.0 compilation"
    echo "  --help                   Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Multiple datasets (outermost loop):"
    echo "  $0 --datasets \"eurosat dtd oxford_pets\" --gpus 1,2,3 --prototypes \"128 256\" --seeds \"0 1 42\""
    echo ""
    echo "  # Single dataset:"
    echo "  $0 --dataset dtd --gpus 1,2,3 --prototypes \"128 256 512\" --seeds \"0 1 42\""
    exit 1
}

# -----------------------------------------------------------------------------
# Parse named arguments
# -----------------------------------------------------------------------------
DATASETS=""
GPUS=""
PROTOTYPES="128 256 512 1024 2048 4096"
SEEDS="0 1 42"
PRETRAIN_SEED="42"
PRETRAIN_EPOCHS=""
PRETRAIN_CHECKPOINT=""
SKIP_PRETRAINING=""
CLASSIFY_EPOCHS="30"
CLASSIFY_LR="0.0001"
CLASSIFY_MODE="finetune"
BATCH_SIZE="128"
KOLEO_WEIGHT="0.1"
ENABLE_GRAM=""
GRAM_WEIGHT=""
OUTPUT_DIR="prototype_analysis_dinov3"
ENCODER_TYPE="teacher"
PRECISION="bf16-mixed"
COMPILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --dataset)
            # Backward compatible: single dataset
            DATASETS="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --prototypes)
            PROTOTYPES="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --pretrain-seed)
            PRETRAIN_SEED="$2"
            shift 2
            ;;
        --pretrain-epochs)
            PRETRAIN_EPOCHS="$2"
            shift 2
            ;;
        --pretrain-checkpoint)
            PRETRAIN_CHECKPOINT="$2"
            shift 2
            ;;
        --skip-pretraining)
            SKIP_PRETRAINING="true"
            shift
            ;;
        --classify-epochs)
            CLASSIFY_EPOCHS="$2"
            shift 2
            ;;
        --classify-lr)
            CLASSIFY_LR="$2"
            shift 2
            ;;
        --classify-mode)
            CLASSIFY_MODE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --koleo-weight)
            KOLEO_WEIGHT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --encoder-type)
            ENCODER_TYPE="$2"
            shift 2
            ;;
        --precision)
            PRECISION="$2"
            shift 2
            ;;
        --compile)
            COMPILE="--compile"
            shift
            ;;
        --enable-gram)
            ENABLE_GRAM="--enable-gram"
            shift
            ;;
        --gram-weight)
            GRAM_WEIGHT="$2"
            shift 2
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "$DATASETS" ]]; then
    echo "ERROR: --datasets (or --dataset) is required"
    usage
fi

if [[ -z "$GPUS" ]]; then
    echo "ERROR: --gpus is required"
    usage
fi

# Validate skip-pretraining requires a checkpoint
if [[ -n "$SKIP_PRETRAINING" && -z "$PRETRAIN_CHECKPOINT" ]]; then
    echo "ERROR: --skip-pretraining requires --pretrain-checkpoint"
    usage
fi

# Convert datasets to array
read -ra DATASET_ARRAY <<< "$DATASETS"

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Config files - DINOv3 style (from scratch)
declare -A PRETRAIN_CONFIGS=(
    ["dtd"]="configs/DTD/config_ssl_pretraining.yaml"
    ["eurosat"]="configs/eurosat/config_ssl_pretraining.yaml"
    ["oxford_pets"]="configs/oxford_pets/config_ssl_pretraining.yaml"
    ["nctcrche100k"]="configs/NCTCRCHE100K/config_ssl_pretraining.yaml"
    ["tissue"]="configs/tissue/config_ssl_pretraining.yaml"
    ["imagenet1k"]="configs/imagenet1k/config_ssl_pretraining.yaml"
)

# Config files - DINOv3 continued pretraining (initialized from DINOv3 official weights)
declare -A CONTINUED_PRETRAIN_CONFIGS=(
    ["dtd"]="configs/DTD/config_ssl_continued_pretraining.yaml"
    ["eurosat"]="configs/eurosat/config_ssl_continued_pretraining.yaml"
    ["oxford_pets"]="configs/oxford_pets/config_ssl_continued_pretraining.yaml"
    ["nctcrche100k"]="configs/NCTCRCHE100K/config_ssl_continued_pretraining.yaml"
    ["tissue"]="configs/tissue/config_ssl_continued_pretraining.yaml"
    ["imagenet1k"]="configs/imagenet1k/config_ssl_continued_pretraining.yaml"
)

declare -A CLASSIFY_CONFIGS=(
    ["dtd"]="configs/DTD/config_classification.yaml"
    ["eurosat"]="configs/eurosat/config_classification.yaml"
    ["oxford_pets"]="configs/oxford_pets/config_classification.yaml"
    ["nctcrche100k"]="configs/NCTCRCHE100K/config_classification.yaml"
    ["tissue"]="configs/tissue/config_classification.yaml"
    ["imagenet1k"]="configs/imagenet1k/config_classification.yaml"
)

# Validate all datasets before starting
for DS in "${DATASET_ARRAY[@]}"; do
    if [[ -z "${PRETRAIN_CONFIGS[$DS]}" ]]; then
        echo "ERROR: Unknown dataset: $DS"
        echo "Available: ${!PRETRAIN_CONFIGS[@]}"
        exit 1
    fi
done

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

separator() {
    echo "========================================================================"
}

get_device_string() {
    local gpu_list="$1"
    local num_gpus=$(echo "$gpu_list" | tr ',' '\n' | wc -l)
    if [[ $num_gpus -eq 1 ]]; then
        echo "0"
    else
        seq -s',' 0 $((num_gpus - 1))
    fi
}

find_checkpoint() {
    local ckpt_dir="$1"
    if [[ -f "$ckpt_dir/last.ckpt" ]]; then
        echo "$ckpt_dir/last.ckpt"
    else
        find "$ckpt_dir" -name "*.ckpt" -type f 2>/dev/null | head -1
    fi
}

# -----------------------------------------------------------------------------
# Print configuration
# -----------------------------------------------------------------------------
separator
echo "DINOV3 PROTOTYPE ANALYSIS SWEEP"
separator
echo "Datasets:         $DATASETS"
echo "GPUs:             $GPUS"
echo "Prototypes:       $PROTOTYPES"
echo "Pretrain seed:    $PRETRAIN_SEED"
echo "Classify seeds:   $SEEDS"
echo "Output:           $OUTPUT_DIR"
echo "Classify mode:    $CLASSIFY_MODE"
echo "Classify epochs:  $CLASSIFY_EPOCHS"
echo "Classify LR:      $CLASSIFY_LR"
echo "Precision:        $PRECISION"
[[ -n "$PRETRAIN_EPOCHS" ]] && echo "Pretrain epochs:  $PRETRAIN_EPOCHS"
[[ -n "$PRETRAIN_CHECKPOINT" ]] && echo "Pretrain ckpt:    $PRETRAIN_CHECKPOINT"
[[ -n "$SKIP_PRETRAINING" ]] && echo "Skip pretraining: yes (classification only)"
[[ -n "$BATCH_SIZE" ]] && echo "Batch size:       $BATCH_SIZE"
[[ -n "$COMPILE" ]] && echo "Compile:          enabled"
[[ -n "$ENABLE_GRAM" ]] && echo "GRAM loss:        enabled"
[[ -n "$GRAM_WEIGHT" ]] && echo "GRAM weight:      $GRAM_WEIGHT"
separator
echo ""

# Convert to arrays
read -ra PROTO_ARRAY <<< "$PROTOTYPES"
read -ra SEED_ARRAY <<< "$SEEDS"

DEVICES=$(get_device_string "$GPUS")

TOTAL_DATASETS=${#DATASET_ARRAY[@]}
CURRENT_DATASET_IDX=0

# -----------------------------------------------------------------------------
# Main loop - DATASETS as outermost
# -----------------------------------------------------------------------------
for DATASET in "${DATASET_ARRAY[@]}"; do
    CURRENT_DATASET_IDX=$((CURRENT_DATASET_IDX + 1))

    separator
    log "DATASET: $DATASET ($CURRENT_DATASET_IDX/$TOTAL_DATASETS)"
    separator

    # Get config files for this dataset
    # Use continued pretraining config if pretrain checkpoint is provided
    if [[ -n "$PRETRAIN_CHECKPOINT" ]]; then
        PRETRAIN_CONFIG="${CONTINUED_PRETRAIN_CONFIGS[$DATASET]}"
        log "[$DATASET] Using continued pretraining config: $PRETRAIN_CONFIG"
    else
        PRETRAIN_CONFIG="${PRETRAIN_CONFIGS[$DATASET]}"
    fi
    CLASSIFY_CONFIG="${CLASSIFY_CONFIGS[$DATASET]}"

    for SEED in "${SEED_ARRAY[@]}"; do
        separator
        log "[$DATASET] CLASSIFICATION SEED: $SEED"
        separator

        for NUM_PROTO in "${PROTO_ARRAY[@]}"; do
            echo ""
            log "[$DATASET] Processing: proto_${NUM_PROTO} | classify_seed_${SEED}"

            # Directories
            PROTO_DIR="$OUTPUT_DIR/pretraining/$DATASET/proto_${NUM_PROTO}"
            CKPT_DIR="$PROTO_DIR/checkpoints"
            LOG_DIR="$PROTO_DIR/logs"

            CLASS_DIR="$OUTPUT_DIR/classification/$DATASET/proto_${NUM_PROTO}/seed_${SEED}"
            CLASS_CKPT_DIR="$CLASS_DIR/checkpoints"
            CLASS_LOG_DIR="$CLASS_DIR/logs"

            # -----------------------------------------------------------------
            # PRETRAINING (once per prototype, uses PRETRAIN_SEED)
            # -----------------------------------------------------------------
            if [[ -n "$SKIP_PRETRAINING" ]]; then
                # Skip pretraining, use provided checkpoint directly
                PRETRAIN_CKPT="$PRETRAIN_CHECKPOINT"
                log "[$DATASET] Skipping pretraining, using checkpoint: $PRETRAIN_CKPT"
            else
                PRETRAIN_CKPT=$(find_checkpoint "$CKPT_DIR" 2>/dev/null || echo "")

                if [[ -z "$PRETRAIN_CKPT" ]]; then
                    log "[$DATASET] Pretraining proto_${NUM_PROTO} (seed ${PRETRAIN_SEED})..."

                    mkdir -p "$CKPT_DIR" "$LOG_DIR"

                    PRETRAIN_CMD=(
                        python scripts/pretraining/train.py
                        --config "$PRETRAIN_CONFIG"
                        --num_prototypes "$NUM_PROTO"
                        --koleo_weight "$KOLEO_WEIGHT"
                        --devices "$DEVICES"
                        --name "proto_${NUM_PROTO}"
                        --log_base_dir "$LOG_DIR"
                        --checkpoint_base_dir "$CKPT_DIR"
                        --seed "$PRETRAIN_SEED"
                        --precision "$PRECISION"
                    )

                    [[ -n "$PRETRAIN_EPOCHS" ]] && PRETRAIN_CMD+=(--max_epochs "$PRETRAIN_EPOCHS")
                    [[ -n "$BATCH_SIZE" ]] && PRETRAIN_CMD+=(--batch_size "$BATCH_SIZE")
                    [[ -n "$COMPILE" ]] && PRETRAIN_CMD+=(--compile)
                    [[ -n "$PRETRAIN_CHECKPOINT" ]] && PRETRAIN_CMD+=(--pretrain_checkpoint "$PRETRAIN_CHECKPOINT")
                    [[ -n "$ENABLE_GRAM" ]] && PRETRAIN_CMD+=(--enable_gram)
                    [[ -n "$GRAM_WEIGHT" ]] && PRETRAIN_CMD+=(--gram_weight "$GRAM_WEIGHT")

                    # Run with PRETRAIN_SEED
                    if ! CUDA_VISIBLE_DEVICES="$GPUS" PL_GLOBAL_SEED="$PRETRAIN_SEED" PYTHONPATH="$REPO_ROOT:$PYTHONPATH" "${PRETRAIN_CMD[@]}"; then
                        log "ERROR: [$DATASET] Pretraining command failed for proto_${NUM_PROTO}"
                        log "       Check GPU memory, disk space, or config issues"
                        # Exit with error - can't continue without pretraining
                        exit 1
                    fi

                    PRETRAIN_CKPT=$(find_checkpoint "$CKPT_DIR")

                    if [[ -z "$PRETRAIN_CKPT" ]]; then
                        log "ERROR: [$DATASET] Pretraining failed for proto_${NUM_PROTO}"
                        continue
                    fi

                    log "[$DATASET] Pretraining complete: $PRETRAIN_CKPT"
                else
                    log "[$DATASET] Using existing checkpoint: $PRETRAIN_CKPT"
                fi
            fi

            # -----------------------------------------------------------------
            # CLASSIFICATION (with current SEED)
            # -----------------------------------------------------------------
            CLASS_EXISTING=$(find_checkpoint "$CLASS_CKPT_DIR" 2>/dev/null || echo "")

            if [[ -n "$CLASS_EXISTING" ]]; then
                log "[$DATASET] Classification already done: $CLASS_EXISTING"
                continue
            fi

            log "[$DATASET] Classification proto_${NUM_PROTO} (seed ${SEED})..."

            mkdir -p "$CLASS_CKPT_DIR" "$CLASS_LOG_DIR"

            CLASSIFY_CMD=(
                python scripts/classification/train.py
                --config "$CLASSIFY_CONFIG"
                --pretrained_path "$PRETRAIN_CKPT"
                --max_epochs "$CLASSIFY_EPOCHS"
                --learning_rate "$CLASSIFY_LR"
                --devices "$DEVICES"
                --encoder_type "$ENCODER_TYPE"
                --checkpoint_dir "$CLASS_CKPT_DIR"
                --log_dir "$CLASS_LOG_DIR"
                --name "proto_${NUM_PROTO}_${CLASSIFY_MODE}_seed_${SEED}"
                --logger csv
                --seed "$SEED"
                --precision "$PRECISION"
            )

            [[ -n "$BATCH_SIZE" ]] && CLASSIFY_CMD+=(--batch_size "$BATCH_SIZE")
            [[ "$CLASSIFY_MODE" == "lineareval" ]] && CLASSIFY_CMD+=(--freeze_backbone)

            if ! CUDA_VISIBLE_DEVICES="$GPUS" PL_GLOBAL_SEED="$SEED" PYTHONPATH="$REPO_ROOT:$PYTHONPATH" "${CLASSIFY_CMD[@]}"; then
                log "ERROR: [$DATASET] Classification command failed for proto_${NUM_PROTO} seed_${SEED}"
                log "       Check GPU memory, disk space, or config issues"
                # Continue to next configuration instead of failing entire sweep
                continue
            fi

            log "[$DATASET] Classification complete: proto_${NUM_PROTO} seed_${SEED}"

            # -----------------------------------------------------------------
            # EVALUATION (run on test set after training, single GPU, best checkpoint)
            # -----------------------------------------------------------------
            if [[ -d "$CLASS_CKPT_DIR" ]]; then
                log "[$DATASET] Evaluating proto_${NUM_PROTO} seed_${SEED} (best checkpoint)..."

                # Results go directly into the classification run directory
                EVAL_OUTPUT_DIR="$CLASS_DIR/results"

                EVAL_CMD=(
                    python scripts/classification/eval.py
                    --config "$CLASSIFY_CONFIG"
                    --checkpoint "$CLASS_CKPT_DIR"
                    --checkpoint_type best
                    --output_dir "$EVAL_OUTPUT_DIR"
                    --devices "0"
                )

                [[ -n "$BATCH_SIZE" ]] && EVAL_CMD+=(--batch_size "$BATCH_SIZE")

                # Use first GPU only for eval
                FIRST_GPU="${GPUS%%,*}"
                CUDA_VISIBLE_DEVICES="$FIRST_GPU" PYTHONPATH="$REPO_ROOT:$PYTHONPATH" "${EVAL_CMD[@]}" || log "WARNING: [$DATASET] Evaluation failed for proto_${NUM_PROTO} seed_${SEED}"

                log "[$DATASET] Evaluation complete: proto_${NUM_PROTO} seed_${SEED}"
            fi
        done
    done

    # -------------------------------------------------------------------------
    # Aggregate results for this dataset
    # -------------------------------------------------------------------------
    separator
    log "[$DATASET] Aggregating results..."
    separator

    if PYTHONPATH="$REPO_ROOT:$PYTHONPATH" python scripts/prototype_analysis/sweep_classification.py \
        --dataset "$DATASET" \
        --pretraining_dir "$OUTPUT_DIR/pretraining/$DATASET" \
        --prototypes ${PROTO_ARRAY[@]} \
        --seeds ${SEED_ARRAY[@]} \
        --mode "$CLASSIFY_MODE" \
        --gpus "$GPUS" \
        --output_dir "$OUTPUT_DIR" \
        --precision "$PRECISION" \
        --skip_existing; then
        log "[$DATASET] Results: $OUTPUT_DIR/$DATASET/"
    else
        log "WARNING: [$DATASET] Aggregation failed (non-fatal, continuing...)"
    fi
    echo ""
done

separator
log "SWEEP COMPLETE FOR ALL DATASETS: $DATASETS"
separator
echo "Results directory: $OUTPUT_DIR/"
