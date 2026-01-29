#!/bin/bash
# =============================================================================
# DINOv3 Pretraining Script - NCTCRCHE100K
# =============================================================================
# Self-supervised pretraining on NCT-CRC-HE-100K dataset
#
# Output structure:
#   output/nctcrche100k/checkpoints/pretraining/last.ckpt
#
# Usage:
#   # Default settings
#   ./scripts/pretraining/nctcrche100k_pretrain.sh
#
#   # Custom prototype count
#   NUM_PROTOTYPES=256 ./scripts/pretraining/nctcrche100k_pretrain.sh
#
#   # Custom GPUs
#   GPUS=0,1 ./scripts/pretraining/nctcrche100k_pretrain.sh
#
# For prototype analysis sweep, use:
#   python scripts/prototype_analysis/sweep_pretraining.py --dataset nctcrche100k --gpus 0,1,2
# =============================================================================

set -e

# Default configuration (can be overridden via environment variables)
GPUS="${GPUS:-0,1}"
NUM_PROTOTYPES="${NUM_PROTOTYPES:-4096}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
KOLEO_WEIGHT="${KOLEO_WEIGHT:-0.1}"
SEED="${SEED:-42}"
PRECISION="${PRECISION:-bf16-mixed}"
OUTPUT_DIR="${OUTPUT_DIR:-output/nctcrche100k}"

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="configs/NCTCRCHE100K/config_ssl_pretraining.yaml"

# Compute number of GPUs and device string
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
if [[ $NUM_GPUS -eq 1 ]]; then
    DEVICES="0"
else
    DEVICES=$(seq -s',' 0 $((NUM_GPUS - 1)))
fi

echo "========================================================================"
echo "DINOv3 SSL Pretraining - NCTCRCHE100K"
echo "========================================================================"
echo "GPUs:         $GPUS ($NUM_GPUS devices)"
echo "Prototypes:   $NUM_PROTOTYPES"
echo "Batch size:   $BATCH_SIZE"
echo "Max epochs:   $MAX_EPOCHS"
echo "Precision:    $PRECISION"
echo "Output:       $OUTPUT_DIR"
echo "========================================================================"

# Run training using the wrapper script
CUDA_VISIBLE_DEVICES="$GPUS" \
PYTHONPATH="$REPO_ROOT:$PYTHONPATH" \
python scripts/pretraining/train.py \
    --config "$CONFIG" \
    --num_prototypes "$NUM_PROTOTYPES" \
    --koleo_weight "$KOLEO_WEIGHT" \
    --devices "$DEVICES" \
    --name "proto_${NUM_PROTOTYPES}" \
    --log_base_dir "$OUTPUT_DIR/logs" \
    --checkpoint_base_dir "$OUTPUT_DIR/checkpoints/pretraining" \
    --seed "$SEED" \
    --precision "$PRECISION" \
    --max_epochs "$MAX_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --compile

echo ""
echo "========================================================================"
echo "Training complete!"
echo "Checkpoints: $OUTPUT_DIR/checkpoints/pretraining/"
echo "========================================================================"
