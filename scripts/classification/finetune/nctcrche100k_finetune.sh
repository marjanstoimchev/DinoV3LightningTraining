#!/bin/bash

# DINOv3 Fine-tuning Script - NCTCRCHE100K
# Full fine-tuning of pre-trained SSL model on NCTCRCHE100K dataset
# Checkpoints saved to: output/nctcrche100k/checkpoints/classification/finetune/

CUDA_VISIBLE_DEVICES=4,5 PYTHONPATH=/home/marjans/DinoV3LightningTraining:$PYTHONPATH torchrun --nproc_per_node=2 --master_port=29502 src/training/classification/train.py \
    --config-file configs/NCTCRCHE100K/config_classification.yaml \
    --checkpoint-path "output/nctcrche100k/checkpoints/pretraining/last.ckpt" \
    --output-dir output/nctcrche100k \
    --dataset-type NCTCRCHE100K \
    --dataset-path "HuggingFace:name=DykeF/NCTCRCHE100K" \
    --num-classes 9 \
    --gpus 2 \
    --num-nodes 1 \
    --precision bf16-mixed \
    --strategy ddp \
    --max-epochs 100 \
    --batch-size 128 \
    --learning-rate 0.0001 \
    --weight-decay 0.05 \
    --encoder-type teacher \
    --log-every-n-steps 10 \
    --save-every-n-steps 100 \
    --progress-log-every-n-steps 10
