#!/bin/bash

# DINOv3 Fine-tuning Script - Tissue
# Full fine-tuning of pre-trained SSL model on Tissue dataset
# Checkpoints saved to: output/tissue/checkpoints/classification/finetune/

CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=/home/marjans/DinoV3LightningTraining:$PYTHONPATH torchrun --nproc_per_node=2 src/training/classification/train.py \
    --config-file configs/tissue/config_classification.yaml \
    --checkpoint-path "output/tissue/checkpoints/pretraining/last.ckpt" \
    --output-dir output/tissue \
    --dataset-type tissue \
    --dataset-path "CustomTIFF:root=../Datasets/tissue/" \
    --num-classes 2 \
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
