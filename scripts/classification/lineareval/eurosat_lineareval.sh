#!/bin/bash

# DINOv3 Linear Evaluation Script - EuroSAT
# Linear evaluation (frozen backbone) of pre-trained SSL model on EuroSAT dataset
# Checkpoints saved to: output/eurosat/checkpoints/classification/lineareval/

CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=/home/marjans/DinoV3LightningTraining:$PYTHONPATH torchrun --nproc_per_node=2 src/training/classification/train.py \
    --config-file configs/eurosat/config_classification.yaml \
    --checkpoint-path "output/eurosat/checkpoints/pretraining/last.ckpt" \
    --output-dir output/eurosat \
    --dataset-type eurosat \
    --dataset-path "HuggingFace:name=blanchon/EuroSAT_RGB" \
    --num-classes 10 \
    --gpus 2 \
    --num-nodes 1 \
    --precision bf16-mixed \
    --strategy ddp \
    --max-epochs 100 \
    --batch-size 128 \
    --learning-rate 0.0001 \
    --weight-decay 0.0 \
    --freeze-backbone \
    --encoder-type teacher \
    --log-every-n-steps 10 \
    --save-every-n-steps 100 \
    --progress-log-every-n-steps 10
