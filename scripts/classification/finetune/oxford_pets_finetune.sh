#!/bin/bash

# DINOv3 Fine-tuning Script - Oxford Pets
# Full fine-tuning of pre-trained SSL model on Oxford Pets dataset
# Checkpoints saved to: output/oxford_pets/checkpoints/classification/finetune/

CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=/home/marjans/DinoV3LightningTraining:$PYTHONPATH torchrun --nproc_per_node=2 src/training/classification/train.py \
    --config-file configs/oxford_pets/config_classification.yaml \
    --checkpoint-path "output/oxford_pets/checkpoints/pretraining/last.ckpt" \
    --output-dir output/oxford_pets \
    --dataset-type oxford_pets \
    --dataset-path "HuggingFace:name=timm/oxford-iiit-pet" \
    --num-classes 37 \
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
