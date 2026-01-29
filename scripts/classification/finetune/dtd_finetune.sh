#!/bin/bash

# DINOv3 Fine-tuning Script - DTD
# Full fine-tuning of pre-trained SSL model on DTD dataset
# Checkpoints saved to: output/dtd/checkpoints/classification/finetune/

CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=/home/marjans/DinoV3LightningTraining:$PYTHONPATH torchrun --nproc_per_node=2 src/training/classification/train.py \
    --config-file configs/DTD/config_classification.yaml \
    --checkpoint-path "output/dtd/checkpoints/pretraining/last.ckpt" \
    --output-dir output/dtd \
    --dataset-type DTD \
    --dataset-path "HuggingFace:name=cansa/Describable-Textures-Dataset-DTD" \
    --num-classes 47 \
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
