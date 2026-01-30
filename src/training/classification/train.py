#!/usr/bin/env python3
"""
Train a classifier using a pretrained DINOv3 SSL encoder.
Supports both fine-tuning (trainable backbone) and linear evaluation (frozen backbone).

Output directory structure:
  output/{dataset}/
  ├── checkpoints/
  │   ├── pretraining/           # SSL pretraining checkpoints
  │   └── classification/
  │       ├── lineareval/
  │       │   └── seed_{seed}/   # Linear evaluation checkpoints (frozen backbone)
  │       └── finetune/
  │           └── seed_{seed}/   # Fine-tuning checkpoints (trainable backbone)
  ├── tensorboard_logs/
  └── csv_logs/

Usage examples:

  # Fine-tuning (trainable backbone) with seed
  python src/training/classification/train.py \\
      --config-file configs/eurosat/config_classification.yaml \\
      --checkpoint-path output/eurosat/checkpoints/pretraining/last.ckpt \\
      --output-dir output/eurosat \\
      --seed 42

  # Linear evaluation (frozen backbone) with seed
  python src/training/classification/train.py \\
      --config-file configs/eurosat/config_classification.yaml \\
      --checkpoint-path output/eurosat/checkpoints/pretraining/last.ckpt \\
      --output-dir output/eurosat \\
      --freeze-backbone \\
      --seed 42
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
    TQDMProgressBar,
)
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import torch

# Add repo root to import path
ROOT = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from src.classification.models.linear_classifier import LinearClassifier
from src.classification.learners.classification_learner import ClassificationLearner
from src.classification.data.datamodule import ClassificationDataModule
from omegaconf import OmegaConf


def parse_args():
    p = argparse.ArgumentParser(
        description="Train Linear Classifier with pretrained DINOv3 encoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config-file", type=str, required=True, help="Path to config.yaml")
    p.add_argument("--checkpoint-path", type=str, required=True, help="Path to pretrained DINOv3 checkpoint")
    p.add_argument("--output-dir", type=str, default="./classification_output", help="Path to save logs and checkpoints")

    # Training arguments
    p.add_argument("--num-classes", type=int, required=True, help="Number of classes for classification")
    p.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for classifier training")
    p.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay for optimizer")
    p.add_argument("--freeze-backbone", action="store_true", help="Freeze the SSL backbone during fine-tuning")
    p.add_argument("--encoder-type", default="teacher", choices=["teacher", "student"],
                   help="Which encoder to use from SSL model (teacher or student)")

    # Lightning-specific arguments
    p.add_argument("--gpus", default=1, type=int, help="Number of GPUs to use")
    p.add_argument("--num-nodes", default=1, type=int, help="Number of nodes for distributed training")
    p.add_argument("--precision", default="bf16-mixed", type=str,
                   choices=["32", "16", "bf16-mixed", "16-mixed"], help="Training precision")
    p.add_argument("--strategy", default="auto", type=str,
                   help="Training strategy (auto, ddp, ddp_sharded, etc.)")
    p.add_argument("--max-epochs", default=100, type=int, help="Number of epochs for classification training")
    p.add_argument("--batch-size", default=64, type=int, help="Batch size for classification training")

    # Logging arguments
    p.add_argument("--log-every-n-steps", default=10, type=int, help="Log every N training steps")
    p.add_argument("--save-every-n-steps", default=100, type=int, help="Save checkpoint every N steps")
    p.add_argument("--progress-log-every-n-steps", default=10, type=int, help="Log progress every N training steps")

    # Data loading arguments
    p.add_argument("--sampler-type", default="distributed", type=str,
                   choices=["infinite", "distributed", "sharded_infinite", "epoch"],
                   help="Sampler type for data loading")

    # Dataset selection arguments
    p.add_argument("--dataset-type", default="custom", type=str,
                   choices=["custom", "NCTCRCHE100K", "eurosat", "DTD", "imagenet1k", "oxford_pets", "tissue"],
                   help="Type of dataset to use for training")
    p.add_argument("--dataset-path", default=None, type=str,
                   help="Path to dataset (overrides config if provided)")

    # Seed for reproducibility
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    return p.parse_args()


def main():
    args = parse_args()

    # Set seed for reproducibility
    seed = args.seed
    pl.seed_everything(seed, workers=True)
    print(f"Using seed: {seed}")

    # Setup output directory and logging
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create the linear classifier
    classifier = LinearClassifier(
        num_classes=args.num_classes,
        img_size=256,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        num_storage_tokens=None,  # Auto-detect from checkpoint
        pretrained_path=args.checkpoint_path,
        use_cls_token=True,
        encoder_type=args.encoder_type
    )

    # Freeze backbone if requested
    if args.freeze_backbone:
        classifier.freeze_backbone()

    # Create the classification learner
    classification_learner = ClassificationLearner(
        model=classifier,
        num_classes=args.num_classes,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        freeze_backbone=args.freeze_backbone,
        encoder_type=args.encoder_type
    )

    # Create data module for classification
    sys.path.append(str(ROOT / 'dinov3'))
    from dinov3.configs import get_default_config
    ssl_cfg_original = OmegaConf.load(args.config_file)
    default_cfg = get_default_config()
    ssl_cfg = OmegaConf.merge(default_cfg, ssl_cfg_original)

    # Detect SLURM-based distribution
    slurm_ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    slurm_launch = os.environ.get("SLURM_LAUNCH", "")

    # Batch size is TOTAL, divide by workers (same as MPP)
    if slurm_launch and slurm_ntasks > 1:
        num_workers = slurm_ntasks
    else:
        num_workers = args.gpus if args.gpus > 0 else 1
    per_gpu_batch_size = args.batch_size // num_workers
    print(f"Batch size: {args.batch_size} total -> {per_gpu_batch_size} per GPU ({num_workers} workers)")

    # Create a temporary config for the data module
    class_cfg = OmegaConf.create({
        'train': {
            'dataset_path': args.dataset_path or ssl_cfg.train.dataset_path,
            'batch_size_per_gpu': per_gpu_batch_size,
            'num_workers': 8
        },
        'crops': {
            'global_crops_size': 256,
            'local_crops_size': 112
        },
        'compute_precision': {
            'param_dtype': 'bf16' if 'bf16' in args.precision else 'fp32'
        }
    })

    # Update dataset path if provided
    if args.dataset_path:
        class_cfg.train.dataset_path = args.dataset_path

    # Update dataset type if needed (only use defaults if --dataset-path was not provided)
    if args.dataset_type and args.dataset_type != "custom":
        dataset_paths = {
            "NCTCRCHE100K": "HuggingFace:name=DykeF/NCTCRCHE100K",
            "eurosat": "HuggingFace:name=blanchon/EuroSAT_RGB",
            "DTD": "HuggingFace:name=cansa/Describable-Textures-Dataset-DTD",
            "imagenet1k": "HuggingFace:name=ILSVRC/imagenet-1k",
            "oxford_pets": "HuggingFace:name=timm/oxford-iiit-pet",
            "tissue": "CustomTIFF:root=../Datasets/tissue/"
        }
        # Only use default dataset path if --dataset-path was not explicitly provided
        if args.dataset_type in dataset_paths and not args.dataset_path:
            class_cfg.train.dataset_path = dataset_paths[args.dataset_type]

    # Create data module
    datamodule = ClassificationDataModule(
        cfg=class_cfg,
        ssl_model=None,
        sampler_type=args.sampler_type,
        num_workers=8,
        batch_size=per_gpu_batch_size
    )

    # Create callbacks
    callbacks = []

    # Determine training mode for checkpoint organization
    training_mode = "lineareval" if args.freeze_backbone else "finetune"

    # Create checkpoints directory with lineareval/finetune subdirectory and seed
    # Structure: output_dir/checkpoints/classification/lineareval/seed_{seed}/ or /finetune/seed_{seed}/
    checkpoint_dir = Path(output_dir) / "checkpoints" / "classification" / training_mode / f"seed_{seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Classification checkpoint callback - monitor validation loss
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=f"{training_mode}_seed_{seed}_epoch_{{epoch:02d}}_step_{{step:06d}}_loss_{{val_loss:.4f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        every_n_train_steps=args.save_every_n_steps,
        save_last=True,
        verbose=True,
        auto_insert_metric_name=True,
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    # Standard Lightning progress bar
    progress_bar = TQDMProgressBar(refresh_rate=1)
    callbacks.append(progress_bar)

    # Early stopping callback (from config)
    early_stopping_cfg = ssl_cfg.get('early_stopping', {})
    if early_stopping_cfg.get('enabled', False):
        early_stopping = EarlyStopping(
            monitor=early_stopping_cfg.get('monitor', 'val_loss'),
            patience=early_stopping_cfg.get('patience', 10),
            mode=early_stopping_cfg.get('mode', 'min'),
            verbose=True,
        )
        callbacks.append(early_stopping)
        print(f"Early stopping enabled: patience={early_stopping_cfg.get('patience', 10)}")

    # Create loggers with seed in the version
    log_version = f"{training_mode}_seed_{seed}"
    tb_logger = TensorBoardLogger(
        save_dir=output_dir,
        name="tensorboard_logs",
        version=log_version,
    )
    csv_logger = CSVLogger(
        save_dir=output_dir,
        name="csv_logs",
        version=log_version,
    )
    loggers = [tb_logger, csv_logger]

    # Detect SLURM-based distribution for trainer
    slurm_procid = os.environ.get("SLURM_PROCID")

    # Determine effective devices and strategy
    effective_devices = args.gpus
    effective_strategy = args.strategy

    if slurm_launch and slurm_ntasks > 1 and slurm_procid is not None:
        # SLURM is handling distribution - devices must match ntasks-per-node
        effective_devices = slurm_ntasks
        effective_strategy = "ddp"
        print(f"SLURM distribution detected: task {slurm_procid}/{slurm_ntasks}")
        print(f"Using devices={slurm_ntasks}, strategy=ddp (Lightning SLURM plugin)")
    elif args.gpus > 1:
        effective_strategy = "ddp" if args.strategy == "auto" else args.strategy

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=effective_devices,
        num_nodes=args.num_nodes,
        strategy=effective_strategy,
        precision=args.precision,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=args.log_every_n_steps,
        sync_batchnorm=True if slurm_ntasks > 1 or args.gpus > 1 or args.num_nodes > 1 else False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,
        limit_train_batches=1.0,
        val_check_interval=1.0,
    )

    # Start training
    trainer.fit(
        classification_learner,
        datamodule=datamodule,
    )

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"{'='*60}")
    print(f"  Mode: {training_mode}")
    print(f"  Seed: {seed}")
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"  Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
