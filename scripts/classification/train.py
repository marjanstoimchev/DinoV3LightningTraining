#!/usr/bin/env python
"""
Wrapper script for DINOv3 classification training.
Provides a CLI interface compatible with the prototype analysis pipeline.

This script wraps src/training/classification/train.py with a simplified interface.

Usage:
    python scripts/classification/train.py \
        --config configs/eurosat/config_classification.yaml \
        --pretrained_path prototype_analysis/pretraining/eurosat/proto_256/checkpoints/last.ckpt \
        --max_epochs 30 \
        --devices 0,1 \
        --checkpoint_dir prototype_analysis/classification/eurosat/proto_256/seed_0/checkpoints \
        --log_dir prototype_analysis/classification/eurosat/proto_256/seed_0/logs \
        --name proto_256_finetune_seed_0
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


# Dataset configurations - num_classes for each dataset
DATASET_NUM_CLASSES = {
    "eurosat": 10,
    "oxford_pets": 37,
    "dtd": 47,
    "nctcrche100k": 9,
    "tissue": 10,
    "imagenet1k": 1000,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="DINOv3 Classification Training Wrapper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True,
                       help="Path to config file")
    parser.add_argument("--pretrained_path", type=str, required=True,
                       help="Path to pretrained SSL checkpoint")
    parser.add_argument("--max_epochs", type=int, default=30,
                       help="Max epochs for training")
    parser.add_argument("--batch_size", type=int, default=128,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                       help="Learning rate")
    parser.add_argument("--devices", type=str, default="0",
                       help="Device indices (comma-separated)")
    parser.add_argument("--encoder_type", type=str, default="teacher",
                       choices=["teacher", "student"],
                       help="Which encoder to use from SSL model")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                       help="Directory for checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs",
                       help="Directory for logs")
    parser.add_argument("--name", type=str, default="classification",
                       help="Experiment name")
    parser.add_argument("--logger", type=str, default="csv",
                       choices=["csv", "tensorboard", "both"],
                       help="Logger type")
    parser.add_argument("--freeze_backbone", action="store_true",
                       help="Freeze backbone for linear evaluation")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--precision", type=str, default="bf16-mixed",
                       choices=["32", "16", "bf16-mixed", "16-mixed"],
                       help="Training precision")
    parser.add_argument("--num_classes", type=int, default=None,
                       help="Number of classes (auto-detected from config if not specified)")
    return parser.parse_args()


def detect_dataset_from_config(config_path: str) -> str:
    """Detect dataset name from config path."""
    config_path = config_path.lower()
    for dataset in DATASET_NUM_CLASSES.keys():
        if dataset in config_path:
            return dataset
    return None


def find_checkpoint(checkpoint_path: str) -> str:
    """Find the actual checkpoint file."""
    path = Path(checkpoint_path)

    # If it's a file, return it
    if path.is_file():
        return str(path)

    # If it's a directory, look for checkpoints
    if path.is_dir():
        # Try last.ckpt first
        last_ckpt = path / "last.ckpt"
        if last_ckpt.exists():
            return str(last_ckpt)

        # Look for any .ckpt file
        ckpts = list(path.glob("**/*.ckpt"))
        if ckpts:
            # Return the most recently modified
            return str(sorted(ckpts, key=lambda x: x.stat().st_mtime)[-1])

    return checkpoint_path


def main():
    args = parse_args()

    # Detect number of classes if not specified
    if args.num_classes is None:
        dataset = detect_dataset_from_config(args.config)
        if dataset and dataset in DATASET_NUM_CLASSES:
            args.num_classes = DATASET_NUM_CLASSES[dataset]
            print(f"Auto-detected dataset: {dataset}, num_classes: {args.num_classes}")
        else:
            print("ERROR: Could not auto-detect num_classes. Please specify --num_classes")
            sys.exit(1)

    # Find the actual checkpoint
    pretrained_path = find_checkpoint(args.pretrained_path)
    if not Path(pretrained_path).exists():
        print(f"ERROR: Checkpoint not found: {pretrained_path}")
        sys.exit(1)

    # Determine number of GPUs
    device_list = args.devices.split(",")
    num_gpus = len(device_list)

    # Check if SLURM is handling distribution (srun with --ntasks-per-node)
    slurm_launch = os.environ.get("SLURM_LAUNCH", "")
    slurm_ntasks = int(os.environ.get("SLURM_NTASKS", "1"))

    # Create directories
    output_dir = str(Path(args.checkpoint_dir).parent)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # Build the command for the actual training script
    train_script = "src/training/classification/train.py"

    # Base command - use torchrun for multi-GPU unless SLURM handles distribution
    if slurm_launch and slurm_ntasks > 1:
        # SLURM with srun handles process spawning - don't use torchrun
        cmd = ["python", train_script]
        num_gpus = slurm_ntasks  # Override with SLURM's task count
        print(f"SLURM-based distribution detected: {slurm_ntasks} tasks")
    elif num_gpus > 1:
        cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            f"--master_port={29500 + hash(args.name) % 100}",
            train_script,
        ]
    else:
        cmd = ["python", train_script]

    # Add arguments
    cmd.extend([
        "--config-file", args.config,
        "--checkpoint-path", pretrained_path,
        "--output-dir", output_dir,
        "--num-classes", str(args.num_classes),
        "--max-epochs", str(args.max_epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--gpus", str(num_gpus),
        "--precision", args.precision,
        "--encoder-type", args.encoder_type,
        "--seed", str(args.seed),
    ])

    if num_gpus > 1:
        cmd.extend(["--strategy", "ddp"])

    if args.freeze_backbone:
        cmd.append("--freeze-backbone")

    # Environment
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd()) + ":" + env.get("PYTHONPATH", "")
    env["PL_GLOBAL_SEED"] = str(args.seed)

    print(f"\n{'-'*60}")
    print(f"DINOv3 Classification Training")
    print(f"{'-'*60}")
    print(f"Config: {args.config}")
    print(f"Pretrained: {pretrained_path}")
    print(f"Mode: {'Linear Eval' if args.freeze_backbone else 'Fine-tune'}")
    print(f"Seed: {args.seed}")
    print(f"Devices: {args.devices}")
    print(f"Output: {output_dir}")
    print(f"{'-'*60}\n")

    # Run training
    result = subprocess.run(cmd, env=env)

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
