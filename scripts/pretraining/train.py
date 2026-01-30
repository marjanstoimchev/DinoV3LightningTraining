#!/usr/bin/env python
"""
Wrapper script for DINOv3 SSL pretraining.
Provides a CLI interface compatible with the prototype analysis pipeline.

This script wraps src/training/ssl/train.py with a simplified interface.

Usage:
    python scripts/pretraining/train.py \
        --config configs/eurosat/config_ssl_pretraining.yaml \
        --num_prototypes 256 \
        --devices 0,1 \
        --name proto_256 \
        --log_base_dir prototype_analysis/pretraining/eurosat/proto_256/logs \
        --checkpoint_base_dir prototype_analysis/pretraining/eurosat/proto_256/checkpoints
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="DINOv3 SSL Pretraining Wrapper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True,
                       help="Path to config file")
    parser.add_argument("--num_prototypes", type=int, default=4096,
                       help="Number of prototypes for DINO/iBOT heads")
    parser.add_argument("--koleo_weight", type=float, default=0.1,
                       help="KoLeo loss weight (not used in DINOv3, kept for compatibility)")
    parser.add_argument("--devices", type=str, default="0",
                       help="Device indices (comma-separated)")
    parser.add_argument("--name", type=str, default="ssl_pretrain",
                       help="Experiment name")
    parser.add_argument("--log_base_dir", type=str, default="./logs",
                       help="Base directory for logs")
    parser.add_argument("--checkpoint_base_dir", type=str, default="./checkpoints",
                       help="Base directory for checkpoints")
    parser.add_argument("--max_epochs", type=int, default=None,
                       help="Max epochs (uses config default if not specified)")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size (uses config default if not specified)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--precision", type=str, default="bf16-mixed",
                       choices=["32", "16", "bf16-mixed", "16-mixed"],
                       help="Training precision")
    parser.add_argument("--compile", action="store_true",
                       help="Enable PyTorch 2.0 compilation")
    parser.add_argument("--sampler_type", type=str, default="distributed",
                       choices=["infinite", "distributed", "sharded_infinite", "epoch"],
                       help="Sampler type for data loading")
    parser.add_argument("--pretrain_checkpoint", type=str, default="",
                       help="Path to DINOv3 checkpoint for continued pretraining (empty for from-scratch)")
    parser.add_argument("--enable_gram", action="store_true",
                       help="Enable GRAM loss (default: disabled)")
    parser.add_argument("--gram_weight", type=float, default=None,
                       help="GRAM loss weight (default: 1.0 when enabled)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine number of GPUs
    device_list = args.devices.split(",")
    num_gpus = len(device_list)

    # Check if SLURM is handling distribution (srun with --ntasks-per-node)
    slurm_launch = os.environ.get("SLURM_LAUNCH", "")
    slurm_ntasks = int(os.environ.get("SLURM_NTASKS", "1"))

    # Create output directory (parent of checkpoint_base_dir)
    output_dir = str(Path(args.checkpoint_base_dir).parent)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_base_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_base_dir).mkdir(parents=True, exist_ok=True)

    # Build the command for the actual training script
    train_script = "src/training/ssl/train.py"

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
            train_script,
        ]
    else:
        cmd = ["python", train_script]

    # Add arguments
    cmd.extend([
        "--config-file", args.config,
        "--output-dir", output_dir,
        "--seed", str(args.seed),
        "--num-prototypes", str(args.num_prototypes),
        "--gpus", str(num_gpus),
        "--precision", args.precision,
        "--sampler-type", args.sampler_type,
    ])

    if num_gpus > 1:
        cmd.extend(["--strategy", "ddp"])

    if args.max_epochs is not None:
        cmd.extend(["--max-epochs", str(args.max_epochs)])

    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(args.batch_size)])

    if args.compile:
        cmd.append("--compile")

    if args.pretrain_checkpoint:
        cmd.extend(["--checkpoint-path", args.pretrain_checkpoint])

    if args.enable_gram:
        cmd.append("--enable-gram")

    if args.gram_weight is not None:
        cmd.extend(["--gram-weight", str(args.gram_weight)])

    # Environment
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd()) + ":" + env.get("PYTHONPATH", "")

    print(f"\n{'='*70}")
    print(f"DINOv3 SSL Pretraining")
    print(f"{'='*70}")
    print(f"Config: {args.config}")
    print(f"Prototypes: {args.num_prototypes}")
    print(f"Devices: {args.devices}")
    print(f"Output: {output_dir}")
    if args.pretrain_checkpoint:
        print(f"Checkpoint: {args.pretrain_checkpoint} (continued pretraining)")
    else:
        print(f"Checkpoint: None (from scratch)")
    if args.enable_gram:
        gram_w = args.gram_weight if args.gram_weight is not None else 1.0
        print(f"GRAM loss: enabled (weight={gram_w})")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    # Run training
    result = subprocess.run(cmd, env=env)

    # Move checkpoints to expected location if needed
    # The actual training saves to output_dir/checkpoints/pretraining/
    actual_ckpt_dir = Path(output_dir) / "checkpoints" / "pretraining"
    target_ckpt_dir = Path(args.checkpoint_base_dir)

    if actual_ckpt_dir.exists() and actual_ckpt_dir != target_ckpt_dir:
        # Copy last.ckpt to target directory
        last_ckpt = actual_ckpt_dir / "last.ckpt"
        if last_ckpt.exists():
            import shutil
            target_last = target_ckpt_dir / "last.ckpt"
            if not target_last.exists():
                shutil.copy2(last_ckpt, target_last)
                print(f"Copied checkpoint to: {target_last}")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
