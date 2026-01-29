#!/usr/bin/env python
"""
sweep_pretraining.py

Sweep over different numbers of prototypes for DINOv3 SSL pretraining.
Each prototype count runs once (no seed variation at pretraining stage).

Output structure:
    prototype_analysis_dinov3/
    └── pretraining/
        └── {dataset}/
            └── proto_{N}/
                ├── checkpoints/{dataset}/proto_{N}_proto{N}_dinov3/
                │   └── last.ckpt
                └── logs/{dataset}/proto_{N}_proto{N}_dinov3/version_0/
                    ├── hparams.yaml
                    └── metrics.csv

Usage:
    python scripts/prototype_analysis/sweep_pretraining.py \
        --dataset dtd \
        --gpus 0,1,2

    python scripts/prototype_analysis/sweep_pretraining.py \
        --dataset dtd \
        --prototypes 64 128 256 512 1024 2048 4096 \
        --gpus 0,1,2,3
"""

import argparse
import subprocess
import sys
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import time

# Dataset configurations
DATASET_CONFIGS = {
    "eurosat": {
        "config": "configs/eurosat/config_ssl_pretraining.yaml",
        "dataset_path": "HuggingFace:name=blanchon/EuroSAT_RGB:split=train",
    },
    "oxford_pets": {
        "config": "configs/oxford_pets/config_ssl_pretraining.yaml",
        "dataset_path": "HuggingFace:name=timm/oxford-iiit-pet",
    },
    "dtd": {
        "config": "configs/DTD/config_ssl_pretraining.yaml",
        "dataset_path": "HuggingFace:name=cansa/Describable-Textures-Dataset-DTD",
    },
    "nctcrche100k": {
        "config": "configs/NCTCRCHE100K/config_ssl_pretraining.yaml",
        "dataset_path": "HuggingFace:name=DykeF/NCTCRCHE100K",
    },
    "tissue": {
        "config": "configs/tissue/config_ssl_pretraining.yaml",
        "dataset_path": "CustomTIFF:root=../Datasets/tissue/",
    },
    "imagenet1k": {
        "config": "configs/imagenet1k/config_ssl_pretraining.yaml",
        "dataset_path": "HuggingFace:name=ILSVRC/imagenet-1k",
    },
}

DEFAULT_PROTOTYPES = [64, 128, 256, 512, 1024, 2048, 4096]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep DINOv3 pretraining over different prototype counts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, required=True,
                       choices=list(DATASET_CONFIGS.keys()),
                       help="Dataset name")
    parser.add_argument("--prototypes", type=int, nargs="+",
                       default=DEFAULT_PROTOTYPES,
                       help="Prototype counts to sweep")

    parser.add_argument("--max_epochs", type=int, default=None,
                       help="Max epochs (uses config default if not specified)")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size (uses config default if not specified)")
    parser.add_argument("--gpus", type=str, default="0",
                       help="GPU indices (comma-separated)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for pretraining")
    parser.add_argument("--precision", type=str, default="bf16-mixed",
                       choices=["32", "16", "bf16-mixed", "16-mixed"],
                       help="Training precision")
    parser.add_argument("--compile", action="store_true",
                       help="Enable PyTorch 2.0 compilation")

    # Output directory
    parser.add_argument("--output_dir", type=str, default="prototype_analysis_dinov3",
                       help="Base output directory (default: prototype_analysis_dinov3)")

    parser.add_argument("--skip_existing", action="store_true",
                       help="Skip prototype counts that already have checkpoints")

    return parser.parse_args()


def run_pretraining(
    dataset: str,
    config: str,
    num_prototypes: int,
    output_base: str,
    gpus: str,
    seed: int,
    precision: str,
    compile_model: bool,
    max_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Run pretraining for a single prototype count."""

    # Experiment name: proto_{N}_proto{N}_dinov3
    exp_name = f"proto_{num_prototypes}_proto{num_prototypes}_dinov3"

    # Output directories
    proto_dir = f"{output_base}/pretraining/{dataset}/proto_{num_prototypes}"
    checkpoint_dir = f"{proto_dir}/checkpoints/{dataset}/{exp_name}"
    log_dir = f"{proto_dir}/logs/{dataset}/{exp_name}"

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Compute device indices
    gpu_list = gpus.split(",")
    num_gpus = len(gpu_list)
    devices = "0" if num_gpus == 1 else ",".join(str(i) for i in range(num_gpus))

    # Build command
    cmd = [
        "python", "scripts/pretraining/train.py",
        "--config", config,
        "--num_prototypes", str(num_prototypes),
        "--devices", devices,
        "--name", exp_name,
        "--log_base_dir", f"{proto_dir}/logs/{dataset}",
        "--checkpoint_base_dir", f"{proto_dir}/checkpoints/{dataset}",
        "--seed", str(seed),
        "--precision", precision,
    ]

    if max_epochs is not None:
        cmd.extend(["--max_epochs", str(max_epochs)])
    if batch_size is not None:
        cmd.extend(["--batch_size", str(batch_size)])
    if compile_model:
        cmd.append("--compile")

    env = {
        "CUDA_VISIBLE_DEVICES": gpus,
        "PYTHONPATH": str(Path.cwd()),
    }

    print(f"\n{'='*70}")
    print(f"Pretraining: {dataset} | Prototypes: {num_prototypes}")
    print(f"{'='*70}")
    print(f"Output: {proto_dir}")
    print(f"{'='*70}\n")

    start_time = time.time()

    result = subprocess.run(
        cmd,
        env={**os.environ, **env},
        capture_output=False,
    )

    elapsed = time.time() - start_time
    checkpoint_path = find_checkpoint(checkpoint_dir)

    run_result = {
        "num_prototypes": num_prototypes,
        "dataset": dataset,
        "success": result.returncode == 0 and checkpoint_path is not None,
        "checkpoint_path": checkpoint_path,
        "elapsed_seconds": elapsed,
        "output_dir": proto_dir,
    }

    if checkpoint_path:
        print(f"SUCCESS: Checkpoint saved at {checkpoint_path}")
    else:
        print(f"WARNING: Pretraining failed for {num_prototypes} prototypes")

    return run_result


def find_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the last.ckpt or best checkpoint in directory."""
    ckpt_path = Path(checkpoint_dir)

    for pattern in ["**/last.ckpt", "**/*.ckpt"]:
        ckpts = list(ckpt_path.glob(pattern))
        if ckpts:
            return str(sorted(ckpts, key=lambda x: x.stat().st_mtime)[-1])

    return None


def main():
    args = parse_args()

    dataset_cfg = DATASET_CONFIGS[args.dataset]
    config = dataset_cfg["config"]

    if not Path(config).exists():
        print(f"ERROR: Config not found: {config}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"DINOv3 Prototype Pretraining Sweep - {args.dataset}")
    print(f"{'='*70}")
    print(f"Prototype counts: {args.prototypes}")
    print(f"Output directory: {args.output_dir}")
    print(f"GPUs: {args.gpus}")
    print(f"{'='*70}\n")

    results = []
    for num_prototypes in args.prototypes:
        if args.skip_existing:
            exp_name = f"proto_{num_prototypes}_proto{num_prototypes}_dinov3"
            proto_dir = f"{args.output_dir}/pretraining/{args.dataset}/proto_{num_prototypes}"
            ckpt_dir = f"{proto_dir}/checkpoints/{args.dataset}/{exp_name}"
            existing_ckpt = find_checkpoint(ckpt_dir)
            if existing_ckpt:
                print(f"Skipping proto_{num_prototypes} (exists: {existing_ckpt})")
                results.append({
                    "num_prototypes": num_prototypes,
                    "dataset": args.dataset,
                    "success": True,
                    "checkpoint_path": existing_ckpt,
                    "skipped": True,
                    "output_dir": proto_dir,
                })
                continue

        result = run_pretraining(
            dataset=args.dataset,
            config=config,
            num_prototypes=num_prototypes,
            output_base=args.output_dir,
            gpus=args.gpus,
            seed=args.seed,
            precision=args.precision,
            compile_model=args.compile,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
        )
        results.append(result)

    # Print summary
    successful = [r for r in results if r.get("success")]
    print(f"\n{'='*70}")
    print(f"PRETRAINING SWEEP COMPLETE - {args.dataset}")
    print(f"{'='*70}")
    print(f"Successful: {len(successful)}/{len(results)}")

    if successful:
        print("\nCheckpoints ready for classification:")
        for r in successful:
            print(f"  proto_{r['num_prototypes']}: {r.get('checkpoint_path', 'N/A')}")

    print(f"\nNext step: Run classification sweep")
    print(f"  python scripts/prototype_analysis/sweep_classification.py \\")
    print(f"      --dataset {args.dataset} \\")
    print(f"      --pretraining_dir {args.output_dir}/pretraining/{args.dataset} \\")
    print(f"      --gpus {args.gpus}")


if __name__ == "__main__":
    main()
