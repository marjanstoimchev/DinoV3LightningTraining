#!/usr/bin/env python
"""
run_analysis.py

Full prototype analysis pipeline for DINOv3:
1. Pretrain with different prototype counts (single run each)
2. Run classification with multiple seeds for each pretrained model
3. Aggregate results

Output structure:
    prototype_analysis_dinov3/
    ├── pretraining/{dataset}/proto_{N}/
    │   ├── checkpoints/{dataset}/proto_{N}_proto{N}_dinov3/
    │   └── logs/{dataset}/proto_{N}_proto{N}_dinov3/version_0/
    │
    └── classification/{dataset}/proto_{N}/seed_{S}/
        ├── checkpoints/
        ├── logs/
        └── results/{dataset}/single_eval/

    results/prototype_analysis_dinov3/{dataset}/
    ├── classification_results.csv   ← Per-run detailed
    └── classification_stats.csv     ← Aggregated per prototype

Usage:
    python scripts/prototype_analysis/run_analysis.py \
        --dataset dtd \
        --gpus 1,2,3 \
        --batch_size 128

    python scripts/prototype_analysis/run_analysis.py \
        --dataset dtd \
        --prototypes 128 256 512 1024 \
        --seeds 0 1 42 \
        --gpus 0,1,2,3
"""

import argparse
import subprocess
import sys
from pathlib import Path

DATASET_CONFIGS = {
    "eurosat": {"num_classes": 10},
    "oxford_pets": {"num_classes": 37},
    "dtd": {"num_classes": 47},
    "nctcrche100k": {"num_classes": 9},
    "tissue": {"num_classes": 10},
    "imagenet1k": {"num_classes": 1000},
}

DEFAULT_PROTOTYPES = [64, 128, 256, 512, 1024, 2048, 4096]
DEFAULT_SEEDS = [0, 1, 42]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full DINOv3 prototype analysis pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, required=True,
                       choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--prototypes", type=int, nargs="+", default=DEFAULT_PROTOTYPES)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--mode", type=str, choices=["finetune", "lineareval"], default="finetune")

    parser.add_argument("--pretrain_epochs", type=int, default=None)
    parser.add_argument("--classify_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--skip_pretraining", action="store_true")
    parser.add_argument("--skip_classification", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")

    # Output directory
    parser.add_argument("--output_dir", type=str, default="prototype_analysis_dinov3")

    return parser.parse_args()


def run_pretraining_sweep(args) -> bool:
    """Run pretraining sweep script."""
    print(f"\n{'='*70}")
    print(f"PHASE 1: PRETRAINING SWEEP")
    print(f"{'='*70}")

    cmd = [
        "python", "scripts/prototype_analysis/sweep_pretraining.py",
        "--dataset", args.dataset,
        "--prototypes", *[str(p) for p in args.prototypes],
        "--gpus", args.gpus,
        "--output_dir", args.output_dir,
        "--precision", args.precision,
    ]

    if args.pretrain_epochs:
        cmd.extend(["--max_epochs", str(args.pretrain_epochs)])
    if args.batch_size:
        cmd.extend(["--batch_size", str(args.batch_size)])
    if args.skip_existing:
        cmd.append("--skip_existing")
    if args.compile:
        cmd.append("--compile")

    print(f"Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def run_classification_sweep(args) -> bool:
    """Run classification sweep script."""
    print(f"\n{'='*70}")
    print(f"PHASE 2: CLASSIFICATION SWEEP")
    print(f"{'='*70}")

    pretraining_dir = f"{args.output_dir}/pretraining/{args.dataset}"

    cmd = [
        "python", "scripts/prototype_analysis/sweep_classification.py",
        "--dataset", args.dataset,
        "--pretraining_dir", pretraining_dir,
        "--prototypes", *[str(p) for p in args.prototypes],
        "--seeds", *[str(s) for s in args.seeds],
        "--mode", args.mode,
        "--gpus", args.gpus,
        "--output_dir", args.output_dir,
        "--precision", args.precision,
    ]

    if args.classify_epochs:
        cmd.extend(["--max_epochs", str(args.classify_epochs)])
    if args.batch_size:
        cmd.extend(["--batch_size", str(args.batch_size)])
    if args.skip_existing:
        cmd.append("--skip_existing")

    print(f"Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    args = parse_args()

    print(f"\n{'='*70}")
    print(f"DINOV3 PROTOTYPE ANALYSIS PIPELINE - {args.dataset}")
    print(f"{'='*70}")
    print(f"Prototypes:     {args.prototypes}")
    print(f"Seeds:          {args.seeds}")
    print(f"Mode:           {args.mode}")
    print(f"GPUs:           {args.gpus}")
    print(f"Output:         {args.output_dir}")
    print(f"{'='*70}")

    # Phase 1: Pretraining
    if not args.skip_pretraining:
        success = run_pretraining_sweep(args)
        if not success:
            print("WARNING: Pretraining sweep had failures")
    else:
        print("\nSkipping pretraining (--skip_pretraining)")

    # Phase 2: Classification
    if not args.skip_classification:
        success = run_classification_sweep(args)
        if not success:
            print("WARNING: Classification sweep had failures")
    else:
        print("\nSkipping classification (--skip_classification)")

    # Summary
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"Checkpoints: {args.output_dir}/")
    output_dir_name = Path(args.output_dir).name
    print(f"Results:     results/{output_dir_name}/{args.dataset}/")
    print(f"  - classification_results.csv (per-run)")
    print(f"  - classification_stats.csv (aggregated)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
