#!/usr/bin/env python
"""
sweep_classification.py

Run classification with multiple seeds for each pretrained DINOv3 checkpoint.

Output structure:
    prototype_analysis_dinov3/
    └── classification/
        └── {dataset}/
            └── proto_{N}/
                └── seed_{S}/
                    ├── checkpoints/
                    │   ├── cls-epoch=XX-val_loss=X.XXXX.ckpt
                    │   └── last.ckpt
                    ├── logs/proto_{N}_finetune_seed_{S}/version_0/
                    │   ├── hparams.yaml
                    │   └── metrics.csv
                    └── results/{dataset}/single_eval/predictions/checkpoints/
                        ├── predictions.npy
                        ├── scores.npy
                        ├── labels.npy
                        ├── confusion_matrix.csv
                        └── per_class_metrics.csv

    results/prototype_analysis_dinov3/{dataset}/
    ├── classification_results.csv   ← Per-run detailed results
    └── classification_stats.csv     ← Aggregated stats per prototype

Usage:
    python scripts/prototype_analysis/sweep_classification.py \
        --dataset dtd \
        --pretraining_dir prototype_analysis_dinov3/pretraining/dtd \
        --seeds 0 1 42 \
        --gpus 0,1,2
"""

import argparse
import subprocess
import sys
import json
import csv
import re
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import time
import numpy as np

DATASET_CONFIGS = {
    "eurosat": {"config": "configs/eurosat/config_classification.yaml", "num_classes": 10},
    "oxford_pets": {"config": "configs/oxford_pets/config_classification.yaml", "num_classes": 37},
    "dtd": {"config": "configs/DTD/config_classification.yaml", "num_classes": 47},
    "nctcrche100k": {"config": "configs/NCTCRCHE100K/config_classification.yaml", "num_classes": 9},
    "tissue": {"config": "configs/tissue/config_classification.yaml", "num_classes": 10},
    "imagenet1k": {"config": "configs/imagenet1k/config_classification.yaml", "num_classes": 1000},
}

DEFAULT_SEEDS = [0, 1, 42]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run classification sweep for pretrained DINOv3 checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, required=True,
                       choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--pretraining_dir", type=str, required=True,
                       help="Directory containing pretrained checkpoints")
    parser.add_argument("--prototypes", type=int, nargs="+", default=None,
                       help="Specific prototype counts (default: all found)")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--mode", type=str, choices=["finetune", "lineareval"], default="finetune")

    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--encoder_type", type=str, default="teacher", choices=["teacher", "student"])
    parser.add_argument("--precision", type=str, default="bf16-mixed")

    parser.add_argument("--output_dir", type=str, default="prototype_analysis_dinov3",
                       help="Base output directory (default: prototype_analysis_dinov3)")
    parser.add_argument("--skip_existing", action="store_true")

    return parser.parse_args()


def find_pretrained_checkpoints(pretraining_dir: str) -> Dict[int, str]:
    """Find all pretrained checkpoints organized by prototype count."""
    checkpoints = {}
    pretraining_path = Path(pretraining_dir)

    if not pretraining_path.exists():
        return checkpoints

    for proto_dir in pretraining_path.glob("proto_*"):
        if not proto_dir.is_dir():
            continue

        match = re.match(r"proto_(\d+)", proto_dir.name)
        if not match:
            continue

        num_prototypes = int(match.group(1))
        checkpoint = find_checkpoint(str(proto_dir / "checkpoints"))

        if checkpoint:
            checkpoints[num_prototypes] = checkpoint
            print(f"Found: proto_{num_prototypes} -> {checkpoint}")

    return checkpoints


def find_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find checkpoint in directory."""
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        return None

    for pattern in ["**/last.ckpt", "**/*.ckpt"]:
        ckpts = list(ckpt_path.glob(pattern))
        if ckpts:
            return str(sorted(ckpts, key=lambda x: x.stat().st_mtime)[-1])
    return None


def run_single_classification(
    dataset: str,
    config: str,
    pretrained_path: str,
    num_prototypes: int,
    seed: int,
    mode: str,
    output_base: str,
    max_epochs: int,
    batch_size: int,
    learning_rate: float,
    gpus: str,
    encoder_type: str,
    freeze_backbone: bool,
    precision: str,
) -> Dict[str, Any]:
    """Run classification for a single (prototype, seed) combination."""

    # Experiment name
    exp_name = f"proto_{num_prototypes}_{mode}_seed_{seed}"

    # Output directories
    run_dir = f"{output_base}/classification/{dataset}/proto_{num_prototypes}/seed_{seed}"
    checkpoint_dir = f"{run_dir}/checkpoints"
    log_dir = f"{run_dir}/logs"

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    gpu_list = gpus.split(",")
    num_gpus = len(gpu_list)
    devices = "0" if num_gpus == 1 else ",".join(str(i) for i in range(num_gpus))

    cmd = [
        "python", "scripts/classification/train.py",
        "--config", config,
        "--pretrained_path", pretrained_path,
        "--max_epochs", str(max_epochs),
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--devices", devices,
        "--encoder_type", encoder_type,
        "--checkpoint_dir", checkpoint_dir,
        "--log_dir", log_dir,
        "--name", exp_name,
        "--logger", "csv",
        "--seed", str(seed),
        "--precision", precision,
    ]

    if freeze_backbone:
        cmd.append("--freeze_backbone")

    env = {
        "CUDA_VISIBLE_DEVICES": gpus,
        "PYTHONPATH": str(Path.cwd()),
        "PL_GLOBAL_SEED": str(seed),
    }

    print(f"\n{'-'*60}")
    print(f"Proto: {num_prototypes} | Seed: {seed} | Mode: {mode}")
    print(f"{'-'*60}")

    start_time = time.time()

    result = subprocess.run(
        cmd,
        env={**os.environ, **env},
        capture_output=False,
    )

    elapsed = time.time() - start_time
    metrics = parse_metrics_from_log(log_dir)

    run_result = {
        "num_prototypes": num_prototypes,
        "seed": seed,
        "mode": mode,
        "success": result.returncode == 0,
        "val_acc": metrics.get("val_acc"),
        "val_loss": metrics.get("val_loss"),
        "pretrained_path": pretrained_path,
        "checkpoint_dir": checkpoint_dir,
        "log_dir": log_dir,
        "elapsed_seconds": elapsed,
    }

    if metrics.get("val_acc") is not None:
        print(f"Result: val_acc = {metrics['val_acc']*100:.2f}%")

    return run_result


def parse_metrics_from_log(log_dir: str) -> Dict[str, Any]:
    """Parse metrics from CSV training log."""
    metrics_file = Path(log_dir) / "metrics.csv"

    if not metrics_file.exists():
        for f in Path(log_dir).rglob("metrics.csv"):
            metrics_file = f
            break

    if not metrics_file.exists():
        return {"val_acc": None, "val_loss": None}

    best_acc = 0.0
    best_loss = float("inf")

    try:
        with open(metrics_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key in ["val_acc", "val_accuracy", "val_acc_epoch"]:
                    if key in row and row[key]:
                        try:
                            acc = float(row[key])
                            if acc > best_acc:
                                best_acc = acc
                        except ValueError:
                            pass

                for key in ["val_loss", "val_loss_epoch"]:
                    if key in row and row[key]:
                        try:
                            loss = float(row[key])
                            if loss < best_loss:
                                best_loss = loss
                        except ValueError:
                            pass
    except Exception as e:
        print(f"Warning: Could not parse metrics: {e}")

    return {
        "val_acc": best_acc if best_acc > 0 else None,
        "val_loss": best_loss if best_loss < float("inf") else None,
    }


def save_aggregated_results(
    results: List[Dict[str, Any]],
    dataset: str,
    output_dir: str = "prototype_analysis_dinov3",
) -> None:
    """Save aggregated results to results/{output_dir_basename}/{dataset}/"""

    # Results directory
    output_dir_name = Path(output_dir).name  # e.g., "prototype_analysis_dinov3"
    results_dir = Path("results") / output_dir_name / dataset
    results_dir.mkdir(parents=True, exist_ok=True)

    # Filter successful runs with metrics
    valid_results = [r for r in results if r.get("success") and r.get("val_acc") is not None]

    if not valid_results:
        print("No valid results to save")
        return

    # === classification_results.csv (detailed per-run) ===
    # Columns: prototype,seed,accuracy,top5_accuracy,f1_macro,f1_weighted,precision_macro,recall_macro,auc_roc_macro,num_samples
    results_file = results_dir / "classification_results.csv"
    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "prototype", "seed", "accuracy", "top5_accuracy", "f1_macro", "f1_weighted",
            "precision_macro", "recall_macro", "auc_roc_macro", "num_samples"
        ])
        for r in sorted(valid_results, key=lambda x: (x["num_prototypes"], x["seed"])):
            writer.writerow([
                r["num_prototypes"],
                r["seed"],
                r["val_acc"],
                r.get("top5_acc", ""),
                r.get("f1_macro", ""),
                r.get("f1_weighted", ""),
                r.get("precision_macro", ""),
                r.get("recall_macro", ""),
                r.get("auc_roc_macro", ""),
                r.get("num_samples", ""),
            ])

    print(f"Saved: {results_file}")

    # === classification_stats.csv (aggregated per prototype) ===
    # Group by prototype
    by_prototype = {}
    for r in valid_results:
        num_proto = r["num_prototypes"]
        if num_proto not in by_prototype:
            by_prototype[num_proto] = []
        by_prototype[num_proto].append(r)

    stats_file = results_dir / "classification_stats.csv"
    with open(stats_file, "w", newline="") as f:
        writer = csv.writer(f)
        # Header
        writer.writerow([
            "prototype",
            "accuracy_mean", "accuracy_std", "accuracy_count",
            "top5_accuracy_mean", "top5_accuracy_std", "top5_accuracy_count",
            "f1_macro_mean", "f1_macro_std", "f1_macro_count",
            "f1_weighted_mean", "f1_weighted_std", "f1_weighted_count",
            "precision_macro_mean", "precision_macro_std", "precision_macro_count",
            "recall_macro_mean", "recall_macro_std", "recall_macro_count",
            "auc_roc_macro_mean", "auc_roc_macro_std", "auc_roc_macro_count",
        ])

        for num_proto in sorted(by_prototype.keys()):
            runs = by_prototype[num_proto]
            accs = [r["val_acc"] for r in runs if r.get("val_acc") is not None]

            if accs:
                acc_mean = np.mean(accs)
                acc_std = np.std(accs) if len(accs) > 1 else 0.0
                acc_count = len(accs)
            else:
                acc_mean, acc_std, acc_count = 0, 0, 0

            # For now, we only have accuracy from training logs
            # Other metrics would come from evaluation
            writer.writerow([
                num_proto,
                acc_mean, acc_std, acc_count,
                "", "", "",  # top5
                "", "", "",  # f1_macro
                "", "", "",  # f1_weighted
                "", "", "",  # precision_macro
                "", "", "",  # recall_macro
                "", "", "",  # auc_roc_macro
            ])

    print(f"Saved: {stats_file}")


def main():
    args = parse_args()

    dataset_cfg = DATASET_CONFIGS[args.dataset]
    config = dataset_cfg["config"]

    if args.max_epochs is None:
        args.max_epochs = 30 if args.mode == "finetune" else 100
    if args.learning_rate is None:
        args.learning_rate = 0.0001 if args.mode == "finetune" else 0.001

    freeze_backbone = args.mode == "lineareval"

    print(f"\nSearching for pretrained checkpoints in: {args.pretraining_dir}")
    checkpoints = find_pretrained_checkpoints(args.pretraining_dir)

    if not checkpoints:
        print(f"ERROR: No pretrained checkpoints found in {args.pretraining_dir}")
        sys.exit(1)

    if args.prototypes:
        checkpoints = {k: v for k, v in checkpoints.items() if k in args.prototypes}

    print(f"\n{'='*70}")
    print(f"DINOv3 Classification Sweep - {args.dataset}")
    print(f"{'='*70}")
    print(f"Mode:             {args.mode}")
    print(f"Prototype counts: {sorted(checkpoints.keys())}")
    print(f"Seeds:            {args.seeds}")
    print(f"Total runs:       {len(checkpoints) * len(args.seeds)}")
    print(f"Output:           {args.output_dir}")
    print(f"{'='*70}")

    results = []
    for num_prototypes in sorted(checkpoints.keys()):
        pretrained_path = checkpoints[num_prototypes]

        for seed in args.seeds:
            if args.skip_existing:
                run_dir = f"{args.output_dir}/classification/{args.dataset}/proto_{num_prototypes}/seed_{seed}"
                existing_ckpt = find_checkpoint(f"{run_dir}/checkpoints")
                if existing_ckpt:
                    print(f"Skipping proto_{num_prototypes}/seed_{seed} (exists)")
                    metrics = parse_metrics_from_log(f"{run_dir}/logs")
                    results.append({
                        "num_prototypes": num_prototypes,
                        "seed": seed,
                        "mode": args.mode,
                        "success": True,
                        "skipped": True,
                        "val_acc": metrics.get("val_acc"),
                        "val_loss": metrics.get("val_loss"),
                        "checkpoint_dir": f"{run_dir}/checkpoints",
                    })
                    continue

            result = run_single_classification(
                dataset=args.dataset,
                config=config,
                pretrained_path=pretrained_path,
                num_prototypes=num_prototypes,
                seed=seed,
                mode=args.mode,
                output_base=args.output_dir,
                max_epochs=args.max_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                gpus=args.gpus,
                encoder_type=args.encoder_type,
                freeze_backbone=freeze_backbone,
                precision=args.precision,
            )
            results.append(result)

    # Save aggregated results
    save_aggregated_results(results, args.dataset, args.output_dir)

    # Print summary
    print(f"\n{'='*70}")
    print(f"CLASSIFICATION RESULTS - {args.dataset} ({args.mode})")
    print(f"{'='*70}")

    by_prototype = {}
    for r in results:
        if r.get("val_acc") is not None:
            num_proto = r["num_prototypes"]
            if num_proto not in by_prototype:
                by_prototype[num_proto] = []
            by_prototype[num_proto].append(r["val_acc"])

    print(f"{'Prototypes':<12} {'Seeds':<8} {'Accuracy (mean +/- std)':<25}")
    print(f"{'-'*50}")

    for num_proto in sorted(by_prototype.keys()):
        accs = by_prototype[num_proto]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs) if len(accs) > 1 else 0.0
        print(f"{num_proto:<12} {len(accs):<8} {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%")

    print(f"{'='*70}")
    output_dir_name = Path(args.output_dir).name
    print(f"\nResults saved to: results/{output_dir_name}/{args.dataset}/")


if __name__ == "__main__":
    main()
