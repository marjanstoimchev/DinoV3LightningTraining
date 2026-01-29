#!/usr/bin/env python
"""
classification/eval.py

Evaluate DINOv3 classification models on the test set.
Supports multiple seeds, best/last checkpoint selection, and aggregated results.

Output structure:
  results/{dataset}/
    {sweep_name}.csv                                    # All individual results
    {sweep_name}_summary.csv                            # Aggregated by experiment (mean +/- std)
    {sweep_name}.json                                   # Full details
    {sweep_name}/predictions/
      {exp_name}/
        labels.npy                                      # Ground truth labels
        predictions.npy                                 # Predicted labels
        scores.npy                                      # Prediction probabilities [N, num_classes]
        confusion_matrix.csv                            # Confusion matrix with class names
        per_class_metrics.csv                           # Per-class precision, recall, F1, AUC

Usage examples:

  # Evaluate all checkpoints in a sweep (using last.ckpt)
  python scripts/classification/eval.py \
      --config configs/eurosat/config_classification.yaml \
      --sweep_dir prototype_analysis_dinov3/classification/eurosat \
      --checkpoint_type last

  # Evaluate using best checkpoint (lowest val_loss)
  python scripts/classification/eval.py \
      --config configs/eurosat/config_classification.yaml \
      --sweep_dir prototype_analysis_dinov3/classification/eurosat \
      --checkpoint_type best

  # Evaluate a single checkpoint file or directory
  python scripts/classification/eval.py \
      --config configs/eurosat/config_classification.yaml \
      --checkpoint prototype_analysis_dinov3/classification/eurosat/proto_256/seed_0/checkpoints \
      --checkpoint_type best
"""

import argparse
import sys
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add repo root to import path
ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / 'dinov3'))

# Metrics imports
try:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        roc_auc_score,
        top_k_accuracy_score,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")


# Dataset configurations
DATASET_CONFIGS = {
    "eurosat": {"num_classes": 10},
    "oxford_pets": {"num_classes": 37},
    "dtd": {"num_classes": 47},
    "nctcrche100k": {"num_classes": 9},
    "tissue": {"num_classes": 10},
    "imagenet1k": {"num_classes": 1000},
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate DINOv3 classification models on test set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, required=True, help="Path to config.yaml")

    # Checkpoint selection
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to single checkpoint file or directory to evaluate")
    p.add_argument("--sweep_dir", type=str, default=None,
                   help="Sweep directory containing proto_N/seed_S subdirs")
    p.add_argument("--checkpoint_type", type=str, default="last", choices=["last", "best"],
                   help="Which checkpoint to use: 'last' or 'best' (lowest val_loss)")

    # Output
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory for results (default: derived from sweep_dir or checkpoint)")

    # Runtime settings
    p.add_argument("--batch_size", type=int, default=128, help="Batch size for evaluation")
    p.add_argument("--devices", type=str, default="0", help="GPU device to use")
    p.add_argument("--num_workers", type=int, default=8, help="Number of data loading workers")
    p.add_argument("--num_classes", type=int, default=None, help="Number of classes (auto-detected if not specified)")

    return p.parse_args()


def detect_dataset_from_config(config_path: str) -> str:
    """Detect dataset name from config path."""
    config_path = config_path.lower()
    for dataset in DATASET_CONFIGS.keys():
        if dataset in config_path:
            return dataset
    return "unknown"


def parse_experiment_name(exp_name: str) -> Dict[str, Any]:
    """
    Parse experiment folder name to extract components.

    Expected formats:
        - proto_256/seed_42 -> {param: "proto_256", seed: 42}
        - proto_256_finetune_seed_42 -> {param: "proto_256", mode: "finetune", seed: 42}
    """
    info = {"exp_name": exp_name, "param": exp_name, "mode": "finetune", "seed": 0, "num_prototypes": 0}

    # Try to extract seed
    seed_match = re.search(r'seed[_=](\d+)', exp_name)
    if seed_match:
        info["seed"] = int(seed_match.group(1))

    # Try to extract prototype count
    proto_match = re.search(r'proto[_=](\d+)', exp_name)
    if proto_match:
        info["num_prototypes"] = int(proto_match.group(1))
        info["param"] = f"proto_{info['num_prototypes']}"

    # Try to extract mode
    if "lineareval" in exp_name.lower() or "linear" in exp_name.lower():
        info["mode"] = "lineareval"
    elif "finetune" in exp_name.lower():
        info["mode"] = "finetune"

    return info


def find_checkpoint(exp_dir: Path, checkpoint_type: str = "last") -> Optional[str]:
    """
    Find checkpoint in experiment directory.

    Args:
        exp_dir: Experiment directory path
        checkpoint_type: "last" for last.ckpt, "best" for lowest val_loss

    Returns:
        Path to checkpoint or None
    """
    if exp_dir.is_file():
        return str(exp_dir)

    if checkpoint_type == "last":
        # Look for last.ckpt
        last_ckpt = exp_dir / "last.ckpt"
        if last_ckpt.exists():
            return str(last_ckpt)
        # Fallback to any .ckpt
        ckpts = list(exp_dir.glob("*.ckpt"))
        if ckpts:
            return str(sorted(ckpts, key=lambda x: x.stat().st_mtime)[-1])

    elif checkpoint_type == "best":
        # Find checkpoint with lowest val_loss in filename
        ckpts = list(exp_dir.glob("*.ckpt"))
        if not ckpts:
            ckpts = list(exp_dir.glob("**/*.ckpt"))

        best_ckpt = None
        best_loss = float('inf')

        for ckpt in ckpts:
            if ckpt.name == "last.ckpt":
                continue
            # Extract val_loss from filename
            match = re.search(r'loss[=_](\d+\.?\d*)', ckpt.name)
            if match:
                loss = float(match.group(1))
                if loss < best_loss:
                    best_loss = loss
                    best_ckpt = ckpt

        if best_ckpt:
            return str(best_ckpt)

        # Fallback to last.ckpt
        last_ckpt = exp_dir / "last.ckpt"
        if last_ckpt.exists():
            return str(last_ckpt)

    return None


def find_all_experiments(sweep_dir: str, checkpoint_type: str = "last") -> List[Dict[str, Any]]:
    """
    Find all experiments in a sweep directory.

    Expected structure:
        sweep_dir/proto_N/seed_S/checkpoints/

    Returns:
        List of dicts with experiment info and checkpoint path
    """
    base_path = Path(sweep_dir)
    experiments = []

    if not base_path.exists():
        print(f"Warning: Sweep directory {sweep_dir} does not exist")
        return experiments

    # Look for proto_N/seed_S structure
    for proto_dir in sorted(base_path.glob("proto_*")):
        if not proto_dir.is_dir():
            continue

        proto_match = re.match(r"proto_(\d+)", proto_dir.name)
        if not proto_match:
            continue

        num_prototypes = int(proto_match.group(1))

        for seed_dir in sorted(proto_dir.glob("seed_*")):
            if not seed_dir.is_dir():
                continue

            seed_match = re.match(r"seed_(\d+)", seed_dir.name)
            if not seed_match:
                continue

            seed = int(seed_match.group(1))

            # Look for checkpoints
            ckpt_dir = seed_dir / "checkpoints"
            if not ckpt_dir.exists():
                ckpt_dir = seed_dir

            ckpt_path = find_checkpoint(ckpt_dir, checkpoint_type)
            if ckpt_path:
                experiments.append({
                    "exp_name": f"proto_{num_prototypes}_seed_{seed}",
                    "param": f"proto_{num_prototypes}",
                    "num_prototypes": num_prototypes,
                    "seed": seed,
                    "mode": "finetune",
                    "checkpoint_path": ckpt_path,
                    "checkpoint_type": checkpoint_type,
                })

    return experiments


def load_model(checkpoint_path: str, num_classes: int, device: torch.device):
    """Load classification model from checkpoint."""
    from src.classification.learners.classification_learner import ClassificationLearner

    model = ClassificationLearner.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        num_classes=num_classes,
        strict=True,
    )
    model.eval()
    model.to(device)

    return model


def create_datamodule(config_path: str, batch_size: int, num_workers: int):
    """Create classification data module."""
    from omegaconf import OmegaConf
    from dinov3.configs import get_default_config
    from src.classification.data.datamodule import ClassificationDataModule

    ssl_cfg_original = OmegaConf.load(config_path)
    default_cfg = get_default_config()
    ssl_cfg = OmegaConf.merge(default_cfg, ssl_cfg_original)

    class_cfg = OmegaConf.create({
        'train': {
            'dataset_path': ssl_cfg.train.dataset_path,
            'batch_size_per_gpu': batch_size,
            'num_workers': num_workers
        },
        'crops': {
            'global_crops_size': 256,
            'local_crops_size': 112
        },
        'compute_precision': {
            'param_dtype': 'bf16'
        }
    })

    datamodule = ClassificationDataModule(
        cfg=class_cfg,
        ssl_model=None,
        sampler_type="distributed",
        num_workers=num_workers,
        batch_size=batch_size
    )

    return datamodule


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
) -> Dict[str, Any]:
    """Compute classification metrics."""
    metrics = {}

    # Basic accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    # Top-5 accuracy (if applicable)
    if num_classes > 5:
        try:
            metrics['top5_accuracy'] = top_k_accuracy_score(y_true, y_prob, k=5)
        except:
            pass

    # Precision, Recall, F1
    for average in ['macro', 'weighted']:
        metrics[f'precision_{average}'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics[f'recall_{average}'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics[f'f1_{average}'] = f1_score(y_true, y_pred, average=average, zero_division=0)

    # AUC-ROC
    try:
        if num_classes == 2:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            metrics['auc_roc_macro'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except Exception:
        pass

    # Confusion matrix (store as list for JSON serialization)
    conf_matrix = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = conf_matrix.tolist()

    metrics['num_samples'] = len(y_true)

    return metrics


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
    class_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute per-class metrics."""
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    conf_matrix = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    support_per_class = conf_matrix.sum(axis=1)

    correct_per_class = np.diag(conf_matrix)
    accuracy_per_class = np.where(support_per_class > 0,
                                   correct_per_class / support_per_class,
                                   0.0)

    # Per-class AUC-ROC (one-vs-rest)
    auc_per_class = np.zeros(num_classes)
    for i in range(num_classes):
        try:
            binary_true = (y_true == i).astype(int)
            if binary_true.sum() > 0 and binary_true.sum() < len(binary_true):
                auc_per_class[i] = roc_auc_score(binary_true, y_prob[:, i])
        except Exception:
            auc_per_class[i] = np.nan

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    per_class_df = pd.DataFrame({
        'class_id': range(num_classes),
        'class_name': class_names[:num_classes],
        'support': support_per_class,
        'accuracy': accuracy_per_class,
        'precision': precision_per_class,
        'recall': recall_per_class,
        'f1': f1_per_class,
        'auc_roc': auc_per_class,
    })

    return per_class_df


@torch.no_grad()
def evaluate_model(
    model,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on dataloader.

    Returns:
        metrics: Dict with computed metrics
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        if isinstance(batch, dict):
            images = batch["images"].to(device)
            labels = batch["labels"]
        else:
            images, labels = batch
            images = images.to(device)

        logits = model(images)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy() if isinstance(labels, torch.Tensor) else labels)
        all_probs.append(probs.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)

    metrics = compute_metrics(y_true, y_pred, y_prob, num_classes)

    return metrics, y_true, y_pred, y_prob


def create_summary_dataframe(results_df: pd.DataFrame) -> pd.DataFrame:
    """Create summary DataFrame with mean +/- std across seeds."""
    metric_cols = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro',
                   'recall_macro', 'auc_roc_macro', 'top5_accuracy']
    metric_cols = [c for c in metric_cols if c in results_df.columns]

    group_cols = ['num_prototypes']

    summary_rows = []
    for group_key, group_df in results_df.groupby(group_cols):
        if isinstance(group_key, (int, float)):
            group_key = (group_key,)

        row = dict(zip(group_cols, group_key))
        row['n_seeds'] = len(group_df)
        row['seeds'] = [int(s) for s in group_df['seed'].unique()]

        for col in metric_cols:
            if col in group_df.columns:
                values = group_df[col].dropna()
                if len(values) > 0:
                    row[f'{col}_mean'] = values.mean()
                    row[f'{col}_std'] = values.std() if len(values) > 1 else 0.0

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values('num_prototypes')

    return summary_df


def main():
    args = parse_args()

    # Setup device
    device = torch.device(f"cuda:{args.devices}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Detect dataset and num_classes
    dataset_name = detect_dataset_from_config(args.config)
    if args.num_classes is None:
        if dataset_name in DATASET_CONFIGS:
            args.num_classes = DATASET_CONFIGS[dataset_name]["num_classes"]
            print(f"Auto-detected dataset: {dataset_name}, num_classes: {args.num_classes}")
        else:
            print("ERROR: Could not auto-detect num_classes. Please specify --num_classes")
            sys.exit(1)

    # Determine what to evaluate
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)

        if checkpoint_path.is_dir():
            resolved_ckpt = find_checkpoint(checkpoint_path, args.checkpoint_type)
            if resolved_ckpt is None:
                print(f"Error: No checkpoint found in {args.checkpoint} for type '{args.checkpoint_type}'")
                sys.exit(1)
            exp_info = parse_experiment_name(checkpoint_path.name)
            experiments = [{
                **exp_info,
                "checkpoint_path": resolved_ckpt,
                "checkpoint_type": args.checkpoint_type
            }]
            print(f"Using checkpoint ({args.checkpoint_type}): {resolved_ckpt}")
        else:
            exp_info = parse_experiment_name(checkpoint_path.parent.name)
            experiments = [{
                **exp_info,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_type": "manual"
            }]
        sweep_name = "single_eval"
        output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "results"

    elif args.sweep_dir:
        experiments = find_all_experiments(args.sweep_dir, args.checkpoint_type)
        sweep_name = Path(args.sweep_dir).name

        if not experiments:
            print(f"Error: No checkpoints found in {args.sweep_dir}")
            sys.exit(1)

        output_dir = Path(args.output_dir) if args.output_dir else Path(args.sweep_dir).parent / "results" / dataset_name
    else:
        print("Error: Must provide either --checkpoint or --sweep_dir")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"EVALUATION: {sweep_name}")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"Num classes: {args.num_classes}")
    print(f"Checkpoint type: {args.checkpoint_type}")
    print(f"Experiments to evaluate: {len(experiments)}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")

    # Create datamodule
    datamodule = create_datamodule(args.config, args.batch_size, args.num_workers)
    datamodule.setup()
    test_loader = datamodule.test_dataloader()

    if test_loader is None:
        print("Error: No test set available")
        sys.exit(1)

    print(f"Test set size: {len(datamodule.test_dataset)}")

    # Create predictions subdirectory
    predictions_dir = output_dir / sweep_name / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate all experiments
    all_results = []

    for exp in tqdm(experiments, desc="Evaluating experiments"):
        print(f"\n  -> {exp['exp_name']} ({exp['checkpoint_type']})")

        try:
            model = load_model(exp['checkpoint_path'], args.num_classes, device)
            metrics, y_true, y_pred, y_prob = evaluate_model(model, test_loader, device, args.num_classes)

            result = {
                'exp_name': exp['exp_name'],
                'param': exp['param'],
                'num_prototypes': exp.get('num_prototypes', 0),
                'mode': exp['mode'],
                'seed': int(exp['seed']),
                'checkpoint_type': exp['checkpoint_type'],
                'checkpoint_path': exp['checkpoint_path'],
                **{k: v for k, v in metrics.items() if k != 'confusion_matrix'}
            }
            all_results.append(result)

            print(f"    Accuracy: {metrics['accuracy']:.4f}, F1 (macro): {metrics['f1_macro']:.4f}")

            # Save numpy arrays
            exp_pred_dir = predictions_dir / exp['exp_name']
            exp_pred_dir.mkdir(parents=True, exist_ok=True)

            np.save(exp_pred_dir / "labels.npy", y_true)
            np.save(exp_pred_dir / "predictions.npy", y_pred)
            np.save(exp_pred_dir / "scores.npy", y_prob)

            # Save confusion matrix
            conf_matrix = np.array(metrics['confusion_matrix'])
            cm_df = pd.DataFrame(conf_matrix)
            cm_df.to_csv(exp_pred_dir / "confusion_matrix.csv")

            # Per-class metrics
            per_class_df = compute_per_class_metrics(y_true, y_pred, y_prob, args.num_classes)
            per_class_df.to_csv(exp_pred_dir / "per_class_metrics.csv", index=False)

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_results:
        print("Error: No successful evaluations")
        sys.exit(1)

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(['num_prototypes', 'seed'])

    # Save full results
    results_file = output_dir / f"{sweep_name}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\nSaved full results to {results_file}")

    # Create and save summary
    summary_df = create_summary_dataframe(results_df)
    summary_file = output_dir / f"{sweep_name}_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved summary to {summary_file}")

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY (mean +/- std across seeds)")
    print(f"{'='*80}")

    for _, row in summary_df.iterrows():
        num_proto = row.get('num_prototypes', 'unknown')
        n = row.get('n_seeds', 1)
        acc_mean = row.get('accuracy_mean', 0)
        acc_std = row.get('accuracy_std', 0)
        f1_mean = row.get('f1_macro_mean', 0)
        f1_std = row.get('f1_macro_std', 0)

        print(f"  proto_{num_proto:4d} (n={n}): "
              f"Acc={acc_mean:.4f}+/-{acc_std:.4f}, F1={f1_mean:.4f}+/-{f1_std:.4f}")

    print(f"{'='*80}\n")

    # Save JSON with all details
    json_file = output_dir / f"{sweep_name}.json"
    with open(json_file, 'w') as f:
        json.dump({
            'sweep_name': sweep_name,
            'dataset': dataset_name,
            'checkpoint_type': args.checkpoint_type,
            'num_experiments': len(all_results),
            'results': all_results,
            'summary': summary_df.to_dict(orient='records')
        }, f, indent=2, default=str)
    print(f"Saved JSON to {json_file}")


if __name__ == "__main__":
    main()
