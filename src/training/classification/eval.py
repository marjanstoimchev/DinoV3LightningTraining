#!/usr/bin/env python3
"""
eval.py

Evaluate classification models on the test set.
Supports multiple seeds, best/last checkpoint selection, and aggregated results.

Output structure:
  results/classification/{dataset}/
    {sweep_name}.csv                                    # All individual results
    {sweep_name}_summary.csv                            # Aggregated by experiment (mean +/- std)
    {sweep_name}.json                                   # Full details
    {sweep_name}/predictions/
      seed_{seed}/
        labels.npy                                      # Ground truth labels
        predictions.npy                                 # Predicted labels
        scores.npy                                      # Prediction probabilities [N, num_classes]
        confusion_matrix.csv                            # Confusion matrix with class names
        per_class_metrics.csv                           # Per-class precision, recall, F1, AUC

Usage examples:

  # Evaluate all seed checkpoints for lineareval
  python src/training/classification/eval.py \\
      --config-file configs/eurosat/config_classification.yaml \\
      --sweep-dir output/eurosat/checkpoints/classification/lineareval \\
      --checkpoint-type last \\
      --dataset-type eurosat \\
      --dataset-path "HuggingFace:name=blanchon/EuroSAT_RGB" \\
      --num-classes 10

  # Evaluate using best checkpoint (lowest val_loss)
  python src/training/classification/eval.py \\
      --config-file configs/eurosat/config_classification.yaml \\
      --sweep-dir output/eurosat/checkpoints/classification/finetune \\
      --checkpoint-type best \\
      --dataset-type eurosat \\
      --num-classes 10

  # Evaluate a single checkpoint
  python src/training/classification/eval.py \\
      --config-file configs/eurosat/config_classification.yaml \\
      --checkpoint output/eurosat/checkpoints/classification/finetune/seed_42/last.ckpt \\
      --dataset-type eurosat \\
      --num-classes 10
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
ROOT = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from src.classification.models.linear_classifier import LinearClassifier
from src.classification.learners.classification_learner import ClassificationLearner
from src.classification.data.datamodule import ClassificationDataModule
from omegaconf import OmegaConf

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


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate classification models on test set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config-file", type=str, required=True, help="Path to config.yaml")

    # Checkpoint selection
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to single checkpoint to evaluate")
    p.add_argument("--sweep-dir", type=str, default=None,
                   help="Sweep directory containing seed subdirs (e.g., output/eurosat/checkpoints/classification/lineareval)")
    p.add_argument("--checkpoint-type", type=str, default="last", choices=["last", "best"],
                   help="Which checkpoint to use: 'last' or 'best' (lowest val_loss)")

    # Output
    p.add_argument("--output-dir", type=str, default="results/classification",
                   help="Base output directory for results")

    # Runtime settings
    p.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation")
    p.add_argument("--device", type=str, default="0", help="GPU device to use (e.g., '0' or 'cuda:0')")
    p.add_argument("--num-workers", type=int, default=8, help="Number of data loading workers")

    # Dataset settings
    p.add_argument("--dataset-type", default="custom", type=str,
                   choices=["custom", "NCTCRCHE100K", "eurosat", "DTD", "imagenet1k", "oxford_pets", "tissue"],
                   help="Type of dataset to use")
    p.add_argument("--dataset-path", default=None, type=str,
                   help="Path to dataset (overrides config if provided)")
    p.add_argument("--num-classes", type=int, required=True, help="Number of classes")

    # Model settings
    p.add_argument("--encoder-type", default="teacher", choices=["teacher", "student"],
                   help="Which encoder type was used (teacher or student)")

    return p.parse_args()


def parse_experiment_name(exp_name: str) -> Dict[str, Any]:
    """
    Parse experiment folder name to extract components.

    Expected format: seed_{seed}
    Examples:
        - seed_42 -> {seed: 42}
        - seed_123 -> {seed: 123}
    """
    info = {"exp_name": exp_name, "seed": 0, "mode": "unknown"}

    # Try to extract seed
    seed_match = re.search(r'seed_(\d+)', exp_name)
    if seed_match:
        info["seed"] = int(seed_match.group(1))

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
        # Format: {mode}_seed_{seed}_epoch_XX_step_XXXXXX_loss_Y.YYYY.ckpt
        ckpts = list(exp_dir.glob("*.ckpt"))

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
    Find all experiments (seed directories) in a sweep directory.

    Returns:
        List of dicts with experiment info and checkpoint path
    """
    base_path = Path(sweep_dir)
    experiments = []

    if not base_path.exists():
        print(f"Warning: Sweep directory {sweep_dir} does not exist")
        return experiments

    # Look for seed_* subdirectories
    for subdir in sorted(base_path.iterdir()):
        if subdir.is_dir() and subdir.name.startswith("seed_"):
            ckpt_path = find_checkpoint(subdir, checkpoint_type)
            if ckpt_path:
                exp_info = parse_experiment_name(subdir.name)
                exp_info["checkpoint_path"] = ckpt_path
                exp_info["checkpoint_type"] = checkpoint_type
                # Determine mode from parent directory name
                exp_info["mode"] = base_path.name  # e.g., "lineareval" or "finetune"
                experiments.append(exp_info)

    return experiments


def load_model(
    checkpoint_path: str,
    num_classes: int,
    encoder_type: str,
    device: torch.device
) -> ClassificationLearner:
    """Load classification model from checkpoint."""
    model = ClassificationLearner.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        num_classes=num_classes,
        encoder_type=encoder_type,
    )
    model.eval()
    model.to(device)

    return model


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
        metrics['top5_accuracy'] = top_k_accuracy_score(y_true, y_prob, k=5)

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
    # Per-class precision, recall, f1
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Per-class support (number of samples)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    support_per_class = conf_matrix.sum(axis=1)

    # Per-class accuracy
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

    # Build DataFrame
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
    model: ClassificationLearner,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on dataloader.

    Returns:
        metrics: Dict with computed metrics
        y_true: Ground truth labels (numpy array)
        y_pred: Predicted labels (numpy array)
        y_prob: Prediction probabilities (numpy array, shape [N, num_classes])
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        # Handle both dictionary format and tuple format
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
    """
    Create summary DataFrame with mean +/- std across seeds.

    Groups by mode and computes statistics.
    """
    # Columns to aggregate
    metric_cols = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro',
                   'recall_macro', 'auc_roc_macro', 'top5_accuracy']
    metric_cols = [c for c in metric_cols if c in results_df.columns]

    # Group by mode
    group_cols = ['mode']

    summary_rows = []
    for group_key, group_df in results_df.groupby(group_cols):
        if isinstance(group_key, str):
            group_key = (group_key,)

        row = dict(zip(group_cols, group_key))
        row['n_seeds'] = len(group_df)
        row['seeds'] = [int(s) for s in group_df['seed'].unique()]  # Convert to native int

        for col in metric_cols:
            if col in group_df.columns:
                values = group_df[col].dropna()
                if len(values) > 0:
                    row[f'{col}_mean'] = values.mean()
                    row[f'{col}_std'] = values.std() if len(values) > 1 else 0.0

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    return summary_df


def main():
    args = parse_args()

    # Setup device
    if args.device.isdigit():
        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine dataset name from dataset-type or sweep directory
    dataset_name = args.dataset_type
    if args.sweep_dir:
        # Try to extract dataset name from sweep directory path
        # e.g., output/eurosat/checkpoints/classification/lineareval -> eurosat
        parts = Path(args.sweep_dir).parts
        if "output" in parts:
            idx = parts.index("output")
            if idx + 1 < len(parts):
                dataset_name = parts[idx + 1]

    num_classes = args.num_classes

    # Setup output directory
    output_dir = Path(args.output_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine what to evaluate
    if args.checkpoint:
        # Single checkpoint
        exp_info = parse_experiment_name(Path(args.checkpoint).parent.name)
        # Determine mode from path
        mode = "unknown"
        if "lineareval" in args.checkpoint:
            mode = "lineareval"
        elif "finetune" in args.checkpoint:
            mode = "finetune"
        exp_info["mode"] = mode

        experiments = [{
            **exp_info,
            "checkpoint_path": args.checkpoint,
            "checkpoint_type": "manual"
        }]
        sweep_name = "single_eval"

    elif args.sweep_dir:
        # Full sweep over seeds
        experiments = find_all_experiments(args.sweep_dir, args.checkpoint_type)
        # Determine mode from directory name
        mode = Path(args.sweep_dir).name  # e.g., "lineareval" or "finetune"
        sweep_name = f"{dataset_name}_{mode}"

        if not experiments:
            print(f"Error: No checkpoints found in {args.sweep_dir}")
            sys.exit(1)
    else:
        print("Error: Must provide either --checkpoint or --sweep-dir")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"EVALUATION: {sweep_name}")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"Num classes: {num_classes}")
    print(f"Checkpoint type: {args.checkpoint_type}")
    print(f"Experiments to evaluate: {len(experiments)}")
    print(f"{'='*80}\n")

    # Create datamodule for test data
    sys.path.append(str(ROOT / 'dinov3'))
    from dinov3.configs import get_default_config
    ssl_cfg_original = OmegaConf.load(args.config_file)
    default_cfg = get_default_config()
    ssl_cfg = OmegaConf.merge(default_cfg, ssl_cfg_original)

    # Determine dataset path
    dataset_path = args.dataset_path
    if not dataset_path:
        dataset_paths = {
            "NCTCRCHE100K": "HuggingFace:name=DykeF/NCTCRCHE100K",
            "eurosat": "HuggingFace:name=blanchon/EuroSAT_RGB",
            "DTD": "HuggingFace:name=cansa/Describable-Textures-Dataset-DTD",
            "imagenet1k": "HuggingFace:name=ILSVRC/imagenet-1k",
            "oxford_pets": "HuggingFace:name=timm/oxford-iiit-pet",
            "tissue": "CustomTIFF:root=../Datasets/tissue/"
        }
        dataset_path = dataset_paths.get(args.dataset_type, ssl_cfg.train.dataset_path)

    # Create a temporary config for the data module
    class_cfg = OmegaConf.create({
        'train': {
            'dataset_path': dataset_path,
            'batch_size_per_gpu': args.batch_size,
            'num_workers': args.num_workers
        },
        'crops': {
            'global_crops_size': 256,
            'local_crops_size': 112
        },
        'compute_precision': {
            'param_dtype': 'bf16'
        }
    })

    # Create data module
    datamodule = ClassificationDataModule(
        cfg=class_cfg,
        ssl_model=None,
        sampler_type="distributed",
        num_workers=args.num_workers,
        batch_size=args.batch_size
    )
    datamodule.setup()
    test_loader = datamodule.test_dataloader()

    if test_loader is None:
        print("Error: No test set available")
        sys.exit(1)

    print(f"Test set size: {len(datamodule.test_dataset)}")

    # Try to get class names from dataset
    class_names = None
    try:
        if hasattr(datamodule.test_dataset, 'classes'):
            class_names = datamodule.test_dataset.classes
        elif hasattr(datamodule.test_dataset, 'class_names'):
            class_names = datamodule.test_dataset.class_names
        elif hasattr(datamodule.test_dataset, 'dataset'):
            inner = datamodule.test_dataset.dataset
            if hasattr(inner, 'features'):
                # HuggingFace dataset
                features = inner.features
                if 'label' in features and hasattr(features['label'], 'names'):
                    class_names = features['label'].names
    except Exception:
        pass

    if class_names:
        print(f"Class names: {class_names[:5]}{'...' if len(class_names) > 5 else ''}")

    # Create predictions subdirectory
    predictions_dir = output_dir / sweep_name / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate all experiments
    all_results = []

    for exp in tqdm(experiments, desc="Evaluating experiments"):
        print(f"\n  -> seed_{exp['seed']} ({exp['checkpoint_type']})")

        try:
            model = load_model(
                exp['checkpoint_path'],
                num_classes,
                args.encoder_type,
                device
            )
            metrics, y_true, y_pred, y_prob = evaluate_model(model, test_loader, device, num_classes)

            # Add experiment info
            result = {
                'exp_name': exp['exp_name'],
                'mode': exp['mode'],
                'seed': int(exp['seed']),  # Ensure native int
                'checkpoint_type': exp['checkpoint_type'],
                'checkpoint_path': exp['checkpoint_path'],
                **{k: v for k, v in metrics.items() if k != 'confusion_matrix'}
            }
            all_results.append(result)

            print(f"    Accuracy: {metrics['accuracy']:.4f}, F1 (macro): {metrics['f1_macro']:.4f}")

            # Save numpy arrays (labels, predictions, scores)
            exp_pred_dir = predictions_dir / f"seed_{exp['seed']}"
            exp_pred_dir.mkdir(parents=True, exist_ok=True)

            np.save(exp_pred_dir / "labels.npy", y_true)
            np.save(exp_pred_dir / "predictions.npy", y_pred)
            np.save(exp_pred_dir / "scores.npy", y_prob)

            # Save confusion matrix
            conf_matrix = np.array(metrics['confusion_matrix'])
            cm_df = pd.DataFrame(conf_matrix)
            if class_names:
                cm_df.columns = class_names[:num_classes]
                cm_df.index = class_names[:num_classes]
            cm_df.to_csv(exp_pred_dir / "confusion_matrix.csv")

            # Compute and save per-class metrics
            per_class_df = compute_per_class_metrics(y_true, y_pred, y_prob, num_classes, class_names)
            per_class_df.to_csv(exp_pred_dir / "per_class_metrics.csv", index=False)

            # Clean up model
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

    # Sort by mode and seed
    results_df = results_df.sort_values(['mode', 'seed'])

    # Save full results
    results_file = output_dir / f"{sweep_name}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\nSaved full results to {results_file}")

    # Create and save summary (aggregated across seeds)
    summary_df = create_summary_dataframe(results_df)
    summary_file = output_dir / f"{sweep_name}_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved summary to {summary_file}")

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY (mean +/- std across seeds)")
    print(f"{'='*80}")

    for _, row in summary_df.iterrows():
        mode = row.get('mode', 'unknown')
        n = row.get('n_seeds', 1)
        acc_mean = row.get('accuracy_mean', 0)
        acc_std = row.get('accuracy_std', 0)
        f1_mean = row.get('f1_macro_mean', 0)
        f1_std = row.get('f1_macro_std', 0)

        print(f"  {mode:15s} (n={n}): "
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
