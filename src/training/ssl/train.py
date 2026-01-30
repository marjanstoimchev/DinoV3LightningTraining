#!/usr/bin/env python3
"""
PyTorch Lightning training script for DINOv3 SSL pretraining.
Supports multiple datasets with YAML configuration.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from omegaconf import OmegaConf

# Add paths for DINOv3 modules and src modules
ROOT = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / 'dinov3'))

from src.callbacks.enhanced_progress_bar import DINOv3EnhancedProgressBar
from src.callbacks.base_progress_bar import DINOv3BaseProgressBar
from src.checkpointing.model_checkpoint import DINOv3ModelCheckpoint
from src.ssl.models.ssl_learner import SSLLearner
from src.ssl.data.datamodule import SSLDataModule, MultiResolutionSSLDataModule
from dinov3.configs import get_default_config


def setup_logging(output_dir: str):
    """Setup logging exactly like DINOv3"""
    os.makedirs(output_dir, exist_ok=True)

    # Setup root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger("ssl_training")
    return logger


def get_args_parser():
    """Argument parser similar to DINOv3"""
    parser = argparse.ArgumentParser("DINOv3 SSL Pretraining", add_help=True)

    # Required arguments
    parser.add_argument(
        "--config-file",
        required=True,
        metavar="FILE",
        help="path to config file (e.g., configs/eurosat/config_ssl_pretraining.yaml)"
    )
    parser.add_argument(
        "--checkpoint-path",
        default="",
        type=str,
        help="Path to pretrained DINOv3 checkpoint (empty for from-scratch training)"
    )

    # Training arguments
    parser.add_argument(
        "--output-dir",
        default="./lightning_output",
        type=str,
        help="Path to save logs and checkpoints"
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="RNG seed"
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to Lightning checkpoint to resume from"
    )

    # Lightning-specific arguments
    parser.add_argument(
        "--gpus",
        default=1,
        type=int,
        help="Number of GPUs to use"
    )
    parser.add_argument(
        "--num-nodes",
        default=1,
        type=int,
        help="Number of nodes for distributed training"
    )
    parser.add_argument(
        "--precision",
        default="bf16-mixed",
        type=str,
        choices=["32", "16", "bf16-mixed", "16-mixed"],
        help="Training precision"
    )
    parser.add_argument(
        "--strategy",
        default="auto",
        type=str,
        help="Training strategy (auto, ddp, ddp_sharded, etc.)"
    )
    parser.add_argument(
        "--accumulate-grad-batches",
        default=1,
        type=int,
        help="Accumulate gradients over N batches"
    )
    parser.add_argument(
        "--max-epochs",
        default=None,
        type=int,
        help="Override max epochs from config"
    )
    parser.add_argument(
        "--limit-train-batches",
        default=1.0,
        type=float,
        help="Limit training batches (useful for testing)"
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Fast dev run for testing"
    )

    # Logging arguments
    parser.add_argument(
        "--log-every-n-steps",
        default=10,
        type=int,
        help="Log every N training steps"
    )
    parser.add_argument(
        "--save-every-n-steps",
        default=100,
        type=int,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--progress-log-every-n-steps",
        default=10,
        type=int,
        help="Log progress every N training steps"
    )

    # Data loading arguments
    parser.add_argument(
        "--sampler-type",
        default=None,
        type=str,
        choices=["infinite", "distributed", "sharded_infinite", "epoch"],
        help="Override sampler type (infinite, distributed, sharded_infinite)"
    )
    parser.add_argument(
        "--batch-size",
        default=None,
        type=int,
        help="Total batch size across all GPUs (will be divided by number of GPUs)"
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable PyTorch 2.0 compilation for faster training"
    )

    # Dataset selection arguments
    parser.add_argument(
        "--dataset-type",
        default="custom",
        type=str,
        choices=["custom", "NCTCRCHE100K", "eurosat", "DTD", "imagenet1k", "oxford_pets", "tissue"],
        help="Type of dataset to use for training"
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        type=str,
        help="Path to dataset (overrides config if provided)"
    )
    parser.add_argument(
        "--num-prototypes",
        default=None,
        type=int,
        help="Number of prototypes for DINO and iBOT heads (overrides config if provided)"
    )
    parser.add_argument(
        "--enable-gram",
        action="store_true",
        help="Enable GRAM loss (default: disabled)"
    )
    parser.add_argument(
        "--gram-weight",
        default=None,
        type=float,
        help="GRAM loss weight (default: 1.0 when enabled)"
    )

    return parser


def create_callbacks(cfg: OmegaConf, output_dir: str, args):
    """Create Lightning callbacks"""
    callbacks = []

    # Create checkpoints directory and ensure it's absolute path
    # Store pretraining checkpoints in checkpoints/pretraining from first stage
    checkpoint_dir = os.path.abspath(os.path.join(output_dir, "checkpoints", "pretraining"))
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Determine checkpoint save interval: use config if available, else command line arg
    save_every_n_steps = args.save_every_n_steps
    if hasattr(cfg, 'checkpointing') and hasattr(cfg.checkpointing, 'period'):
        save_every_n_steps = cfg.checkpointing.period
    print(f"Checkpoint save interval: every {save_every_n_steps} steps")

    # Step-based checkpoint callback (epoch-based removed due to DDP issues)
    checkpoint_callback = DINOv3ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="model_epoch_{epoch:02d}_step_{step:06d}_loss_{total_loss:.6f}",
        monitor="total_loss",
        mode="min",
        save_top_k=cfg.checkpointing.max_to_keep if hasattr(cfg, 'checkpointing') else 3,
        every_n_train_steps=save_every_n_steps,
        save_last=True,
        save_on_train_epoch_end=False,  # Don't save at every epoch end, only at step intervals
        verbose=True,
        auto_insert_metric_name=True,
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    # Custom progress bar disabled - using Lightning's native progress bar
    # import torch.distributed as dist
    # will_use_distributed = (args.sampler_type and args.sampler_type.lower() == "distributed" and
    #                        dist.is_available() and dist.is_initialized())
    # will_use_epoch = (args.sampler_type and args.sampler_type.lower() == "epoch") or \
    #                 (args.sampler_type and args.sampler_type.lower() == "distributed" and
    #                  not (dist.is_available() and dist.is_initialized()))
    #
    # if will_use_distributed or will_use_epoch:
    #     progress_bar = DINOv3BaseProgressBar(
    #         refresh_rate=1,
    #         leave=True,
    #         log_every_n_steps=args.progress_log_every_n_steps
    #     )
    #     sampler_name = "distributed" if will_use_distributed else "epoch"
    #     print(f"Using DINOv3BaseProgressBar for {sampler_name} sampler")
    # else:
    #     progress_bar = DINOv3EnhancedProgressBar(
    #         refresh_rate=1,
    #         leave=True,
    #         log_every_n_steps=args.progress_log_every_n_steps
    #     )
    #     print("Using DINOv3EnhancedProgressBar for infinite sampler")
    #
    # callbacks.append(progress_bar)

    return callbacks


def create_loggers(output_dir: str):
    """Create Lightning loggers"""
    loggers = []

    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=output_dir,
        name="tensorboard_logs",
        version="",
    )
    loggers.append(tb_logger)

    # CSV logger
    csv_logger = CSVLogger(
        save_dir=output_dir,
        name="csv_logs",
        version="",
    )
    loggers.append(csv_logger)

    return loggers


def main():
    # Parse arguments
    parser = get_args_parser()
    args = parser.parse_args()

    # Set seed
    pl.seed_everything(args.seed, workers=True)

    # Setup output directory and logging
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir)

    logger.info(f"Starting DINOv3 SSL pretraining")
    logger.info(f"Arguments: {args}")

    # Load configuration
    logger.info(f"Loading config from {args.config_file}")
    cfg = OmegaConf.load(args.config_file)

    # Merge with default config
    default_cfg = get_default_config()
    cfg = OmegaConf.merge(default_cfg, cfg)

    # Override config with command line args if provided
    if args.max_epochs:
        cfg.optim.epochs = args.max_epochs

    # Override number of prototypes if provided
    if args.num_prototypes:
        if hasattr(cfg, 'dino') and hasattr(cfg.dino, 'head_n_prototypes'):
            cfg.dino.head_n_prototypes = args.num_prototypes
        if hasattr(cfg, 'ibot') and hasattr(cfg.ibot, 'head_n_prototypes'):
            cfg.ibot.head_n_prototypes = args.num_prototypes
        logger.info(f"Overriding number of prototypes: {args.num_prototypes}")

    # Override dataset path if provided
    if args.dataset_path:
        cfg.train.dataset_path = args.dataset_path
        logger.info(f"Overriding dataset_path: {args.dataset_path}")

    # Override dataset type if needed
    if args.dataset_type and args.dataset_type != "custom":
        # Map dataset types to appropriate paths/configs
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
            cfg.train.dataset_path = dataset_paths[args.dataset_type]
            logger.info(f"Using {args.dataset_type} dataset path: {cfg.train.dataset_path}")

    # Override batch size if provided (batch_size is TOTAL, divide by GPUs - same as MPP)
    # Check for SLURM-based distribution
    slurm_ntasks_for_batch = int(os.environ.get("SLURM_NTASKS", "1"))
    slurm_launch_for_batch = os.environ.get("SLURM_LAUNCH", "")
    if args.batch_size:
        # Use SLURM task count if SLURM-based distribution, otherwise use args.gpus
        if slurm_launch_for_batch and slurm_ntasks_for_batch > 1:
            num_workers = slurm_ntasks_for_batch
        else:
            num_workers = args.gpus if args.gpus > 0 else 1
        per_gpu_batch_size = args.batch_size // num_workers
        cfg.train.batch_size_per_gpu = per_gpu_batch_size
        logger.info(f"Batch size: {args.batch_size} total -> {per_gpu_batch_size} per GPU ({num_workers} workers)")

    # Override compile setting if provided
    if args.compile:
        cfg.train.compile = True
        logger.info("Enabling PyTorch compilation")

    # Override GRAM settings if provided
    if args.enable_gram:
        cfg.gram.use_loss = True
        logger.info("Enabling GRAM loss")
    if args.gram_weight is not None:
        cfg.gram.loss_weight = args.gram_weight
        logger.info(f"Setting GRAM loss weight: {args.gram_weight}")

    # Update output directory in config
    cfg.train.output_dir = output_dir

    logger.info(f"Final configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Create model
    logger.info("Creating SSL Learner...")
    model = SSLLearner(
        cfg_path=cfg,
        checkpoint_path=args.checkpoint_path if os.path.exists(args.checkpoint_path) else None
    )

    # Create data module
    logger.info("Creating data module...")
    # Check if we need multi-resolution data loading
    if (hasattr(cfg.crops, 'global_crops_size') and isinstance(cfg.crops.global_crops_size, list) and
        len(cfg.crops.global_crops_size) > 1):
        logger.info("Using multi-resolution data module")
        datamodule = MultiResolutionSSLDataModule(cfg, model.ssl_model, args.sampler_type)
    else:
        logger.info("Using standard data module")
        datamodule = SSLDataModule(cfg, model.ssl_model, args.sampler_type)

    # Create callbacks and loggers
    callbacks = create_callbacks(cfg, output_dir, args)
    loggers = create_loggers(output_dir)

    # Calculate max steps
    # For distributed and epoch samplers, ignore OFFICIAL_EPOCH_LENGTH and use actual dataset size
    if (hasattr(cfg.train, 'OFFICIAL_EPOCH_LENGTH') and
        args.sampler_type not in ["distributed", "epoch"]):
        max_steps = cfg.optim.epochs * cfg.train.OFFICIAL_EPOCH_LENGTH
        logger.info(f"Using OFFICIAL_EPOCH_LENGTH: {cfg.train.OFFICIAL_EPOCH_LENGTH} steps per epoch")
    else:
        max_steps = -1  # Let Lightning determine based on actual dataset size
        if args.sampler_type in ["distributed", "epoch"]:
            logger.info(f"Using {args.sampler_type} sampler - ignoring OFFICIAL_EPOCH_LENGTH, using actual dataset size")

    # Detect SLURM-based distribution (srun with multiple tasks)
    slurm_ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    slurm_procid = os.environ.get("SLURM_PROCID")
    slurm_launch = os.environ.get("SLURM_LAUNCH", "")

    # Determine effective devices and strategy
    effective_devices = args.gpus
    effective_strategy = args.strategy
    effective_num_nodes = args.num_nodes

    if slurm_launch and slurm_ntasks > 1 and slurm_procid is not None:
        # SLURM is handling distribution - each task gets 1 GPU
        effective_devices = 1
        effective_strategy = "ddp"
        logger.info(f"SLURM distribution detected: task {slurm_procid}/{slurm_ntasks}")
        logger.info(f"Using devices=1, strategy=ddp (SLURM handles multi-GPU)")
    elif args.gpus > 1:
        effective_strategy = "ddp" if args.strategy == "auto" else args.strategy
        logger.info(f"Multi-GPU training: {args.gpus} GPUs, strategy={effective_strategy}")

    # Create trainer
    logger.info("Creating Lightning trainer...")
    trainer = pl.Trainer(
        max_epochs=cfg.optim.epochs,
        max_steps=max_steps if max_steps > 0 else -1,
        accelerator="auto",
        devices=effective_devices,
        num_nodes=effective_num_nodes,
        strategy=effective_strategy,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        # Manual optimization handles gradient clipping internally
        gradient_clip_val=None,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=args.log_every_n_steps,
        limit_train_batches=args.limit_train_batches,
        fast_dev_run=args.fast_dev_run,
        sync_batchnorm=True if slurm_ntasks > 1 or args.gpus > 1 or args.num_nodes > 1 else False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,
        use_distributed_sampler=False,  # DINOv3 handles its own sampling
    )

    logger.info(f"Trainer configuration: {trainer}")

    # Start training
    logger.info("Starting training...")
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=args.resume_from_checkpoint
    )

    logger.info("Training completed!")

    logger.info("Checkpoints saved automatically by Lightning - 'last.ckpt' and best models based on monitored metric.")


if __name__ == "__main__":
    main()
