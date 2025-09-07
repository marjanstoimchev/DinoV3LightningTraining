#!/usr/bin/env python3
"""
PyTorch Lightning training script for DINOv3 fine-tuning
Exact replica of DINOv3 training functionality using Lightning framework
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
sys.path.append(str(Path(__file__).parent.parent.parent / 'dinov3'))
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.callbacks.enhanced_progress_bar import DINOv3EnhancedProgressBar
from src.callbacks.base_progress_bar import DINOv3BaseProgressBar
from src.checkpointing.model_checkpoint import DINOv3ModelCheckpoint
from src.models.dinov3_lightning_model import DINOv3LightningModule  
from src.models.dinov3_lightning_datamodule import DINOv3DataModule, MultiResolutionDINOv3DataModule
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
    
    logger = logging.getLogger("dinov3_lightning")
    return logger


def get_args_parser():
    """Argument parser similar to DINOv3"""
    parser = argparse.ArgumentParser("DINOv3 PyTorch Lightning training", add_help=True)
    
    # Required arguments
    parser.add_argument(
        "--config-file", 
        required=True,
        metavar="FILE", 
        help="path to config file (e.g., ../DinoV3Tr/custom_config_finetuned.yaml)"
    )
    parser.add_argument(
        "--checkpoint-path",
        default="dinov3/dinov3/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        type=str,
        help="Path to pretrained DINOv3 checkpoint"
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
    
    return parser


def create_callbacks(cfg: OmegaConf, output_dir: str, args):
    """Create Lightning callbacks"""
    callbacks = []
    
    # Create checkpoints directory and ensure it's absolute path
    checkpoint_dir = os.path.abspath(os.path.join(output_dir, "checkpoints"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Step-based checkpoint callback (epoch-based removed due to DDP issues)
    checkpoint_callback = DINOv3ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="model_epoch_{epoch:02d}_step_{step:06d}_loss_{total_loss:.6f}",
        monitor="total_loss",
        mode="min",
        save_top_k=cfg.checkpointing.max_to_keep if hasattr(cfg, 'checkpointing') else 3,
        every_n_train_steps=args.save_every_n_steps,
        save_last=True,
        verbose=True,
        auto_insert_metric_name=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Choose appropriate progress bar based on actual sampler type that will be used
    import torch.distributed as dist
    will_use_distributed = (args.sampler_type and args.sampler_type.lower() == "distributed" and 
                           dist.is_available() and dist.is_initialized())
    will_use_epoch = (args.sampler_type and args.sampler_type.lower() == "epoch") or \
                    (args.sampler_type and args.sampler_type.lower() == "distributed" and 
                     not (dist.is_available() and dist.is_initialized()))
    
    if will_use_distributed or will_use_epoch:
        # Use base progress bar for distributed and epoch samplers (finite length)
        progress_bar = DINOv3BaseProgressBar(
            refresh_rate=1, 
            leave=True, 
            log_every_n_steps=args.progress_log_every_n_steps
        )
        sampler_name = "distributed" if will_use_distributed else "epoch"
        print(f"Using DINOv3BaseProgressBar for {sampler_name} sampler")
    else:
        # Use enhanced progress bar for infinite samplers (default)
        progress_bar = DINOv3EnhancedProgressBar(
            refresh_rate=1, 
            leave=True, 
            log_every_n_steps=args.progress_log_every_n_steps
        )
        print("Using DINOv3EnhancedProgressBar for infinite sampler")
    
    callbacks.append(progress_bar)
    
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
    
    logger.info(f"Starting DINOv3 Lightning training")
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
    
    # Override batch size if provided
    if args.batch_size:
        per_gpu_batch_size = args.batch_size // args.gpus
        cfg.train.batch_size_per_gpu = per_gpu_batch_size
        logger.info(f"Overriding batch_size_per_gpu: {per_gpu_batch_size} (total: {args.batch_size}, gpus: {args.gpus})")
    
    # Override compile setting if provided
    if args.compile:
        cfg.train.compile = True
        logger.info("Enabling PyTorch compilation")
    
    # Update output directory in config
    cfg.train.output_dir = output_dir
    
    logger.info(f"Final configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Create model
    logger.info("Creating DINOv3 Lightning model...")
    model = DINOv3LightningModule(
        cfg_path=cfg,
        checkpoint_path=args.checkpoint_path if os.path.exists(args.checkpoint_path) else None
    )
    
    # Create data module
    logger.info("Creating data module...")
    # Check if we need multi-resolution data loading
    if (hasattr(cfg.crops, 'global_crops_size') and isinstance(cfg.crops.global_crops_size, list) and 
        len(cfg.crops.global_crops_size) > 1):
        logger.info("Using multi-resolution data module")
        datamodule = MultiResolutionDINOv3DataModule(cfg, model.ssl_model, args.sampler_type)
    else:
        logger.info("Using standard data module")
        datamodule = DINOv3DataModule(cfg, model.ssl_model, args.sampler_type)
    
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
    
    # Create trainer
    logger.info("Creating Lightning trainer...")
    trainer = pl.Trainer(
        max_epochs=cfg.optim.epochs,
        max_steps=max_steps if max_steps > 0 else -1,
        accelerator="auto",
        devices=args.gpus,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        # Manual optimization handles gradient clipping internally
        gradient_clip_val=None,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=args.log_every_n_steps,
        limit_train_batches=args.limit_train_batches,
        fast_dev_run=args.fast_dev_run,
        sync_batchnorm=True if args.gpus > 1 or args.num_nodes > 1 else False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True,
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
    
    # Save final model
    final_checkpoint_path = os.path.join(output_dir, "final_model.ckpt")
    trainer.save_checkpoint(final_checkpoint_path)
    logger.info(f"Final model saved to: {final_checkpoint_path}")
    
    # Extract and save just the SSL model state dict for compatibility
    ssl_model_path = os.path.join(output_dir, "final_ssl_model.pth")
    torch.save(model.ssl_model.state_dict(), ssl_model_path)
    logger.info(f"SSL model state dict saved to: {ssl_model_path}")


if __name__ == "__main__":
    main()