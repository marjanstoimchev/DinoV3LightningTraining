#!/usr/bin/env python3
"""
Enhanced progress bar for DINOv3 Lightning training
Shows individual losses, proper epoch/step counting, ETA, and speed
Designed to work with infinite samplers and custom epoch lengths
"""

import time
from typing import Any, Dict, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from rich.console import Console
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    SpinnerColumn,
)


class DINOv3EnhancedProgressBar(RichProgressBar):
    """Enhanced progress bar for DINOv3 with individual loss display and proper epoch tracking"""
    
    def __init__(self, refresh_rate: int = 1, leave: bool = False, log_every_n_steps: int = 50):
        # Custom theme for better visibility
        theme = RichProgressBarTheme(
            description="white",
            progress_bar="#6206E0",
            progress_bar_finished="#6206E0", 
            progress_bar_pulse="#6206E0",
            batch_progress="white",
            time="grey54",
            processing_speed="grey70",
            metrics="white",
        )
        super().__init__(refresh_rate=refresh_rate, leave=leave, theme=theme)
        
        self.start_time = None
        self.total_iterations = 0
        self.steps_per_epoch = 0
        self.log_every_n_steps = log_every_n_steps
        self.last_loss_log_step = -1
        self.cached_loss_info = ""
        
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Initialize training progress tracking with proper DINOv3 calculations"""
        # Only initialize from rank 0 to avoid DDP conflicts
        if trainer.global_rank != 0:
            return
            
        super().on_train_start(trainer, pl_module)
        
        self.start_time = time.time()
        
        # Calculate total iterations based on DINOv3 configuration
        try:
            if hasattr(pl_module.cfg.train, 'OFFICIAL_EPOCH_LENGTH'):
                self.steps_per_epoch = pl_module.cfg.train.OFFICIAL_EPOCH_LENGTH
                self.total_iterations = pl_module.cfg.optim.epochs * self.steps_per_epoch
            else:
                # Fallback calculation
                try:
                    dataset_size = len(trainer.datamodule.train_dataloader().dataset)
                    batch_size = pl_module.cfg.train.batch_size_per_gpu * trainer.world_size
                    self.steps_per_epoch = max(1, dataset_size // batch_size)
                    self.total_iterations = pl_module.cfg.optim.epochs * self.steps_per_epoch
                except:
                    self.steps_per_epoch = 1000
                    self.total_iterations = pl_module.cfg.optim.epochs * self.steps_per_epoch
        except:
            self.steps_per_epoch = 1000
            self.total_iterations = 30 * 1000  # Ultimate fallback
        
        # Ensure values are reasonable
        self.steps_per_epoch = max(1, min(self.steps_per_epoch, 100000))
        self.total_iterations = max(1, min(self.total_iterations, 10000000))
        
    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Update epoch progress with proper DINOv3 epoch calculation"""
        # Only update from rank 0 to avoid DDP conflicts
        if trainer.global_rank != 0:
            return
            
        # Call parent to ensure progress bar is initialized
        try:
            super().on_train_epoch_start(trainer, pl_module)
        except (AssertionError, AttributeError):
            pass
        
        if self.train_progress_bar_id is not None:
            # Calculate current DINOv3 epoch based on global steps
            current_step = trainer.global_step
            current_dinov3_epoch = (current_step // self.steps_per_epoch) + 1
            total_dinov3_epochs = pl_module.cfg.optim.epochs
            
            description = (f"Epoch {current_dinov3_epoch}/{total_dinov3_epochs} | "
                         f"Steps per epoch: {self.steps_per_epoch} | "
                         f"Total steps: {self.total_iterations}")
            self.progress.update(self.train_progress_bar_id, description=description)
    
    def on_train_batch_end(
        self, 
        trainer: "pl.Trainer", 
        pl_module: "pl.LightningModule", 
        outputs: Any, 
        batch: Any, 
        batch_idx: int
    ) -> None:
        """Update progress with detailed information and individual losses"""
        # Only update from rank 0 to avoid DDP conflicts
        if trainer.global_rank != 0:
            return
            
        # Call parent but catch any assertion errors gracefully
        try:
            super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        except (AssertionError, AttributeError):
            pass
        
        if self.train_progress_bar_id is not None:
            current_step = trainer.global_step
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            
            # Calculate DINOv3 epoch and step within epoch
            current_dinov3_epoch = (current_step // self.steps_per_epoch) + 1
            step_in_epoch = (current_step % self.steps_per_epoch) + 1
            total_dinov3_epochs = pl_module.cfg.optim.epochs
            
            # Calculate progress percentage within current epoch
            epoch_progress = (step_in_epoch / self.steps_per_epoch) * 100
            
            # Calculate speed (iterations per second)
            speed = current_step / elapsed_time if elapsed_time > 0 and current_step > 0 else 0.0
            
            # Calculate ETA
            remaining_steps = max(0, self.total_iterations - current_step)
            eta_seconds = remaining_steps / speed if speed > 0 and remaining_steps > 0 else 0
            
            # Ensure eta_seconds is reasonable
            if not isinstance(eta_seconds, (int, float)) or eta_seconds == float('inf') or eta_seconds > 999999:
                eta_seconds = 0
            
            # Format ETA
            if eta_seconds > 3600:
                eta_str = f"{eta_seconds/3600:.1f}h"
            elif eta_seconds > 60:
                eta_str = f"{eta_seconds/60:.1f}m"
            elif eta_seconds > 0:
                eta_str = f"{eta_seconds:.0f}s"
            else:
                eta_str = "N/A"
            
            # Format elapsed time
            if elapsed_time > 3600:
                elapsed_str = f"{elapsed_time/3600:.1f}h"
            elif elapsed_time > 60:
                elapsed_str = f"{elapsed_time/60:.1f}m"
            else:
                elapsed_str = f"{elapsed_time:.0f}s"
            
            # Format speed
            if speed > 1:
                speed_str = f"{speed:.2f} it/s"
            elif speed > 0:
                speed_str = f"{1/speed:.2f} s/it"
            else:
                speed_str = "0.00 it/s"
            
            # Get individual losses from the model if available - show every 5 steps for more frequent updates
            loss_info = ""
            if (hasattr(pl_module, 'current_loss_components') and 
                current_step - self.last_loss_log_step >= 5):
                losses = pl_module.current_loss_components
                if losses:
                    # Show main loss components
                    main_losses = []
                    if 'dino_local_crops_loss' in losses:
                        main_losses.append(f"DINO_L:{losses['dino_local_crops_loss']:.4f}")
                    if 'dino_global_crops_loss' in losses:
                        main_losses.append(f"DINO_G:{losses['dino_global_crops_loss']:.4f}")
                    if 'koleo_loss' in losses:
                        main_losses.append(f"KOLEO:{losses['koleo_loss']:.4f}")
                    if 'ibot_loss' in losses:
                        main_losses.append(f"IBOT:{losses['ibot_loss']:.4f}")
                    if 'gram_loss' in losses:
                        main_losses.append(f"GRAM:{losses['gram_loss']:.4f}")
                    
                    if main_losses:
                        loss_info = f" | {' '.join(main_losses)}"
                        # Store for persistent display
                        self.cached_loss_info = loss_info
                        
                self.last_loss_log_step = current_step
            elif hasattr(self, 'cached_loss_info'):
                # Use cached loss info if no new losses available
                loss_info = self.cached_loss_info
            
            # Build comprehensive description
            description = (
                f"Epoch {current_dinov3_epoch}/{total_dinov3_epochs} | "
                f"Step {step_in_epoch}/{self.steps_per_epoch} ({epoch_progress:.1f}%) | "
                f"ETA: {eta_str} | Speed: {speed_str} | Elapsed: {elapsed_str}"
                f"{loss_info}"
            )
            
            # Update progress bar
            self.progress.update(self.train_progress_bar_id, description=description)

    def configure_columns(self, trainer) -> list:
        """Configure progress bar columns"""
        return [
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(complete_style="rgb(165,66,129)", finished_style="rgb(165,66,129)"),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        ]