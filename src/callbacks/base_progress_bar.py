#!/usr/bin/env python3
"""
Base progress bar for DINOv3 Lightning training with distributed samplers
Shows individual losses, proper epoch-based counting, ETA, and speed
Designed to work with distributed samplers that have known dataset lengths
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


class DINOv3BaseProgressBar(RichProgressBar):
    """Base progress bar for DINOv3 with distributed samplers and epoch-based progress"""
    
    def __init__(self, refresh_rate: int = 1, leave: bool = False, log_every_n_steps: int = 50):
        # Custom theme matching enhanced_progress_bar colors
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
        self.epoch_start_time = None
        self.steps_per_epoch = 0
        self.log_every_n_steps = log_every_n_steps
        self.last_loss_log_step = -1
        self.cached_loss_info = ""
        
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Initialize training progress tracking with distributed sampler calculations"""
        # Only initialize from rank 0 to avoid DDP conflicts
        if trainer.global_rank != 0:
            return
            
        super().on_train_start(trainer, pl_module)
        
        self.start_time = time.time()
        
        # Calculate steps per epoch based on actual dataset size and distributed setup
        try:
            if hasattr(trainer.datamodule, 'train_dataloader'):
                dataloader = trainer.datamodule.train_dataloader()
                # For distributed samplers, the dataloader length gives us steps per epoch
                self.steps_per_epoch = len(dataloader)
                print(f"Detected steps per epoch from distributed dataloader: {self.steps_per_epoch}")
            else:
                # Fallback calculation
                dataset_size = len(trainer.datamodule.train_dataset) if hasattr(trainer.datamodule, 'train_dataset') else 1000
                batch_size = pl_module.cfg.train.batch_size_per_gpu * trainer.world_size
                self.steps_per_epoch = max(1, dataset_size // batch_size)
                print(f"Calculated steps per epoch: {self.steps_per_epoch} (dataset: {dataset_size}, batch_size: {batch_size})")
        except Exception as e:
            print(f"Failed to calculate steps per epoch: {e}, using fallback")
            self.steps_per_epoch = 100  # Fallback
        
        # Ensure reasonable values
        self.steps_per_epoch = max(1, min(self.steps_per_epoch, 100000))
        
    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Update epoch progress for distributed sampler"""
        # Only update from rank 0 to avoid DDP conflicts
        if trainer.global_rank != 0:
            return
            
        # Call parent to ensure progress bar is initialized
        try:
            super().on_train_epoch_start(trainer, pl_module)
        except (AssertionError, AttributeError):
            pass
        
        self.epoch_start_time = time.time()
        
        if self.train_progress_bar_id is not None:
            # Update progress bar description with current epoch
            current_epoch = trainer.current_epoch + 1
            max_epochs = trainer.max_epochs if trainer.max_epochs > 0 else pl_module.cfg.optim.epochs
            
            description = f"[bold blue]Epoch {current_epoch}/{max_epochs}[/bold blue]"
            self.progress.update(self.train_progress_bar_id, description=description, total=self.steps_per_epoch)
            self.progress.reset(self.train_progress_bar_id)
            
    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Update progress with loss information for distributed training"""
        # Only update from rank 0 to avoid DDP conflicts
        if trainer.global_rank != 0:
            return
            
        # Call parent to ensure progress bar update
        try:
            super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        except (AssertionError, AttributeError):
            pass
        
        # Update step count
        current_step = batch_idx + 1
        
        # Log losses every step to keep them current
        should_log = True  # Always update loss display
        
        if should_log and self.train_progress_bar_id is not None:
            try:
                # Get loss information from trainer logs
                loss_info = self._format_loss_info(trainer)
                self.cached_loss_info = loss_info
                self.last_loss_log_step = current_step
                
                # Calculate speeds and times
                elapsed_time = time.time() - self.start_time if self.start_time else 0
                epoch_elapsed = time.time() - self.epoch_start_time if self.epoch_start_time else 0
                
                # Calculate iteration speed
                if epoch_elapsed > 0 and current_step > 0:
                    it_per_sec = current_step / epoch_elapsed
                    speed_info = f"[grey70]({it_per_sec:.2f}it/s)[/grey70]"
                else:
                    speed_info = ""
                
                # Update progress bar postfix with losses and speed
                postfix = f"{loss_info} {speed_info}".strip()
                if postfix:
                    self.progress.update(self.train_progress_bar_id, description=f"[bold blue]Epoch {trainer.current_epoch + 1}/{trainer.max_epochs or 'N/A'}[/bold blue] {postfix}")
                    
            except Exception as e:
                pass  # Don't let progress bar errors break training
    
    def _format_loss_info(self, trainer: "pl.Trainer") -> str:
        """Format loss information from trainer logs with consistent display"""
        try:
            logged_metrics = trainer.logged_metrics
            loss_parts = []
            
            # Show individual DINO losses (more informative than total)
            if 'train/dino_local_crops_loss' in logged_metrics:
                dino_local = float(logged_metrics['train/dino_local_crops_loss'])
                loss_parts.append(f"[#A23B72]Dino_L: {dino_local:.4f}[/#A23B72]")
            else:
                loss_parts.append(f"[#A23B72]Dino_L: ---.----[/#A23B72]")
                
            if 'train/dino_global_crops_loss' in logged_metrics:
                dino_global = float(logged_metrics['train/dino_global_crops_loss'])
                loss_parts.append(f"[#F18F01]Dino_G: {dino_global:.4f}[/#F18F01]")
            else:
                loss_parts.append(f"[#F18F01]Dino_G: ---.----[/#F18F01]")
            
            # Always show Koleo and iBOT (they should be computed from start in DINOv3)
            if 'train/koleo_loss' in logged_metrics:
                koleo = float(logged_metrics['train/koleo_loss'])
                loss_parts.append(f"[#C73E1D]Koleo: {koleo:.4f}[/#C73E1D]")
            else:
                loss_parts.append(f"[#C73E1D]Koleo: ---.----[/#C73E1D]")
                
            if 'train/ibot_loss' in logged_metrics:
                ibot = float(logged_metrics['train/ibot_loss'])
                loss_parts.append(f"[#7209B7]iBOT: {ibot:.4f}[/#7209B7]")
            else:
                loss_parts.append(f"[#7209B7]iBOT: ---.----[/#7209B7]")
            
            # Return all main losses (consistent display)
            return " | ".join(loss_parts)
            
        except Exception:
            return self.cached_loss_info if hasattr(self, 'cached_loss_info') else "[white]Loss: ---.----[/white]"
    
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Complete epoch progress for distributed training"""
        # Only update from rank 0 to avoid DDP conflicts
        if trainer.global_rank != 0:
            return
            
        # Call parent
        try:
            super().on_train_epoch_end(trainer, pl_module)
        except (AssertionError, AttributeError):
            pass
        
        # Ensure progress bar is completed for this epoch
        if self.train_progress_bar_id is not None:
            self.progress.update(self.train_progress_bar_id, advance=0)
            
        # Show final epoch summary
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            avg_it_per_sec = self.steps_per_epoch / epoch_time if epoch_time > 0 else 0
            
            current_epoch = trainer.current_epoch + 1
            max_epochs = trainer.max_epochs if trainer.max_epochs > 0 else pl_module.cfg.optim.epochs
            
            final_loss_info = self._format_loss_info(trainer)
            print(f"Epoch {current_epoch}/{max_epochs} completed in {epoch_time:.1f}s ({avg_it_per_sec:.2f}it/s) | {final_loss_info}")

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Show training completion summary"""
        # Only from rank 0
        if trainer.global_rank != 0:
            return
            
        super().on_train_end(trainer, pl_module)
        
        if self.start_time:
            total_time = time.time() - self.start_time
            print(f"\n✓ Training completed in {total_time:.1f}s")