#!/usr/bin/env python3
"""
Enhanced ModelCheckpoint callback for DINOv3 training
Supports step-based checkpointing with detailed naming
"""

import os
import math
import time
from typing import Any, Dict, Optional, Union

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer


class DINOv3ModelCheckpoint(ModelCheckpoint):
    """
    Simple ModelCheckpoint for DINOv3 using Lightning's native saving with custom filename
    """
    
    
    def __init__(
        self,
        dirpath: Optional[Union[str, os.PathLike]] = None,
        filename: Optional[str] = "model_epoch_{epoch:02d}_step_{step:06d}_loss_{total_loss:.6f}",
        monitor: Optional[str] = "total_loss",
        verbose: bool = False,
        save_last: Optional[bool] = True,
        save_top_k: int = 3,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: Optional[int] = None,
        train_time_interval: Optional[Union[int, float]] = None,
        save_on_train_epoch_end: Optional[bool] = None,
        **kwargs
    ):
            
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=auto_insert_metric_name,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
            save_on_train_epoch_end=save_on_train_epoch_end,
            **kwargs
        )
        
        if verbose and hasattr(torch.distributed, 'is_available') and torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                print(f"DINOv3ModelCheckpoint initialized:", flush=True)
                print(f"  dirpath: {dirpath}", flush=True)
                print(f"  filename: {filename}", flush=True)
                print(f"  every_n_train_steps: {every_n_train_steps}", flush=True)
                print(f"  save_top_k: {save_top_k}", flush=True)
                print(f"  monitor: {monitor}", flush=True)