#!/usr/bin/env python3
"""
Show live training status with individual losses and progress
"""
import pandas as pd
import time
import os
import sys
from pathlib import Path

def get_latest_metrics():
    """Get the latest training metrics"""
    csv_path = "output_multi_gpu/csv_logs/metrics.csv"
    
    if not os.path.exists(csv_path):
        return None
        
    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            return None
            
        # Get the latest row
        latest = df.iloc[-1]
        return latest
    except Exception as e:
        return None

def format_loss(value):
    """Format loss values nicely"""
    try:
        if pd.isna(value) or value == "":
            return "N/A"
        return f"{float(value):.4f}"
    except:
        return "N/A"

def format_lr(value):
    """Format learning rate nicely"""
    try:
        if pd.isna(value) or value == "":
            return "N/A"
        return f"{float(value):.2e}"
    except:
        return "N/A"

def show_status():
    """Show current training status"""
    metrics = get_latest_metrics()
    
    if metrics is None:
        print("No training metrics found. Make sure training is running.")
        return
        
    # Calculate epoch info
    step = int(metrics.get('step', 0))
    epoch_length = 200  # OFFICIAL_EPOCH_LENGTH from config
    current_epoch = (step // epoch_length) + 1
    step_in_epoch = (step % epoch_length) + 1
    total_epochs = 30
    
    # Calculate progress percentage
    epoch_progress = (step_in_epoch / epoch_length) * 100
    overall_progress = (step / (total_epochs * epoch_length)) * 100
    
    print(f"\nDINOv3 Training Status - {time.strftime('%H:%M:%S')}")
    print(f"-" * 60)
    
    # Epoch and step info
    print(f"Epoch {current_epoch}/{total_epochs} | Step {step_in_epoch}/{epoch_length} ({epoch_progress:.1f}%)")
    print(f"Overall Progress: {overall_progress:.1f}% (Step {step}/{total_epochs * epoch_length})")
    
    # Main losses
    total_loss = format_loss(metrics.get('total_loss'))
    dino_local = format_loss(metrics.get('train/dino_local_crops_loss'))
    dino_global = format_loss(metrics.get('train/dino_global_crops_loss'))
    koleo_loss = format_loss(metrics.get('train/koleo_loss'))
    ibot_loss = format_loss(metrics.get('train/ibot_loss'))
    
    print(f"\nLosses:")
    print(f"   Total Loss:       {total_loss}")
    print(f"   DINO Global:      {dino_global}")
    print(f"   DINO Local:       {dino_local}")
    print(f"   KOLEO:            {koleo_loss}")
    print(f"   IBOT:             {ibot_loss}")
    
    # Training hyperparameters
    lr = format_lr(metrics.get('train/lr'))
    momentum = format_loss(metrics.get('train/momentum'))
    teacher_temp = format_loss(metrics.get('train/teacher_temp'))
    wd = format_lr(metrics.get('train/wd'))
    
    print(f"\nHyperparameters:")
    print(f"   Learning Rate:    {lr}")
    print(f"   Momentum:         {momentum}")
    print(f"   Teacher Temp:     {teacher_temp}")
    print(f"   Weight Decay:     {wd}")
    
    # Batch info
    batch_size = metrics.get('train/global_batch_size', 32)
    print(f"\nBatch Size: {batch_size}")
    
    print(f"-" * 60)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--watch":
        # Watch mode - continuously update
        try:
            while True:
                os.system('clear' if os.name != 'nt' else 'cls')
                show_status()
                time.sleep(2)
        except KeyboardInterrupt:
            print("\nStopped monitoring")
    else:
        # Single status check
        show_status()