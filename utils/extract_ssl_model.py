#!/usr/bin/env python3
"""
Utility script to extract SSL model state dict from Lightning checkpoints
Usage: python extract_ssl_model.py <checkpoint_path>
"""

import os
import sys
import torch
from pathlib import Path

def extract_ssl_model_from_checkpoint(checkpoint_path: str, output_path: str = None):
    """Extract SSL model state dict from Lightning checkpoint"""
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return False
    
    try:
        # Load Lightning checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract SSL model state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            
            # Filter for SSL model parameters (remove 'ssl_model.' prefix)
            ssl_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('ssl_model.'):
                    ssl_key = key[len('ssl_model.'):]
                    ssl_state_dict[ssl_key] = value
            
            # Determine output path
            if output_path is None:
                output_path = checkpoint_path.replace('.pth', '_ssl_model.pth')
            
            # Save SSL model state dict
            torch.save(ssl_state_dict, output_path)
            print(f"SSL model extracted to: {output_path}")
            return True
            
        else:
            print("Error: No state_dict found in checkpoint")
            return False
            
    except Exception as e:
        print(f"Error extracting SSL model: {e}")
        return False

def extract_all_checkpoints_in_dir(checkpoint_dir: str):
    """Extract SSL models from all checkpoints in directory"""
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        print(f"Error: Directory not found: {checkpoint_dir}")
        return
    
    # Find all .pth files that don't already have SSL model extracted
    checkpoint_files = []
    for pth_file in checkpoint_dir.glob("*.pth"):
        if not str(pth_file).endswith('_ssl_model.pth'):
            ssl_file = str(pth_file).replace('.pth', '_ssl_model.pth')
            if not os.path.exists(ssl_file):
                checkpoint_files.append(pth_file)
    
    if not checkpoint_files:
        print("No checkpoints found that need SSL model extraction")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoints to process:")
    for ckpt_file in checkpoint_files:
        print(f"  {ckpt_file.name}")
    
    # Extract SSL models
    success_count = 0
    for ckpt_file in checkpoint_files:
        if extract_ssl_model_from_checkpoint(str(ckpt_file)):
            success_count += 1
    
    print(f"\nExtracted SSL models from {success_count}/{len(checkpoint_files)} checkpoints")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python extract_ssl_model.py <checkpoint_path>        # Extract single checkpoint")
        print("  python extract_ssl_model.py <checkpoint_directory>   # Extract all checkpoints in dir")
        sys.exit(1)
    
    path = sys.argv[1]
    
    if os.path.isfile(path):
        # Single checkpoint
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        extract_ssl_model_from_checkpoint(path, output_path)
    elif os.path.isdir(path):
        # Directory of checkpoints
        extract_all_checkpoints_in_dir(path)
    else:
        print(f"Error: Path not found: {path}")
        sys.exit(1)