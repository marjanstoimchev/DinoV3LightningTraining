#!/usr/bin/env python3
"""
Hugging Face dataset loader for DINOv3 training pipeline
Supports datasets from Hugging Face Hub with automatic split concatenation
Compatible with various Hugging Face dataset formats
"""

import logging
import os
import sys
from typing import Callable, Optional, Tuple, List, Union
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dinov3")

try:
    from datasets import load_dataset, DatasetDict, Dataset as HFDataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("Hugging Face datasets not available. Install with: pip install datasets")


class HuggingFaceDataset(Dataset):
    """
    Hugging Face dataset loader for DINOv3 training pipeline.
    
    Supports:
    - Automatic split concatenation (train, test, validation)
    - Various image formats
    - Flexible label handling
    - Streaming for large datasets
    
    Usage examples:
    - HuggingFace:name=jonathancui/oxford-pets
    - HuggingFace:name=imagenet-1k:split=train
    - HuggingFace:name=food101:streaming=true
    """
    
    def __init__(
        self,
        name: str,
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        streaming: bool = False,
        image_key: str = "image",
        label_key: str = "label",
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        """
        Initialize Hugging Face dataset loader.
        
        Args:
            name: Dataset name on Hugging Face Hub (e.g., "jonathancui/oxford-pets")
            split: Specific split to use. If None, concatenates available splits
            transform: Transform to apply to images
            target_transform: Transform to apply to targets
            streaming: Use streaming mode for large datasets
            image_key: Column name containing images (default: "image")
            label_key: Column name containing labels (default: "label") 
            cache_dir: Cache directory for downloaded datasets
            trust_remote_code: Allow remote code execution for custom datasets
        """
        if not HF_AVAILABLE:
            raise ImportError("Hugging Face datasets not available. Install with: pip install datasets")
            
        self.name = name
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.streaming = streaming
        self.image_key = image_key
        self.label_key = label_key
        
        logger.info(f"Loading Hugging Face dataset: {name}")
        if streaming:
            logger.info("Using streaming mode")
            
        # Load dataset
        try:
            if split:
                # Load specific split
                self.dataset = load_dataset(
                    name, 
                    split=split, 
                    streaming=streaming,
                    cache_dir=cache_dir,
                    trust_remote_code=trust_remote_code
                )
                logger.info(f"Loaded split '{split}'")
            else:
                # Load all splits and concatenate
                dataset_dict = load_dataset(
                    name, 
                    streaming=streaming,
                    cache_dir=cache_dir,
                    trust_remote_code=trust_remote_code
                )
                
                if isinstance(dataset_dict, DatasetDict):
                    # Get available splits
                    available_splits = list(dataset_dict.keys())
                    logger.info(f"Available splits: {available_splits}")
                    
                    # Concatenate all splits
                    if streaming:
                        # For streaming, we'll use the first available split
                        # Concatenation is more complex with streaming
                        first_split = available_splits[0]
                        self.dataset = dataset_dict[first_split]
                        logger.info(f"Using streaming from split '{first_split}' (concatenation not supported in streaming mode)")
                    else:
                        # Concatenate all splits for regular mode
                        from datasets import concatenate_datasets
                        datasets_to_concat = [dataset_dict[split] for split in available_splits]
                        self.dataset = concatenate_datasets(datasets_to_concat)
                        logger.info(f"Concatenated {len(available_splits)} splits: {available_splits}")
                else:
                    self.dataset = dataset_dict
                    
        except Exception as e:
            logger.error(f"Failed to load dataset {name}: {str(e)}")
            raise
            
        # Validate required columns
        if not streaming:
            columns = self.dataset.column_names
            logger.info(f"Dataset columns: {columns}")
            
            if image_key not in columns:
                # Try common alternatives
                image_alternatives = ["img", "images", "picture", "photo"]
                for alt in image_alternatives:
                    if alt in columns:
                        self.image_key = alt
                        logger.info(f"Using '{alt}' as image column")
                        break
                else:
                    raise ValueError(f"Image column '{image_key}' not found. Available columns: {columns}")
                    
            if label_key not in columns:
                # Try common alternatives or use dummy labels
                label_alternatives = ["labels", "class", "category", "target", "y"]
                for alt in label_alternatives:
                    if alt in columns:
                        self.label_key = alt
                        logger.info(f"Using '{alt}' as label column")
                        break
                else:
                    logger.warning(f"Label column '{label_key}' not found. Using dummy labels for self-supervised learning.")
                    self.label_key = None
        
        # For streaming datasets, we can't get the length
        if not streaming:
            self.length = len(self.dataset)
            logger.info(f"Dataset size: {self.length:,} samples")
        else:
            self.length = None
            logger.info("Streaming dataset - length unknown")
            
    def __len__(self) -> int:
        if self.length is not None:
            return self.length
        else:
            # For streaming datasets, return a large number
            return 1000000  # Arbitrary large number for streaming
            
    def __getitem__(self, idx: int) -> Tuple:
        try:
            if self.streaming:
                # For streaming, we need to iterate to the desired index
                # This is not efficient for random access, but works for training
                sample = None
                for i, sample in enumerate(self.dataset):
                    if i == idx:
                        break
                if sample is None:
                    raise IndexError(f"Index {idx} out of range for streaming dataset")
            else:
                sample = self.dataset[idx]
                
            # Extract image
            image_data = sample[self.image_key]
            
            # Handle different image formats
            if hasattr(image_data, 'convert'):
                # PIL Image
                image = image_data.convert('RGB')
            elif isinstance(image_data, np.ndarray):
                # NumPy array
                image = Image.fromarray(image_data).convert('RGB')
            elif isinstance(image_data, dict) and 'bytes' in image_data:
                # Image stored as bytes
                from io import BytesIO
                image = Image.open(BytesIO(image_data['bytes'])).convert('RGB')
            else:
                # Try to convert to PIL Image
                try:
                    image = Image.fromarray(np.array(image_data)).convert('RGB')
                except Exception as e:
                    logger.error(f"Failed to convert image data: {type(image_data)}, error: {e}")
                    raise
                    
            # Apply image transform
            if self.transform:
                image = self.transform(image)
                
            # Extract label
            if self.label_key and self.label_key in sample:
                target = sample[self.label_key]
                if isinstance(target, (list, tuple)):
                    target = target[0]  # Take first label if multiple
            else:
                # Use index as dummy target for self-supervised learning
                target = idx
                
            # Apply target transform
            if self.target_transform:
                target = self.target_transform(target)
                
            return image, target
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {str(e)}")
            # Return a dummy sample to avoid crashing training
            from PIL import Image
            dummy_image = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, idx
            
    def get_image_data(self, index: int) -> bytes:
        """Method expected by DINOv3 data loading pipeline"""
        # This is a fallback - the actual image data is handled in __getitem__
        return b''
        
    def get_target(self, index: int) -> int:
        """Method expected by DINOv3 data loading pipeline"""
        try:
            if self.streaming:
                # For streaming, we need to iterate to the desired index
                for i, sample in enumerate(self.dataset):
                    if i == index:
                        if self.label_key and self.label_key in sample:
                            target = sample[self.label_key]
                            return target if isinstance(target, int) else 0
                        return index
                return index
            else:
                sample = self.dataset[index]
                if self.label_key and self.label_key in sample:
                    target = sample[self.label_key]
                    return target if isinstance(target, int) else 0
                return index
        except:
            return index


# Convenience aliases for different dataset types
HuggingFaceVisionDataset = HuggingFaceDataset
HFDataset = HuggingFaceDataset