#!/usr/bin/env python3
"""
Custom dataset loader for various image formats
Compatible with DINOv3 training pipeline
Supports TIFF, PNG, JPG, JPEG formats with flexible directory structure
"""

import os
import glob
from typing import Callable, Optional, Tuple, List
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    """
    Custom dataset class for various image formats (TIFF, PNG, JPG, JPEG)
    Supports flexible directory structures:
    - Single directory with images
    - Parent directory with subdirectories containing images
    """
    
    def __init__(
        self,
        root: str = "../Datasets/composite/SLIDE-0018/",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        supported_extensions: Tuple[str, ...] = ('.tiff', '.tif', '.png', '.jpg', '.jpeg'),
        recursive: bool = True,
    ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.supported_extensions = tuple(ext.lower() for ext in supported_extensions)
        self.recursive = recursive
        
        # Find all supported image files
        self.image_paths = self._find_images()
        self.image_paths.sort()  # Ensure consistent ordering
        
        print(f"Found {len(self.image_paths)} images with extensions {self.supported_extensions} in {root}")
        if len(self.image_paths) == 0:
            print(f"Warning: No images found in {root} with supported extensions {self.supported_extensions}")
        
    def __len__(self) -> int:
        return len(self.image_paths)
        
    def __getitem__(self, idx: int) -> Tuple:
        img_path = self.image_paths[idx]
        
        # Load TIFF image and convert to RGB
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        # For self-supervised learning, target is usually just the index or dummy
        target = idx
        if self.target_transform:
            target = self.target_transform(target)
            
        return image, target

    def _find_images(self) -> List[str]:
        """Find all supported image files in the root directory."""
        image_paths = []
        
        if self.recursive:
            # Search recursively in subdirectories
            for ext in self.supported_extensions:
                pattern = os.path.join(self.root, "**", f"*{ext}")
                image_paths.extend(glob.glob(pattern, recursive=True))
                # Also search for uppercase extensions
                pattern = os.path.join(self.root, "**", f"*{ext.upper()}")
                image_paths.extend(glob.glob(pattern, recursive=True))
        else:
            # Search only in the specified directory
            for ext in self.supported_extensions:
                pattern = os.path.join(self.root, f"*{ext}")
                image_paths.extend(glob.glob(pattern))
                # Also search for uppercase extensions
                pattern = os.path.join(self.root, f"*{ext.upper()}")
                image_paths.extend(glob.glob(pattern))
        
        return list(set(image_paths))  # Remove duplicates
    
    def get_image_paths(self) -> List[str]:
        """Return all image paths for external use (e.g., feature extraction)."""
        return self.image_paths.copy()

    def get_image_data(self, index: int) -> bytes:
        """Method expected by DINOv3 data loading pipeline"""
        img_path = self.image_paths[index]
        with open(img_path, 'rb') as f:
            return f.read()
            
    def get_target(self, index: int) -> int:
        """Method expected by DINOv3 data loading pipeline"""
        return index  # For self-supervised learning, target is just index


# Backward compatibility alias
CustomTIFFDataset = CustomImageDataset