#!/usr/bin/env python3
"""
CSV dataset loader for DINOv3 training pipeline
Supports datasets specified in CSV files with image paths and optional labels
Compatible with various image formats
"""

import os
import logging
import pandas as pd
from typing import Callable, Optional, Tuple, List
from PIL import Image
from torch.utils.data import Dataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dinov3")


class CSVDataset(Dataset):
    """
    CSV dataset loader for DINOv3 training pipeline.

    Loads images from paths specified in a CSV file with optional label columns.

    CSV format example:
        magnification,image_path
        10X,/path/to/image1.tiff
        20X,/path/to/image2.tiff

    Usage examples:
    - CSV:path=patches.csv
    - CSV:path=/path/to/data.csv:image_col=image_path:label_col=magnification
    - CSV:path=data.csv:image_col=file_path:sep=;
    """

    def __init__(
        self,
        csv_path: str,
        image_col: str = "image_path",
        label_col: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        separator: str = ",",
        skip_missing: bool = True,
        base_path: Optional[str] = None,
    ):
        """
        Initialize CSV dataset loader.

        Args:
            csv_path: Path to CSV file containing image paths
            image_col: Column name containing image file paths (default: "image_path")
            label_col: Column name containing labels. If None, uses index (default: None)
            transform: Transform to apply to images
            target_transform: Transform to apply to targets
            separator: CSV separator character (default: ",")
            skip_missing: Skip missing/invalid image files instead of raising error (default: True)
            base_path: Base directory for resolving relative image paths.
                      If None, paths are resolved relative to CSV file location (default: None)
        """
        self.csv_path = csv_path
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform
        self.separator = separator
        self.skip_missing = skip_missing

        # Resolve CSV path
        if not os.path.isabs(csv_path):
            # Make path relative to current working directory
            csv_path = os.path.abspath(csv_path)

        # Set base path for resolving relative image paths
        if base_path is None:
            # Default: resolve relative to CSV file location
            self.base_path = os.path.dirname(csv_path)
        else:
            self.base_path = os.path.abspath(base_path)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        logger.info(f"Loading CSV dataset from: {csv_path}")

        # Load CSV file
        try:
            self.df = pd.read_csv(csv_path, sep=separator)
            logger.info(f"Loaded CSV with {len(self.df)} rows")
            logger.info(f"Available columns: {list(self.df.columns)}")
        except Exception as e:
            raise ValueError(f"Failed to load CSV file {csv_path}: {str(e)}")

        # Validate image column exists
        if image_col not in self.df.columns:
            raise ValueError(
                f"Image column '{image_col}' not found. "
                f"Available columns: {list(self.df.columns)}"
            )

        # Validate label column if specified
        if label_col and label_col not in self.df.columns:
            logger.warning(
                f"Label column '{label_col}' not found. "
                f"Available columns: {list(self.df.columns)}. "
                f"Using dummy labels for self-supervised learning."
            )
            self.label_col = None

        # Filter out rows with missing image paths
        initial_count = len(self.df)
        self.df = self.df[self.df[image_col].notna()]
        if len(self.df) < initial_count:
            logger.warning(f"Removed {initial_count - len(self.df)} rows with missing image paths")

        # Validate image files exist and resolve paths
        if skip_missing:
            valid_indices = []
            missing_count = 0
            for idx, row in self.df.iterrows():
                img_path = row[image_col]

                # Resolve relative paths
                if not os.path.isabs(img_path):
                    img_path = os.path.join(self.base_path, img_path)

                # Normalize path (resolve .. and .)
                img_path = os.path.normpath(img_path)

                if os.path.exists(img_path):
                    valid_indices.append(idx)
                else:
                    missing_count += 1
                    if missing_count <= 5:  # Log first 5 missing files
                        logger.warning(f"Image file not found: {img_path}")

            if missing_count > 0:
                logger.warning(f"Total missing files: {missing_count}/{initial_count}")
                self.df = self.df.loc[valid_indices].reset_index(drop=True)
                logger.info(f"Using {len(self.df)} valid images")

        # Create label mapping if using categorical labels
        self.label_mapping = None
        if self.label_col:
            unique_labels = self.df[self.label_col].unique()
            logger.info(f"Found {len(unique_labels)} unique labels in column '{self.label_col}'")

            # Create mapping for categorical labels
            if not pd.api.types.is_numeric_dtype(self.df[self.label_col]):
                self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
                logger.info(f"Created label mapping: {self.label_mapping}")

        logger.info(f"Dataset initialized with {len(self.df)} images")

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, img_path: str) -> str:
        """Resolve image path relative to base_path if it's a relative path."""
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.base_path, img_path)
        return os.path.normpath(img_path)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Get image and target at specified index.

        Returns:
            Tuple of (image, target) where image is a PIL Image or transformed tensor
        """
        try:
            row = self.df.iloc[idx]
            img_path = self._resolve_path(row[self.image_col])

            # Load image
            image = Image.open(img_path).convert('RGB')

            # Apply image transform
            if self.transform:
                image = self.transform(image)

            # Get target/label
            if self.label_col:
                target = row[self.label_col]
                # Map categorical labels to integers
                if self.label_mapping:
                    target = self.label_mapping[target]
                # Ensure target is int
                if not isinstance(target, int):
                    try:
                        target = int(target)
                    except:
                        target = idx
            else:
                # Use index as dummy target for self-supervised learning
                target = idx

            # Apply target transform
            if self.target_transform:
                target = self.target_transform(target)

            return image, target

        except Exception as e:
            logger.error(f"Error loading image at index {idx}: {str(e)}")
            if idx < len(self.df):
                logger.error(f"Image path: {self.df.iloc[idx][self.image_col]}")

            # Return a dummy sample to avoid crashing training
            dummy_image = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, idx

    def get_image_data(self, index: int) -> bytes:
        """
        Method expected by DINOv3 data loading pipeline.
        Returns raw image bytes.
        """
        try:
            img_path = self._resolve_path(self.df.iloc[index][self.image_col])
            with open(img_path, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading image data at index {index}: {str(e)}")
            return b''

    def get_target(self, index: int) -> int:
        """
        Method expected by DINOv3 data loading pipeline.
        Returns the target/label for the specified index.
        """
        try:
            if self.label_col:
                target = self.df.iloc[index][self.label_col]
                # Map categorical labels to integers
                if self.label_mapping:
                    target = self.label_mapping[target]
                # Ensure target is int
                if not isinstance(target, int):
                    try:
                        target = int(target)
                    except:
                        target = index
                return target
            else:
                return index
        except:
            return index

    def get_image_paths(self) -> List[str]:
        """Return all image paths for external use (e.g., feature extraction)."""
        return self.df[self.image_col].tolist()

    def get_labels(self) -> Optional[List]:
        """Return all labels if label column is specified."""
        if self.label_col:
            return self.df[self.label_col].tolist()
        return None

    def get_dataframe(self) -> pd.DataFrame:
        """Return the underlying DataFrame for advanced use cases."""
        return self.df.copy()


# Convenience alias
CSVImageDataset = CSVDataset
