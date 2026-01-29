#!/usr/bin/env python3
"""
PyTorch Lightning DataModule for classification tasks.
Handles dataset loading for downstream classification (linear eval and fine-tuning).
"""

import sys
from pathlib import Path
from functools import partial
from typing import Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

# Add paths for DINOv3 modules and src modules
sys.path.append('dinov3')
sys.path.append(str(Path(__file__).parent.parent.parent))

from dinov3.data import make_dataset
from data.custom_dataset import CustomImageDataset
from data.csv_dataset import CSVDataset
from data.huggingface_dataset import HuggingFaceDataset


def classification_collate(batch):
    """
    Custom collate function for classification that properly converts labels to tensors.

    Handles both tuple format (image, label) and dict format {'image': ..., 'label': ...}.
    Ensures labels are converted to a proper tensor (handles numpy scalars, etc.).
    """
    if isinstance(batch[0], dict):
        # Dict format
        images = torch.stack([x["image"] for x in batch])
        labels = torch.tensor([int(x.get("label", -1)) for x in batch], dtype=torch.long)
        return {"images": images, "labels": labels}
    else:
        # Tuple format (image, label)
        images = torch.stack([x[0] for x in batch])
        labels = torch.tensor([int(x[1]) for x in batch], dtype=torch.long)
        return images, labels


class ClassificationDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for classification tasks.
    Handles dataset loading for downstream classification with proper label handling.
    """

    def __init__(
        self,
        cfg: OmegaConf,
        ssl_model=None,  # Not used for classification, but kept for interface compatibility
        sampler_type: Optional[str] = None,
        num_workers: int = 8,
        batch_size: int = 64,
    ):
        super().__init__()
        self.cfg = cfg
        self.ssl_model = ssl_model

        # Data loading parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_path = cfg.train.dataset_path if hasattr(cfg.train, 'dataset_path') else cfg.dataset_path

        # Sampler type override
        self.sampler_type_override = sampler_type

        # Initialize components
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets"""
        if stage == "fit" or stage is None:
            self._setup_train_val_datasets()
        elif stage == "test":
            self._setup_test_dataset()

    def _setup_train_val_datasets(self):
        """Setup training and validation datasets for classification"""
        # For classification, we need standard image transforms and label handling
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode

        # Training transforms with standard augmentation for classification
        train_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
        ])

        # Validation transforms - deterministic, no augmentation
        val_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
        ])

        # Handle different dataset types for classification
        if self.dataset_path.startswith("CustomTIFF:"):
            # Parse custom dataset path
            root_path = self.dataset_path.replace("CustomTIFF:root=", "")

            # For classification datasets, split into train/val/test (0.7/0.1/0.2)
            full_dataset = CustomImageDataset(
                root=root_path,
                transform=None,  # Apply separately for train/val/test
                target_transform=None,
            )

            # Split dataset
            total_size = len(full_dataset)
            train_size = int(0.7 * total_size)  # 70% for training
            val_size = int(0.1 * total_size)    # 10% for validation
            test_size = total_size - train_size - val_size  # 20% for test

            from torch.utils.data import random_split
            train_indices, val_indices, test_indices = random_split(
                range(total_size),
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )

            # Create datasets with transforms
            full_dataset_with_train_transform = CustomImageDataset(
                root=root_path,
                transform=train_transform,
                target_transform=None,
            )
            full_dataset_with_val_transform = CustomImageDataset(
                root=root_path,
                transform=val_transform,
                target_transform=None,
            )

            from torch.utils.data import Subset
            self.train_dataset = Subset(full_dataset_with_train_transform, train_indices.indices)
            self.val_dataset = Subset(full_dataset_with_val_transform, val_indices.indices)
            self.test_dataset = Subset(full_dataset_with_val_transform, test_indices.indices)

        elif self.dataset_path.startswith("HuggingFace:"):
            # Parse HuggingFace dataset path
            from datasets import load_dataset, DatasetDict

            # Parse the dataset string: HuggingFace:name=dataset_name[:split=split_name]
            params = {}
            parts = self.dataset_path.replace("HuggingFace:", "").split(":")

            for part in parts:
                if "=" in part:
                    key, value = part.split("=", 1)
                    params[key] = value

            # Extract parameters
            dataset_name = params.get("name")
            if not dataset_name:
                raise ValueError("HuggingFace dataset must specify 'name' parameter")

            split = params.get("split", None)  # Don't default to train for classification
            image_key = params.get("image_key", "image")
            label_key = params.get("label_key", "label")
            streaming = params.get("streaming", "false").lower() == "true"

            # Auto-detect available splits
            dataset_dict = load_dataset(dataset_name)

            if isinstance(dataset_dict, DatasetDict):
                available_splits = list(dataset_dict.keys())
                print(f"Available splits for {dataset_name}: {available_splits}")

                # Handle different split combinations
                if 'train' in available_splits and 'validation' in available_splits and 'test' in available_splits:
                    # All three splits available - use them as-is
                    print("Using existing train/validation/test splits")
                    self.train_dataset = HuggingFaceDataset(
                        name=dataset_name,
                        split='train',
                        transform=train_transform,
                        target_transform=None,
                        streaming=streaming,
                        image_key=image_key,
                        label_key=label_key
                    )
                    self.val_dataset = HuggingFaceDataset(
                        name=dataset_name,
                        split='validation',
                        transform=val_transform,
                        target_transform=None,
                        streaming=streaming,
                        image_key=image_key,
                        label_key=label_key
                    )
                    self.test_dataset = HuggingFaceDataset(
                        name=dataset_name,
                        split='test',
                        transform=val_transform,
                        target_transform=None,
                        streaming=streaming,
                        image_key=image_key,
                        label_key=label_key
                    )
                elif 'train' in available_splits and 'test' in available_splits:
                    # Train and test available - split train into train/val (0.9/0.1)
                    print("Using existing test split, splitting train into train/val (0.9/0.1)")

                    # Split the train dataset
                    full_train_dataset = HuggingFaceDataset(
                        name=dataset_name,
                        split='train',
                        transform=None,  # Apply later
                        target_transform=None,
                        streaming=streaming,
                        image_key=image_key,
                        label_key=label_key
                    )

                    total_size = len(full_train_dataset)
                    val_size = int(0.1 * total_size)  # 10% for validation
                    train_size = total_size - val_size  # 90% for training

                    from torch.utils.data import random_split
                    train_indices, val_indices = random_split(
                        range(total_size),
                        [train_size, val_size],
                        generator=torch.Generator().manual_seed(42)
                    )

                    # Create datasets with transforms
                    full_train_with_train_transform = HuggingFaceDataset(
                        name=dataset_name,
                        split='train',
                        transform=train_transform,
                        target_transform=None,
                        streaming=streaming,
                        image_key=image_key,
                        label_key=label_key
                    )
                    full_train_with_val_transform = HuggingFaceDataset(
                        name=dataset_name,
                        split='train',
                        transform=val_transform,
                        target_transform=None,
                        streaming=streaming,
                        image_key=image_key,
                        label_key=label_key
                    )

                    from torch.utils.data import Subset
                    self.train_dataset = Subset(full_train_with_train_transform, train_indices.indices)
                    self.val_dataset = Subset(full_train_with_val_transform, val_indices.indices)

                    # Use the existing test split as test dataset
                    self.test_dataset = HuggingFaceDataset(
                        name=dataset_name,
                        split='test',
                        transform=val_transform,
                        target_transform=None,
                        streaming=streaming,
                        image_key=image_key,
                        label_key=label_key
                    )
                elif 'train' in available_splits:
                    # Only train available - split into train/val/test (0.7/0.1/0.2)
                    print("Only train split available, splitting into train/val/test (0.7/0.1/0.2)")

                    full_train_dataset = HuggingFaceDataset(
                        name=dataset_name,
                        split='train',
                        transform=None,  # Apply later
                        target_transform=None,
                        streaming=streaming,
                        image_key=image_key,
                        label_key=label_key
                    )

                    total_size = len(full_train_dataset)
                    train_size = int(0.7 * total_size)  # 70% for training
                    val_size = int(0.1 * total_size)    # 10% for validation
                    test_size = total_size - train_size - val_size  # 20% for test

                    from torch.utils.data import random_split
                    train_indices, val_indices, test_indices = random_split(
                        range(total_size),
                        [train_size, val_size, test_size],
                        generator=torch.Generator().manual_seed(42)
                    )

                    # Create datasets with transforms
                    full_train_with_train_transform = HuggingFaceDataset(
                        name=dataset_name,
                        split='train',
                        transform=train_transform,
                        target_transform=None,
                        streaming=streaming,
                        image_key=image_key,
                        label_key=label_key
                    )
                    full_train_with_val_transform = HuggingFaceDataset(
                        name=dataset_name,
                        split='train',
                        transform=val_transform,
                        target_transform=None,
                        streaming=streaming,
                        image_key=image_key,
                        label_key=label_key
                    )

                    from torch.utils.data import Subset
                    self.train_dataset = Subset(full_train_with_train_transform, train_indices.indices)
                    self.val_dataset = Subset(full_train_with_val_transform, val_indices.indices)
                    self.test_dataset = Subset(full_train_with_val_transform, test_indices.indices)
                else:
                    # No standard splits, use first available split and split it (0.7/0.1/0.2)
                    first_split = available_splits[0]
                    print(f"No standard splits found, using '{first_split}' and splitting (0.7/0.1/0.2)")

                    full_dataset = HuggingFaceDataset(
                        name=dataset_name,
                        split=first_split,
                        transform=None,
                        target_transform=None,
                        streaming=streaming,
                        image_key=image_key,
                        label_key=label_key
                    )

                    total_size = len(full_dataset)
                    train_size = int(0.7 * total_size)
                    val_size = int(0.1 * total_size)
                    test_size = total_size - train_size - val_size

                    from torch.utils.data import random_split
                    train_indices, val_indices, test_indices = random_split(
                        range(total_size),
                        [train_size, val_size, test_size],
                        generator=torch.Generator().manual_seed(42)
                    )

                    full_with_train_transform = HuggingFaceDataset(
                        name=dataset_name,
                        split=first_split,
                        transform=train_transform,
                        target_transform=None,
                        streaming=streaming,
                        image_key=image_key,
                        label_key=label_key
                    )
                    full_with_val_transform = HuggingFaceDataset(
                        name=dataset_name,
                        split=first_split,
                        transform=val_transform,
                        target_transform=None,
                        streaming=streaming,
                        image_key=image_key,
                        label_key=label_key
                    )
                    full_with_test_transform = HuggingFaceDataset(
                        name=dataset_name,
                        split=first_split,
                        transform=val_transform,
                        target_transform=None,
                        streaming=streaming,
                        image_key=image_key,
                        label_key=label_key
                    )

                    from torch.utils.data import Subset
                    self.train_dataset = Subset(full_with_train_transform, train_indices.indices)
                    self.val_dataset = Subset(full_with_val_transform, val_indices.indices)
                    self.test_dataset = Subset(full_with_test_transform, test_indices.indices)
            else:
                # Single dataset, split it (0.7/0.1/0.2)
                total_size = len(dataset_dict)
                train_size = int(0.7 * total_size)
                val_size = int(0.1 * total_size)
                test_size = total_size - train_size - val_size

                from torch.utils.data import random_split
                train_indices, val_indices, test_indices = random_split(
                    range(total_size),
                    [train_size, val_size, test_size],
                    generator=torch.Generator().manual_seed(42)
                )

                full_dataset = HuggingFaceDataset(
                    name=dataset_name,
                    split=None,  # Will be handled by subset
                    transform=train_transform,
                    target_transform=None,
                    streaming=streaming,
                    image_key=image_key,
                    label_key=label_key
                )
                full_dataset_val = HuggingFaceDataset(
                    name=dataset_name,
                    split=None,
                    transform=val_transform,
                    target_transform=None,
                    streaming=streaming,
                    image_key=image_key,
                    label_key=label_key
                )
                full_dataset_test = HuggingFaceDataset(
                    name=dataset_name,
                    split=None,
                    transform=val_transform,
                    target_transform=None,
                    streaming=streaming,
                    image_key=image_key,
                    label_key=label_key
                )

                from torch.utils.data import Subset
                self.train_dataset = Subset(full_dataset, train_indices.indices)
                self.val_dataset = Subset(full_dataset_val, val_indices.indices)
                self.test_dataset = Subset(full_dataset_test, test_indices.indices)

        elif self.dataset_path.endswith(".csv") or self.dataset_path.startswith("CSV:"):
            # Handle CSV dataset - auto-detect .csv files or explicit CSV: prefix

            # Parse CSV dataset parameters
            if self.dataset_path.startswith("CSV:"):
                # Parse format: CSV:path=/path/to/file.csv[:image_col=col_name][:label_col=col_name][:sep=,][:base_path=/path]
                params = {}
                parts = self.dataset_path.replace("CSV:", "").split(":")

                for part in parts:
                    if "=" in part:
                        key, value = part.split("=", 1)
                        params[key] = value

                # Extract parameters
                csv_path = params.get("path")
                if not csv_path:
                    raise ValueError("CSV dataset must specify 'path' parameter")

                image_col = params.get("image_col", "image_path")
                label_col = params.get("label_col", None)
                separator = params.get("sep", ",")
                base_path = params.get("base_path", None)
            else:
                # Auto-detected CSV file - use defaults
                csv_path = self.dataset_path
                image_col = "image_path"
                label_col = None
                separator = ","
                base_path = None

            # For CSV datasets, split the CSV (0.7/0.1/0.2)
            import pandas as pd
            df = pd.read_csv(csv_path, sep=separator)

            # Shuffle the dataframe
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)

            # Split into train/val/test (0.7/0.1/0.2)
            total_size = len(df)
            train_size = int(0.7 * total_size)
            val_size = int(0.1 * total_size)
            test_size = total_size - train_size - val_size

            train_df = df.iloc[:train_size]
            val_df = df.iloc[train_size:train_size + val_size]
            test_df = df.iloc[train_size + val_size:]

            # Save temporary CSVs for train/val/test splits
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
                train_df.to_csv(tmp.name, index=False, sep=separator)
                train_csv_path = tmp.name

            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
                val_df.to_csv(tmp.name, index=False, sep=separator)
                val_csv_path = tmp.name

            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
                test_df.to_csv(tmp.name, index=False, sep=separator)
                test_csv_path = tmp.name

            self.train_dataset = CSVDataset(
                csv_path=train_csv_path,
                image_col=image_col,
                label_col=label_col,
                transform=train_transform,
                target_transform=None,
                separator=separator,
                skip_missing=True,
                base_path=base_path
            )

            self.val_dataset = CSVDataset(
                csv_path=val_csv_path,
                image_col=image_col,
                label_col=label_col,
                transform=val_transform,
                target_transform=None,
                separator=separator,
                skip_missing=True,
                base_path=base_path
            )

            self.test_dataset = CSVDataset(
                csv_path=test_csv_path,
                image_col=image_col,
                label_col=label_col,
                transform=val_transform,  # Use validation transform for test
                target_transform=None,
                separator=separator,
                skip_missing=True,
                base_path=base_path
            )

            # Register cleanup for temp files
            self.tmp_files = [train_csv_path, val_csv_path, test_csv_path]

        else:
            # Use DINOv3's make_dataset function for standard datasets
            # For classification, we need to handle train/val/test splits (0.7/0.1/0.2)
            full_dataset = make_dataset(
                dataset_str=self.dataset_path,
                transform=None,  # Apply separately for train/val/test
                target_transform=None,
            )

            # Split dataset into train/val/test (0.7/0.1/0.2)
            total_size = len(full_dataset)
            train_size = int(0.7 * total_size)  # 70% for training
            val_size = int(0.1 * total_size)    # 10% for validation
            test_size = total_size - train_size - val_size  # 20% for test

            from torch.utils.data import random_split
            train_indices, val_indices, test_indices = random_split(
                range(total_size),
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )

            # Create datasets with appropriate transforms
            full_dataset_with_train_transform = make_dataset(
                dataset_str=self.dataset_path,
                transform=train_transform,
                target_transform=None,
            )
            full_dataset_with_val_transform = make_dataset(
                dataset_str=self.dataset_path,
                transform=val_transform,
                target_transform=None,
            )
            full_dataset_with_test_transform = make_dataset(
                dataset_str=self.dataset_path,
                transform=val_transform,  # Use validation transform for test
                target_transform=None,
            )

            # Use Subset to create the split datasets
            from torch.utils.data import Subset
            self.train_dataset = Subset(full_dataset_with_train_transform, train_indices.indices)
            self.val_dataset = Subset(full_dataset_with_val_transform, val_indices.indices)
            self.test_dataset = Subset(full_dataset_with_test_transform, test_indices.indices)

        print(f"Train dataset setup complete. Found {len(self.train_dataset)} samples.")
        print(f"Val dataset setup complete. Found {len(self.val_dataset)} samples.")
        if self.test_dataset:
            print(f"Test dataset setup complete. Found {len(self.test_dataset)} samples.")

    def _setup_test_dataset(self):
        """Setup test dataset"""
        # For test dataset, use validation transform (no augmentation)
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode

        test_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
        ])

        if self.dataset_path.startswith("HuggingFace:"):
            from data.huggingface_dataset import HuggingFaceDataset

            # Parse the dataset string
            params = {}
            parts = self.dataset_path.replace("HuggingFace:", "").split(":")

            for part in parts:
                if "=" in part:
                    key, value = part.split("=", 1)
                    params[key] = value

            dataset_name = params.get("name")
            split = params.get("split", "test")  # Default to test for testing
            image_key = params.get("image_key", "image")
            label_key = params.get("label_key", "label")
            streaming = params.get("streaming", "false").lower() == "true"

            self.test_dataset = HuggingFaceDataset(
                name=dataset_name,
                split=split,
                transform=test_transform,
                target_transform=None,
                streaming=streaming,
                image_key=image_key,
                label_key=label_key
            )

    def train_dataloader(self):
        """Create training dataloader using standard PyTorch DataLoader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,  # Standard DataLoader handles shuffling
            drop_last=True,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=classification_collate,
        )

    def val_dataloader(self):
        """Create validation dataloader using standard PyTorch DataLoader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,  # No shuffling for validation
            drop_last=False,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=classification_collate,
        )

    def test_dataloader(self):
        """Create test dataloader using standard PyTorch DataLoader"""
        if self.test_dataset is None:
            return None

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,  # No shuffling for test
            drop_last=False,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=classification_collate,
        )


# Backward compatibility alias
DINOv3ClassificationDataModule = ClassificationDataModule
