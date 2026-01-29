"""
Models module - provides backward compatibility imports.

The models have been reorganized into:
- src/ssl/models/ - SSL pretraining models
- src/classification/models/ - Classification downstream models
- src/classification/learners/ - Classification Lightning modules

This file provides backward compatibility aliases.
"""

# Backward compatibility imports - redirect to new locations
from src.ssl.models.ssl_learner import SSLLearner, DINOv3LightningModule
from src.ssl.data.datamodule import SSLDataModule, MultiResolutionSSLDataModule, DINOv3DataModule, MultiResolutionDINOv3DataModule
from src.classification.models.linear_classifier import LinearClassifier, DINOv3LinearClassifier
from src.classification.learners.classification_learner import ClassificationLearner
from src.classification.data.datamodule import ClassificationDataModule, DINOv3ClassificationDataModule

__all__ = [
    # SSL (backward compatibility)
    'DINOv3LightningModule',
    'DINOv3DataModule',
    'MultiResolutionDINOv3DataModule',
    # SSL (new names)
    'SSLLearner',
    'SSLDataModule',
    'MultiResolutionSSLDataModule',
    # Classification (backward compatibility)
    'DINOv3LinearClassifier',
    'DINOv3ClassificationDataModule',
    # Classification (new names)
    'LinearClassifier',
    'ClassificationLearner',
    'ClassificationDataModule',
]
