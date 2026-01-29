"""
Classification downstream task module.
Contains linear evaluation and fine-tuning components.
"""

from src.classification.models.linear_classifier import LinearClassifier, DINOv3LinearClassifier
from src.classification.learners.classification_learner import ClassificationLearner
from src.classification.data.datamodule import ClassificationDataModule, DINOv3ClassificationDataModule

__all__ = [
    "LinearClassifier",
    "DINOv3LinearClassifier",  # Backward compatibility
    "ClassificationLearner",
    "ClassificationDataModule",
    "DINOv3ClassificationDataModule",  # Backward compatibility
]
