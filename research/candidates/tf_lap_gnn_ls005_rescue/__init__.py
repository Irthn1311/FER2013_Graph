"""Isolated label-smoothing 0.05 rescue adapter for frozen LAP-GNN."""

from .loss_adapter import (
    LABEL_SMOOTHING,
    NUM_CLASSES,
    hard_sparse_cross_entropy_compatibility,
    registered_smoothed_targets,
    smoothed_sparse_cross_entropy,
)

__all__ = [
    "LABEL_SMOOTHING",
    "NUM_CLASSES",
    "hard_sparse_cross_entropy_compatibility",
    "registered_smoothed_targets",
    "smoothed_sparse_cross_entropy",
]
