"""Cumulative ablation ladder (7 configurations) for MPG-FER."""

from .model import (
    CUMULATIVE_ABLATION_ORDER,
    CumulativeAblationMode,
    CumulativeAblationMPGFER,
    CumulativeAblationSpec,
    cumulative_registry_document,
)

__all__ = [
    "CUMULATIVE_ABLATION_ORDER",
    "CumulativeAblationMode",
    "CumulativeAblationMPGFER",
    "CumulativeAblationSpec",
    "cumulative_registry_document",
]
