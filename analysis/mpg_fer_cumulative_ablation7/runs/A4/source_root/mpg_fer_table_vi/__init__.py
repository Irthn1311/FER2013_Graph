"""Canonical seven-variant MPG-FER Table VI framework (Issue #101)."""

from .model import ABLATION_REGISTRY, AblationMode, AblationMPGFER
from .protocol import AblationConfig

__all__ = [
    "ABLATION_REGISTRY",
    "AblationMode",
    "AblationMPGFER",
    "AblationConfig",
]
