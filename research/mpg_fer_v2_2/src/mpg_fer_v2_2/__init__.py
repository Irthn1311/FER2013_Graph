"""Root package for the Issue #95 MPG-FER v2.2 implementation."""

from __future__ import annotations

from .config import MPGConfig
from .features import PixelFeatureExtractor
from .graph import PixelGraphTopology
from .motif import SpatialMotifComposer
from .model import MPGFER
from .ema import ModelEMA

__all__ = [
    "MPGConfig",
    "PixelFeatureExtractor",
    "PixelGraphTopology",
    "SpatialMotifComposer",
    "MPGFER",
    "ModelEMA",
]
