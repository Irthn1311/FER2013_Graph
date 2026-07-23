"""Portable PyTorch extraction of the locked OFIX7-mid LAP-GNN candidate."""

from lap_gnn.constants import PACKAGE_NAME, PACKAGE_STATUS
from lap_gnn.model.d16_model import D16Model

__all__ = ["D16Model", "PACKAGE_NAME", "PACKAGE_STATUS"]
__version__ = "0.1.0"
