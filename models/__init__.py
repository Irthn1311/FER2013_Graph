"""D5/D6 model package."""

import sys
from pathlib import Path

_PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))

from models.class_pixel_motif_graph_retrieval import ClassPixelMotifGraphRetrieval
from models.d10_slot_motif_model import D10SlotMotifModel
from models.d11_global_local_model import D11GlobalLocalModel
from models.d12_global_local_motif_model import D12GlobalLocalMotifModel
from models.dual_branch_graph_swin_motif import DualBranchGraphSwinMotifD7
from models.face_aware_graph_swin_d8b import FaceAwareGraphSwinD8B
from models.fixed_motif_classifier import FixedMotifMLPClassifier
from models.graph_swin_prepart_d6b import GraphSwinPrePartD6BD8A
from models.motif_discovery import MotifDiscoveryDebugModel, MotifDiscoveryModule
from models.motif_graph_classifier import FrozenMotifGraphClassifier, MotifGraphClassifier
from models.motif_semantic_projector import FrozenMotifSemanticProjector, MotifSemanticProjector
from models.registry import build_model
from models.slot_pixel_part_graph_motif import SlotPixelPartGraphMotif

__all__ = [
    "ClassPixelMotifGraphRetrieval",
    "D10SlotMotifModel",
    "D11GlobalLocalModel",
    "D12GlobalLocalMotifModel",
    "DualBranchGraphSwinMotifD7",
    "FaceAwareGraphSwinD8B",
    "FixedMotifMLPClassifier",
    "GraphSwinPrePartD6BD8A",
    "MotifDiscoveryDebugModel",
    "MotifDiscoveryModule",
    "MotifGraphClassifier",
    "FrozenMotifGraphClassifier",
    "MotifSemanticProjector",
    "FrozenMotifSemanticProjector",
    "SlotPixelPartGraphMotif",
    "build_model",
]
