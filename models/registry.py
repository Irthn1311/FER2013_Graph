"""Model registry for the clean D5/D6 project."""

from __future__ import annotations

from typing import Any, Dict

from torch import nn

from models.class_pixel_motif_graph_retrieval import ClassPixelMotifGraphRetrieval
from models.d9_relation_motif_model import D9RelationMotifClassifier
from models.d9_sanity_models import D9SanityGlobalPoolClassifier
from models.dual_branch_graph_swin_motif import DualBranchGraphSwinMotifD7
from models.face_aware_graph_swin_d8b import FaceAwareGraphSwinD8B
from models.graph_swin_prepart_d6b import GraphSwinPrePartD6BD8A
from models.motif_graph_classifier import MotifGraphClassifier
from models.motif_semantic_projector import MotifSemanticProjector
from models.motif_discovery import MotifDiscoveryDebugModel
from models.slot_pixel_part_graph_motif import SlotPixelPartGraphMotif


def build_model(config: Dict[str, Any]) -> nn.Module:
    cfg = dict(config)
    name = cfg.pop("name", "class_pixel_motif_graph_retrieval")
    if name == "class_pixel_motif_graph_retrieval":
        return ClassPixelMotifGraphRetrieval.from_config(cfg)
    if name == "slot_pixel_part_graph_motif":
        return SlotPixelPartGraphMotif.from_config(cfg)
    if name == "slot_pixel_part_graph_motif_d6b":
        cfg["use_class_part_attention"] = True
        return SlotPixelPartGraphMotif.from_config(cfg)
    if name in ("dual_branch_graph_swin_motif_d7", "dual_branch_graph_swin_motif"):
        return DualBranchGraphSwinMotifD7.from_config(cfg)
    if name in ("graph_swin_prepart_d6b_d8a", "d8a_graph_swin_prepart_d6b"):
        return GraphSwinPrePartD6BD8A.from_config(cfg)
    if name in ("face_aware_graph_swin_d8b", "d8b_face_aware_graph_swin"):
        return FaceAwareGraphSwinD8B.from_config(cfg)
    if name in ("motif_discovery_debug", "d8m_motif_discovery_debug"):
        return MotifDiscoveryDebugModel.from_config(cfg)
    if name in ("motif_graph_classifier", "frozen_motif_graph_classifier"):
        return MotifGraphClassifier.from_config(cfg)
    if name in ("motif_semantic_projector", "frozen_motif_semantic_projector"):
        return MotifSemanticProjector.from_config(cfg)
    if name in ("d9_relation_motif_classifier", "d9_rg_mr", "d9_rg_mr_classifier"):
        return D9RelationMotifClassifier.from_config(cfg)
    if name in ("d9_sanity_global_pool_classifier", "d9_sanity_global_pool_b"):
        return D9SanityGlobalPoolClassifier.from_config(cfg)
    raise ValueError(f"Unknown model: {name}")
