"""D9 relation-guided pixel encoder plus motif-relation classifier."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from models.d9_motif_relation_classifier import (
    D9MotifRelationClassifier,
    D9PooledMotifMLPClassifier,
    D9ResidualPairRelationClassifier,
)
from models.d9_relation_encoder import EdgeAwarePixelEncoder
from models.motif_discovery import MotifDiscoveryModule


class SimplePixelProjector(nn.Module):
    """Node projection used by the no-relation D9 ablation."""

    def __init__(self, node_dim: int, hidden_dim: int = 64, dropout: float = 0.2, **_: Any) -> None:
        super().__init__()
        self.node_dim = int(node_dim)
        self.hidden_dim = int(hidden_dim)
        self.node_proj = nn.Sequential(
            nn.LayerNorm(self.node_dim),
            nn.Linear(self.node_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del edge_index, edge_attr
        if x.ndim != 3:
            raise ValueError(f"x must be [B, N, D], got {tuple(x.shape)}")
        if int(x.shape[-1]) != self.node_dim:
            raise ValueError(f"Expected node_dim={self.node_dim}, got {int(x.shape[-1])}")
        h = self.node_proj(x)
        if node_mask is not None:
            h = h * node_mask.to(device=h.device, dtype=h.dtype).unsqueeze(-1)
        return h


class D9RelationMotifClassifier(nn.Module):
    """End-to-end D9-RG-MR model for FER pixel graph classification."""

    def __init__(
        self,
        node_dim: int = 3,
        edge_dim: int = 5,
        hidden_dim: int = 64,
        num_classes: int = 7,
        num_motifs: int = 16,
        relation_layers: int = 2,
        image_size: int = 48,
        height: Optional[int] = None,
        width: Optional[int] = None,
        dropout: float = 0.2,
        relation_encoder: Optional[Dict[str, Any]] = None,
        motif_discovery: Optional[Dict[str, Any]] = None,
        motif_relation_classifier: Optional[Dict[str, Any]] = None,
        selection_temperature: float = 1.0,
        **_: Any,
    ) -> None:
        super().__init__()
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.num_motifs = int(num_motifs)
        self.height = int(height or image_size)
        self.width = int(width or image_size)
        self.selection_temperature = float(selection_temperature)
        if self.selection_temperature <= 0.0:
            raise ValueError(f"selection_temperature must be > 0, got {selection_temperature}")

        enc_cfg = dict(relation_encoder or {})
        enc_cfg.setdefault("hidden_dim", self.hidden_dim)
        enc_cfg.setdefault("layers", int(enc_cfg.pop("relation_layers", relation_layers)))
        enc_cfg.setdefault("dropout", float(dropout))
        self.relation_encoder_enabled = bool(enc_cfg.pop("enabled", True))
        if self.relation_encoder_enabled:
            self.pixel_encoder = EdgeAwarePixelEncoder(
                node_dim=self.node_dim,
                edge_dim=self.edge_dim,
                **enc_cfg,
            )
        else:
            self.pixel_encoder = SimplePixelProjector(
                node_dim=self.node_dim,
                hidden_dim=self.hidden_dim,
                dropout=float(dropout),
            )

        motif_cfg = dict(motif_discovery or {})
        motif_cfg.pop("reuse_stage1_module", None)
        motif_cfg.setdefault("hidden_dim", self.hidden_dim)
        motif_cfg.setdefault("num_motifs", self.num_motifs)
        motif_cfg.setdefault("dropout", float(dropout))
        self.motif_discovery = MotifDiscoveryModule(
            image_hw=(self.height, self.width),
            **motif_cfg,
        )

        clf_cfg = dict(motif_relation_classifier or {})
        clf_type = str(clf_cfg.pop("type", "relation")).lower()
        clf_cfg.setdefault("hidden_dim", self.hidden_dim)
        clf_cfg.setdefault("num_classes", self.num_classes)
        clf_cfg.setdefault("dropout", float(dropout))
        if clf_type in {"pooled_mlp", "pooled", "mlp"}:
            self.motif_relation_classifier = D9PooledMotifMLPClassifier(
                embedding_dim=self.hidden_dim,
                **clf_cfg,
            )
        elif clf_type in {"residual_pair_relation", "mr_v2_residual", "residual_relation"}:
            self.motif_relation_classifier = D9ResidualPairRelationClassifier(
                embedding_dim=self.hidden_dim,
                **clf_cfg,
            )
        elif clf_type in {"relation", "mr", "motif_relation"}:
            self.motif_relation_classifier = D9MotifRelationClassifier(
                embedding_dim=self.hidden_dim,
                **clf_cfg,
            )
        else:
            raise ValueError(f"Unsupported motif_relation_classifier.type={clf_type!r}")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "D9RelationMotifClassifier":
        return cls(**dict(config))

    def forward(
        self,
        batch_or_x,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del y
        if isinstance(batch_or_x, dict):
            batch = batch_or_x
            x = batch.get("x", batch.get("node_features"))
            edge_index = batch["edge_index"]
            edge_attr = batch["edge_attr"]
            node_mask = batch.get("node_mask")
        else:
            x = batch_or_x
        if x is None:
            raise KeyError("D9RelationMotifClassifier needs 'x' or 'node_features'")
        if edge_index is None or edge_attr is None:
            raise KeyError("D9RelationMotifClassifier requires edge_index and edge_attr")
        if int(x.shape[-1]) != self.node_dim:
            raise ValueError(f"Input node dim={int(x.shape[-1])} does not match model node_dim={self.node_dim}")
        if int(edge_attr.shape[-1]) != self.edge_dim:
            raise ValueError(f"Input edge dim={int(edge_attr.shape[-1])} does not match model edge_dim={self.edge_dim}")

        h_pixel = self.pixel_encoder(x, edge_index=edge_index, edge_attr=edge_attr, node_mask=node_mask)
        motif_out = self.motif_discovery(h_pixel, image_hw=(self.height, self.width), node_mask=node_mask)
        selection_weights = torch.softmax(motif_out["motif_scores"] / self.selection_temperature, dim=1)
        relation_out = self.motif_relation_classifier(
            motif_embeddings=motif_out["motif_embeddings"],
            motif_maps=motif_out["motif_assignment_maps"],
            selection_weights=selection_weights,
            centers=motif_out.get("motif_centers"),
            area=motif_out.get("motif_area"),
        )
        out = {
            **motif_out,
            **relation_out,
            "selection_weights": selection_weights,
            "motif_maps": motif_out["motif_assignment_maps"],
            "motif_aux": motif_out.get("motif_audit", {}),
            "pixel_embeddings": h_pixel,
            "h_pixel": h_pixel,
        }
        return out
