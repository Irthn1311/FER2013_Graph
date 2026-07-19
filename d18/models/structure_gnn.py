"""D18 Structure-Guided Pixel GNN."""

from __future__ import annotations

from typing import Any, Dict

import torch

from d18.models.edge_context_gnn import EdgeContextGNNEncoder
from d18.models.readout import GatedGlobalReadout


class StructureGNN(torch.nn.Module):
    def __init__(
        self,
        input_dim: int = 10,
        edge_attr_dim: int = 6,
        hidden_dim: int = 96,
        num_classes: int = 7,
        dropout: float = 0.2,
        gnn: Dict[str, Any] | None = None,
        edge_type_conditioning: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(int(input_dim), int(hidden_dim)),
            torch.nn.LayerNorm(int(hidden_dim)),
            torch.nn.GELU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(int(hidden_dim), int(hidden_dim)),
            torch.nn.LayerNorm(int(hidden_dim)),
        )
        conditioning_cfg = dict(edge_type_conditioning or {})
        self.edge_type_conditioning_enabled = bool(conditioning_cfg.get("enabled", False))
        self.edge_type_conditioning_mode = str(conditioning_cfg.get("mode", "null"))
        self.edge_type_num_types = int(conditioning_cfg.get("num_types", 2))
        self.edge_type_embedding_dim = int(conditioning_cfg.get("embedding_dim", 8))
        self.edge_type_null_id = int(conditioning_cfg.get("null_id", 0))
        self.edge_type_combine = str(conditioning_cfg.get("combine", "concat"))
        self.base_edge_attr_dim = int(edge_attr_dim)
        if self.edge_type_conditioning_enabled:
            if self.edge_type_conditioning_mode not in {"null", "correct"}:
                raise ValueError(f"Unsupported edge_type_conditioning.mode={self.edge_type_conditioning_mode!r}")
            if self.edge_type_num_types != 2:
                raise ValueError("D19-A1-ID requires exactly two retained relation IDs")
            if self.edge_type_embedding_dim != 8:
                raise ValueError("D19-A1-ID requires embedding_dim=8")
            if self.edge_type_combine != "concat":
                raise ValueError("D19-A1-ID supports only combine='concat'")
            if not 0 <= self.edge_type_null_id < self.edge_type_num_types:
                raise ValueError("edge_type_conditioning.null_id is outside the embedding table")
            self.edge_type_embedding = torch.nn.Embedding(
                self.edge_type_num_types,
                self.edge_type_embedding_dim,
            )
            conditioned_edge_attr_dim = self.base_edge_attr_dim + self.edge_type_embedding_dim
        else:
            self.edge_type_embedding = None
            conditioned_edge_attr_dim = self.base_edge_attr_dim
        self.conditioned_edge_attr_dim = int(conditioned_edge_attr_dim)
        gnn_cfg = dict(gnn or {})
        gnn_cfg.setdefault("hidden_dim", hidden_dim)
        configured_edge_dim = int(gnn_cfg.get("edge_attr_dim", edge_attr_dim))
        if configured_edge_dim != self.base_edge_attr_dim:
            raise ValueError(
                f"model edge_attr_dim={configured_edge_dim} does not match graph base edge_attr_dim={self.base_edge_attr_dim}"
            )
        gnn_cfg["edge_attr_dim"] = self.conditioned_edge_attr_dim
        self.gnn = EdgeContextGNNEncoder.from_config(gnn_cfg)
        self.readout = GatedGlobalReadout(hidden_dim=hidden_dim, dropout=dropout)
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(self.readout.output_dim),
            torch.nn.Linear(self.readout.output_dim, int(hidden_dim) * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(int(hidden_dim) * 2, int(num_classes)),
        )

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], input_dim: int = 10, edge_attr_dim: int = 6) -> "StructureGNN":
        model_cfg = cfg.get("model", {}) or {}
        return cls(
            input_dim=input_dim,
            edge_attr_dim=edge_attr_dim,
            hidden_dim=int(model_cfg.get("hidden_dim", 96)),
            num_classes=int(model_cfg.get("num_classes", 7)),
            dropout=float(model_cfg.get("dropout", 0.2)),
            gnn=model_cfg.get("edge_context_gnn", {}) or {},
            edge_type_conditioning=model_cfg.get("edge_type_conditioning", {}) or {},
        )

    def conditioning_edge_types(
        self,
        edge_type: torch.Tensor,
        mode: str | None = None,
    ) -> torch.Tensor:
        if not self.edge_type_conditioning_enabled:
            raise RuntimeError("edge-type conditioning is disabled")
        if edge_type is None:
            raise ValueError("edge_type_cat is required when edge-type conditioning is enabled")
        if edge_type.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError(f"edge_type_cat must be integer, got {edge_type.dtype}")
        edge_type = edge_type.long()
        if edge_type.ndim != 1:
            raise ValueError(f"edge_type_cat must have shape [E], got {tuple(edge_type.shape)}")
        present = set(int(value) for value in edge_type.detach().cpu().unique().tolist())
        unexpected = present.difference(range(self.edge_type_num_types))
        if unexpected:
            raise ValueError(f"Unexpected retained edge type IDs for D19-A1-ID: {sorted(unexpected)}")
        active_mode = self.edge_type_conditioning_mode if mode is None else str(mode)
        if active_mode == "null":
            return torch.full_like(edge_type, self.edge_type_null_id)
        if active_mode == "correct":
            return edge_type
        if active_mode == "swapped":
            return 1 - edge_type
        raise ValueError(f"Unsupported edge-type conditioning treatment={active_mode!r}")

    def conditioned_edge_attributes(
        self,
        edge_attr: torch.Tensor,
        edge_type: torch.Tensor,
        mode: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.edge_type_conditioning_enabled:
            return edge_attr, edge_type
        if edge_attr.ndim != 2 or edge_attr.size(1) != self.base_edge_attr_dim:
            raise ValueError(
                f"Expected base edge_attr [E,{self.base_edge_attr_dim}], got {tuple(edge_attr.shape)}"
            )
        if edge_type.numel() != edge_attr.size(0):
            raise ValueError("edge_type count does not match edge_attr edge count")
        conditioning_type = self.conditioning_edge_types(edge_type, mode=mode)
        type_embedding = self.edge_type_embedding(conditioning_type.to(edge_attr.device))
        type_embedding = type_embedding.to(dtype=edge_attr.dtype)
        return torch.cat([edge_attr, type_embedding], dim=1), conditioning_type

    def conditioning_schema(self) -> Dict[str, Any]:
        return {
            "enabled": self.edge_type_conditioning_enabled,
            "mode": self.edge_type_conditioning_mode if self.edge_type_conditioning_enabled else "disabled",
            "num_types": self.edge_type_num_types if self.edge_type_conditioning_enabled else 0,
            "embedding_dim": self.edge_type_embedding_dim if self.edge_type_conditioning_enabled else 0,
            "null_id": self.edge_type_null_id if self.edge_type_conditioning_enabled else None,
            "combine": self.edge_type_combine if self.edge_type_conditioning_enabled else None,
            "base_edge_attr_dim": self.base_edge_attr_dim,
            "conditioned_edge_attr_dim": self.conditioned_edge_attr_dim,
            "embedding_initialization": "torch.nn.Embedding default initialization",
            "shared_across_gnn_layers": self.edge_type_conditioning_enabled,
        }

    def forward(self, batch, conditioning_mode: str | None = None) -> Dict[str, torch.Tensor]:
        h = self.encoder(batch.x_cat)
        edge_type = getattr(batch, "edge_type_cat", None)
        edge_attr = batch.edge_attr_cat
        if self.edge_type_conditioning_enabled:
            edge_attr, _ = self.conditioned_edge_attributes(edge_attr, edge_type, mode=conditioning_mode)
        h = self.gnn(h, batch.edge_index_cat, edge_attr, edge_type=edge_type)
        z = self.readout(h, batch.batch_index, batch.num_graphs)
        logits = self.classifier(z)
        out = {"logits": logits, "z_image": z, "node_embeddings": h}
        penalty = self.gnn.structure_gate_penalty()
        if penalty is not None:
            out["structure_gate_penalty"] = penalty
        gate_stats = self.gnn.gate_stats()
        if gate_stats:
            out["gate_stats"] = gate_stats
        return out

