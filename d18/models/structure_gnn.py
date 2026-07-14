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
        gnn_cfg = dict(gnn or {})
        gnn_cfg.setdefault("hidden_dim", hidden_dim)
        gnn_cfg.setdefault("edge_attr_dim", edge_attr_dim)
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
        )

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        h = self.encoder(batch.x_cat)
        edge_type = getattr(batch, "edge_type_cat", None)
        h = self.gnn(h, batch.edge_index_cat, batch.edge_attr_cat, edge_type=edge_type)
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

