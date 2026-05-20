"""D13B diagnostic slot bottleneck model.

D13B reuses the D13A hierarchical reduction stack and adds a slot bottleneck
over region nodes. The slots are diagnostic candidates only, not motif or
semantic-region claims.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch import nn

from models.d13_hierarchical_reduction_model import (
    D13HierarchicalReductionModel,
    RegionGraphEncoder,
)
from models.d13_slots import MotifSlotAttention


class SlotSelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=float(dropout), batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class SlotAttentionReadout(nn.Module):
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, slots: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(slots).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        return (slots * weights.unsqueeze(-1)).sum(dim=1), weights


class D13BMotifSlotModel(D13HierarchicalReductionModel):
    """D13A reduction + diagnostic slot bottleneck + classifier."""

    def __init__(
        self,
        num_slots: int = 8,
        slot_dim: int | None = None,
        slot_iters: int = 2,
        slot_dropout: float | None = None,
        slot_self_attention_layers: int = 0,
        slot_readout: str = "attention_pool",
        region_encoder_layers: int | None = None,
        freeze_d13a_backbone: bool = False,
        **kwargs: Any,
    ) -> None:
        kwargs = dict(kwargs)
        hidden_dim = int(kwargs.get("hidden_dim", 64))
        if region_encoder_layers is not None:
            kwargs["region_layers"] = int(region_encoder_layers)
        kwargs["return_region_embeddings"] = True
        super().__init__(**kwargs)
        self.num_slots = int(num_slots)
        self.slot_dim = int(slot_dim or hidden_dim)
        self.slot_readout_name = str(slot_readout)
        dropout = float(slot_dropout if slot_dropout is not None else kwargs.get("dropout", 0.1))
        self.slot_attention = MotifSlotAttention(
            hidden_dim=hidden_dim,
            num_slots=self.num_slots,
            slot_dim=self.slot_dim,
            num_iters=int(slot_iters),
            dropout=dropout,
        )
        self.slot_self_attention = nn.ModuleList(
            SlotSelfAttentionBlock(self.slot_dim, dropout=dropout)
            for _ in range(int(slot_self_attention_layers))
        )
        self.slot_readout = SlotAttentionReadout(self.slot_dim)
        readout_dim = self.slot_dim * 3
        self.classifier = nn.Sequential(
            nn.LayerNorm(readout_dim),
            nn.Linear(readout_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(kwargs.get("dropout", 0.1))),
            nn.Linear(hidden_dim, self.num_classes),
        )
        if bool(freeze_d13a_backbone):
            self.freeze_d13a_backbone()

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "D13BMotifSlotModel":
        return cls(**dict(cfg))

    def freeze_d13a_backbone(self) -> None:
        for module in (self.pixel_encoder, self.reduction, self.region_encoder):
            for param in module.parameters():
                param.requires_grad = False

    @staticmethod
    def _packed_to_dense(h: torch.Tensor, batch: torch.Tensor, pos: torch.Tensor | None = None):
        graph_ids = torch.unique(batch.long(), sorted=True)
        counts = [int((batch.long() == graph_id).sum().item()) for graph_id in graph_ids]
        if len(set(counts)) != 1:
            raise ValueError(f"D13B expects equal region count per graph, got counts={counts[:8]}")
        k = counts[0]
        h_dense = torch.stack([h[batch.long() == graph_id] for graph_id in graph_ids.tolist()], dim=0)
        if pos is None:
            return h_dense, None
        pos_dense = torch.stack([pos[batch.long() == graph_id] for graph_id in graph_ids.tolist()], dim=0)
        return h_dense, pos_dense

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        base = super().forward(batch)
        h_region = base.get("h_region")
        if h_region is None:
            raise RuntimeError("D13B requires D13A h_region; return_region_embeddings must be true")
        region_batch = base["region_batch"]
        region_pos = base.get("region_pos")
        region_dense, region_pos_dense = self._packed_to_dense(h_region, region_batch, region_pos)
        slot_embeddings, slot_attention, slot_aux = self.slot_attention(
            region_dense,
            region_pos=region_pos_dense,
        )
        for layer in self.slot_self_attention:
            slot_embeddings = layer(slot_embeddings)
        slot_mean = slot_embeddings.mean(dim=1)
        slot_max = slot_embeddings.max(dim=1).values
        slot_attn_pool, slot_readout_weights = self.slot_readout(slot_embeddings)
        logits = self.classifier(torch.cat([slot_mean, slot_max, slot_attn_pool], dim=-1))

        aux = dict(base.get("aux", {}))
        aux.update(
            {
                "slot_attention": slot_attention,
                "slot_embeddings": slot_embeddings,
                "slot_readout_weights": slot_readout_weights,
                **slot_aux,
                "slot_losses": {
                    "slot_diversity_loss": slot_aux["slot_diversity_loss"],
                    "slot_overlap_loss": slot_aux["slot_overlap_loss"],
                    "slot_entropy_loss": slot_aux["slot_entropy_loss"],
                    "slot_balance_loss": slot_aux["slot_balance_loss"],
                },
            }
        )
        diagnostics = dict(base.get("diagnostics", {}))
        for key in (
            "slot_entropy",
            "effective_slots",
            "slot_overlap",
            "slot_dominance",
            "slot_area_mean",
            "slot_center_std",
        ):
            value = aux.get(key)
            if torch.is_tensor(value) and value.numel() == 1:
                diagnostics[key] = value.detach()
        return {
            "logits": logits,
            "aux": aux,
            "diagnostics": diagnostics,
            "slot_embeddings": slot_embeddings,
            "slot_attention": slot_attention,
        }
