"""Relation-aware pixel encoder for D9 motif experiments."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class EdgeAwarePixelEncoderLayer(nn.Module):
    """Sparse edge-aware message passing over the shared pixel graph."""

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        edge_hidden_dim: int | None = None,
        dropout: float = 0.1,
        aggregation: str = "mean",
        use_attention: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.edge_dim = int(edge_dim)
        self.edge_hidden_dim = int(edge_hidden_dim or hidden_dim)
        self.aggregation = str(aggregation or "mean").lower()
        self.use_attention = bool(use_attention)
        if self.aggregation not in {"mean", "sum"}:
            raise ValueError(f"Unsupported aggregation={self.aggregation!r}")
        if self.use_attention:
            raise ValueError("EdgeAwarePixelEncoderLayer attention is intentionally disabled for D9-B prototype")

        self.edge_mlp = nn.Sequential(
            nn.LayerNorm(self.edge_dim),
            nn.Linear(self.edge_dim, self.edge_hidden_dim),
            nn.GELU(),
            nn.Linear(self.edge_hidden_dim, self.hidden_dim),
        )
        self.msg_mlp = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.out_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        if h.ndim != 3:
            raise ValueError(f"h must be [B, N, H], got {tuple(h.shape)}")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must be [2, E], got {tuple(edge_index.shape)}")
        bsz, num_nodes, hidden_dim = h.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(f"Expected hidden_dim={self.hidden_dim}, got {hidden_dim}")

        edge_attr = self._normalize_edge_attr(edge_attr, bsz=bsz, device=h.device, dtype=h.dtype)
        src = edge_index[0].to(device=h.device, dtype=torch.long)
        dst = edge_index[1].to(device=h.device, dtype=torch.long)
        if int(src.max()) >= num_nodes or int(dst.max()) >= num_nodes:
            raise ValueError(f"edge_index references nodes outside N={num_nodes}")

        h_src = h.index_select(dim=1, index=src)
        edge_emb = self.edge_mlp(edge_attr)
        msg = self.msg_mlp(torch.cat([h_src, edge_emb], dim=-1))
        agg = msg.new_zeros(bsz, num_nodes, self.hidden_dim)
        dst_expand = dst.view(1, -1, 1).expand(bsz, -1, self.hidden_dim)
        agg.scatter_add_(dim=1, index=dst_expand, src=msg)
        if self.aggregation == "mean":
            degree = torch.bincount(dst, minlength=num_nodes).to(device=h.device, dtype=agg.dtype)
            agg = agg / degree.clamp_min(1.0).view(1, num_nodes, 1)
        update = self.update_mlp(torch.cat([h, agg], dim=-1))
        return self.out_norm(h + update)

    def _normalize_edge_attr(
        self,
        edge_attr: torch.Tensor,
        *,
        bsz: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if edge_attr.ndim == 2:
            edge_attr = edge_attr.unsqueeze(0).expand(bsz, -1, -1)
        elif edge_attr.ndim == 3:
            if edge_attr.shape[0] != bsz:
                raise ValueError(f"edge_attr batch dim={edge_attr.shape[0]} does not match B={bsz}")
        else:
            raise ValueError(f"edge_attr must be [E, D] or [B, E, D], got {tuple(edge_attr.shape)}")
        if int(edge_attr.shape[-1]) != self.edge_dim:
            raise ValueError(f"Expected edge_dim={self.edge_dim}, got {int(edge_attr.shape[-1])}")
        return edge_attr.to(device=device, dtype=dtype)


class EdgeAwarePixelEncoder(nn.Module):
    """Project masked pixel features and update them with sparse edge messages."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        edge_hidden_dim: int | None = None,
        layers: int = 2,
        dropout: float = 0.1,
        aggregation: str = "mean",
        use_attention: bool = False,
        **_: Any,
    ) -> None:
        super().__init__()
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)
        if self.layers <= 0:
            raise ValueError(f"layers must be positive, got {layers}")
        self.node_proj = nn.Sequential(
            nn.LayerNorm(self.node_dim),
            nn.Linear(self.node_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        self.relation_layers = nn.ModuleList(
            [
                EdgeAwarePixelEncoderLayer(
                    hidden_dim=self.hidden_dim,
                    edge_dim=self.edge_dim,
                    edge_hidden_dim=edge_hidden_dim,
                    dropout=dropout,
                    aggregation=aggregation,
                    use_attention=use_attention,
                )
                for _ in range(self.layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must be [B, N, D], got {tuple(x.shape)}")
        if int(x.shape[-1]) != self.node_dim:
            raise ValueError(f"Expected node_dim={self.node_dim}, got {int(x.shape[-1])}")
        h = self.node_proj(x)
        for layer in self.relation_layers:
            h = layer(h, edge_index=edge_index, edge_attr=edge_attr)
            if node_mask is not None:
                h = h * node_mask.to(device=h.device, dtype=h.dtype).unsqueeze(-1)
        return h
