"""Small dependency-free pixel message passing for D16."""

from __future__ import annotations

import torch


class PartAwareGNN(torch.nn.Module):
    def __init__(self, hidden_dim: int = 96, layers: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim * 2, hidden_dim),
                    torch.nn.LayerNorm(hidden_dim),
                    torch.nn.GELU(),
                    torch.nn.Dropout(dropout),
                )
                for _ in range(int(layers))
            ]
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0].long(), edge_index[1].long()
        for layer in self.layers:
            agg = h.new_zeros(h.shape)
            agg.index_add_(0, dst, h[src])
            deg = h.new_zeros((h.size(0), 1))
            deg.index_add_(0, dst, torch.ones((dst.numel(), 1), device=h.device, dtype=h.dtype))
            agg = agg / deg.clamp_min(1.0)
            h = h + layer(torch.cat([h, agg], dim=1))
        return h
