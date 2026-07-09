"""Prior-free graph-level readouts for D17."""

from __future__ import annotations

import torch


def _scatter_mean(h: torch.Tensor, batch_index: torch.Tensor, num_graphs: int) -> torch.Tensor:
    out = h.new_zeros((num_graphs, h.size(1)))
    out.index_add_(0, batch_index.long(), h)
    counts = h.new_zeros((num_graphs, 1))
    counts.index_add_(0, batch_index.long(), torch.ones((h.size(0), 1), device=h.device, dtype=h.dtype))
    return out / counts.clamp_min(1.0)


def _scatter_max(h: torch.Tensor, batch_index: torch.Tensor, num_graphs: int) -> torch.Tensor:
    if hasattr(torch.Tensor, "scatter_reduce_"):
        out = h.new_full((num_graphs, h.size(1)), -torch.inf)
        index = batch_index.long().view(-1, 1).expand(-1, h.size(1))
        out.scatter_reduce_(0, index, h, reduce="amax", include_self=True)
        return torch.nan_to_num(out, neginf=0.0)
    rows = []
    for graph_id in range(num_graphs):
        mask = batch_index == graph_id
        rows.append(h[mask].max(dim=0).values if bool(mask.any()) else h.new_zeros((h.size(1),)))
    return torch.stack(rows, dim=0)


class GatedGlobalReadout(torch.nn.Module):
    def __init__(self, hidden_dim: int = 96, dropout: float = 0.2) -> None:
        super().__init__()
        self.gate = torch.nn.Sequential(
            torch.nn.LayerNorm(int(hidden_dim)),
            torch.nn.Linear(int(hidden_dim), int(hidden_dim)),
            torch.nn.GELU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(int(hidden_dim), 1),
            torch.nn.Sigmoid(),
        )
        self.output_dim = int(hidden_dim) * 3

    def forward(self, h: torch.Tensor, batch_index: torch.Tensor, num_graphs: int) -> torch.Tensor:
        mean = _scatter_mean(h, batch_index, num_graphs)
        max_pool = _scatter_max(h, batch_index, num_graphs)
        gate = self.gate(h)
        gated_sum = h.new_zeros((num_graphs, h.size(1)))
        gated_sum.index_add_(0, batch_index.long(), h * gate)
        denom = h.new_zeros((num_graphs, 1))
        denom.index_add_(0, batch_index.long(), gate)
        gated = gated_sum / denom.clamp_min(1e-6)
        return torch.cat([mean, max_pool, gated], dim=1)

