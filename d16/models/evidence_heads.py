"""Pooling heads for D16 pixel embeddings.

These heads are model mechanics only; they do not make interpretability,
motif, semantic-region, or causal-evidence claims.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import torch


def masked_weighted_pool(
    h: torch.Tensor,
    weights: torch.Tensor,
    batch_index: torch.Tensor,
    num_graphs: int,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pooled = []
    valid_out = []
    for graph_id in range(num_graphs):
        node_mask = batch_index == graph_id
        h_g = h[node_mask]
        w_g = weights[node_mask]
        if valid_mask is not None and float(valid_mask[graph_id].item()) <= 0.0:
            pooled.append(h.new_zeros((h.size(1),)))
            valid_out.append(False)
            continue
        denom = w_g.sum().clamp_min(1e-6)
        if float(denom.detach().cpu()) <= 1e-5:
            pooled.append(h.new_zeros((h.size(1),)))
            valid_out.append(False)
        else:
            pooled.append((h_g * w_g[:, None]).sum(dim=0) / denom)
            valid_out.append(True)
    return torch.stack(pooled, dim=0), torch.tensor(valid_out, device=h.device, dtype=torch.bool)


class PartPooling(torch.nn.Module):
    def __init__(self, part_names: List[str]) -> None:
        super().__init__()
        self.part_names = list(part_names)
        self.groups: Dict[str, List[str]] = {
            "mouth": ["mouth", "left_mouth_corner", "right_mouth_corner"],
            "eye": ["left_eye", "right_eye"],
            "brow": ["left_brow", "right_brow"],
            "nose_cheek": ["nose", "left_cheek", "right_cheek"],
        }

    def _indices(self, names: Iterable[str]) -> List[int]:
        return [self.part_names.index(name) for name in names if name in self.part_names]

    def forward(self, h, part_soft, batch_index, num_graphs, valid_part_mask):
        pooled_parts = {}
        valid_parts = {}
        for group_name, names in self.groups.items():
            indices = self._indices(names)
            if not indices:
                weights = part_soft.new_zeros((part_soft.size(0),))
                valid = valid_part_mask.new_zeros((num_graphs,))
            else:
                weights = part_soft[:, indices].max(dim=1).values
                valid = (valid_part_mask[:, indices].sum(dim=1) > 0).float()
            pooled, valid_bool = masked_weighted_pool(h, weights, batch_index, num_graphs, valid)
            pooled_parts[group_name] = pooled
            valid_parts[group_name] = valid_bool
        global_pooled = []
        for graph_id in range(num_graphs):
            node_mask = batch_index == graph_id
            global_pooled.append(h[node_mask].mean(dim=0))
        pooled_parts["global"] = torch.stack(global_pooled, dim=0)
        valid_parts["global"] = torch.ones((num_graphs,), device=h.device, dtype=torch.bool)
        return pooled_parts, valid_parts
