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
        self.group_indices: Dict[str, List[int]] = {
            group_name: self._indices(names) for group_name, names in self.groups.items()
        }

    def _indices(self, names: Iterable[str]) -> List[int]:
        return [self.part_names.index(name) for name in names if name in self.part_names]

    def forward(self, h, part_soft, batch_index, num_graphs, valid_part_mask):
        pooled_lists = {group_name: [] for group_name in self.groups}
        valid_lists = {group_name: [] for group_name in self.groups}
        for graph_id in range(num_graphs):
            node_mask = batch_index == graph_id
            h_g = h[node_mask]
            part_g = part_soft[node_mask]
            for group_name, indices in self.group_indices.items():
                if not indices or float(valid_part_mask[graph_id, indices].sum().item()) <= 0.0:
                    pooled_lists[group_name].append(h.new_zeros((h.size(1),)))
                    valid_lists[group_name].append(False)
                    continue
                weights = part_g[:, indices].max(dim=1).values
                denom = weights.sum().clamp_min(1e-6)
                if float(denom.detach().cpu()) <= 1e-5:
                    pooled_lists[group_name].append(h.new_zeros((h.size(1),)))
                    valid_lists[group_name].append(False)
                else:
                    pooled_lists[group_name].append((h_g * weights[:, None]).sum(dim=0) / denom)
                    valid_lists[group_name].append(True)

        pooled_parts = {
            group_name: torch.stack(values, dim=0) for group_name, values in pooled_lists.items()
        }
        valid_parts = {
            group_name: torch.tensor(values, device=h.device, dtype=torch.bool)
            for group_name, values in valid_lists.items()
        }
        global_pooled = []
        for graph_id in range(num_graphs):
            node_mask = batch_index == graph_id
            global_pooled.append(h[node_mask].mean(dim=0))
        pooled_parts["global"] = torch.stack(global_pooled, dim=0)
        valid_parts["global"] = torch.ones((num_graphs,), device=h.device, dtype=torch.bool)
        return pooled_parts, valid_parts
