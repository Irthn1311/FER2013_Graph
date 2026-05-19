"""Local learnable assignment pooling for D13 hierarchical reduction."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class LocalAssignmentPool(nn.Module):
    """Pool pixels to a per-graph anchor grid with soft local assignments."""

    def __init__(
        self,
        hidden_dim: int = 64,
        grid_size: int = 12,
        assign_m: int = 4,
        neighbor_mode: str = "4",
        dropout: float = 0.1,
        eps: float = 1e-6,
        save_visualization: bool = False,
        target_area: float | None = None,
        assignment_temperature: float = 1.0,
        assignment_temperature_start: float | None = None,
        assignment_temperature_end: float | None = None,
        assignment_temperature_anneal_epochs: int | None = None,
        min_assignment_temperature: float = 0.05,
        **_: Any,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.grid_size = int(grid_size)
        self.assign_m = int(assign_m)
        self.neighbor_mode = str(neighbor_mode)
        self.eps = float(eps)
        self.save_visualization = bool(save_visualization)
        self.target_area = target_area
        self.assignment_temperature = float(assignment_temperature)
        self.assignment_temperature_start = (
            None if assignment_temperature_start is None else float(assignment_temperature_start)
        )
        self.assignment_temperature_end = (
            None if assignment_temperature_end is None else float(assignment_temperature_end)
        )
        self.assignment_temperature_anneal_epochs = (
            None if assignment_temperature_anneal_epochs is None else int(assignment_temperature_anneal_epochs)
        )
        self.min_assignment_temperature = float(min_assignment_temperature)
        self.current_epoch = 0
        if self.grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if self.assign_m <= 0:
            raise ValueError("assign_m must be positive")
        self.assignment_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim + 4, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        self.anchor_offsets = self._grid_offsets(self.grid_size)
        edge_index = self._grid_edges(self.grid_size, mode=self.neighbor_mode)
        self.register_buffer("base_region_edge_index", edge_index, persistent=False)

    @staticmethod
    def _grid_offsets(grid_size: int) -> torch.Tensor:
        lin = torch.linspace(0.0, 1.0, int(grid_size))
        yy, xx = torch.meshgrid(lin, lin, indexing="ij")
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)

    @staticmethod
    def _grid_edges(grid_size: int, mode: str = "4") -> torch.Tensor:
        offsets = [(0, 1), (1, 0)]
        if str(mode) == "8":
            offsets += [(1, 1), (1, -1)]
        edges: List[Tuple[int, int]] = []
        for r in range(grid_size):
            for c in range(grid_size):
                src = r * grid_size + c
                for dr, dc in offsets:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < grid_size and 0 <= cc < grid_size:
                        dst = rr * grid_size + cc
                        edges.append((src, dst))
                        edges.append((dst, src))
        if not edges:
            return torch.empty(2, 0, dtype=torch.long)
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    @property
    def num_regions(self) -> int:
        return self.grid_size * self.grid_size

    def set_save_visualization(self, enabled: bool) -> None:
        self.save_visualization = bool(enabled)

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = max(int(epoch), 0)

    def current_temperature(self) -> float:
        if (
            self.assignment_temperature_start is None
            or self.assignment_temperature_end is None
            or self.assignment_temperature_anneal_epochs is None
            or self.assignment_temperature_anneal_epochs <= 0
        ):
            return max(float(self.assignment_temperature), self.min_assignment_temperature)
        denom = max(float(self.assignment_temperature_anneal_epochs - 1), 1.0)
        progress = min(max(float(self.current_epoch - 1) / denom, 0.0), 1.0)
        temp = self.assignment_temperature_start + progress * (
            self.assignment_temperature_end - self.assignment_temperature_start
        )
        return max(float(temp), self.min_assignment_temperature)

    def forward(
        self,
        h_pixel: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        save_visualization: bool | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if h_pixel.ndim != 2:
            raise ValueError(f"h_pixel must be [N_total, hidden_dim], got {tuple(h_pixel.shape)}")
        if pos.ndim != 2 or pos.shape[1] != 2:
            raise ValueError(f"pos must be [N_total, 2], got {tuple(pos.shape)}")
        if batch.ndim != 1 or batch.shape[0] != h_pixel.shape[0]:
            raise ValueError("batch must be [N_total] and match h_pixel")

        device = h_pixel.device
        dtype = h_pixel.dtype
        graph_ids = torch.unique(batch.long(), sorted=True)
        all_regions: List[torch.Tensor] = []
        all_pos: List[torch.Tensor] = []
        all_batch: List[torch.Tensor] = []
        all_edges: List[torch.Tensor] = []
        area_rows: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        compactness_terms: List[torch.Tensor] = []
        assignment_payload: List[Dict[str, torch.Tensor]] = []
        k = self.num_regions
        m = min(self.assign_m, k)
        base_anchors = self.anchor_offsets.to(device=device, dtype=pos.dtype)
        temperature = self.current_temperature()

        for out_graph_idx, graph_id in enumerate(graph_ids.tolist()):
            mask = batch.long() == int(graph_id)
            h_g = h_pixel[mask]
            pos_g = pos[mask].to(device=device, dtype=pos.dtype)
            if h_g.numel() == 0:
                continue
            if h_g.shape[0] != 2304:
                print(
                    f"[D13 LocalAssignmentPool] WARNING graph_id={graph_id} has "
                    f"{h_g.shape[0]} nodes; expected 2304, continuing with available pos."
                )
            anchors = base_anchors
            d2 = torch.cdist(pos_g.float(), anchors.float(), p=2).to(dtype)
            nearest_dist, nearest_idx = torch.topk(d2, k=m, dim=1, largest=False)
            anchor_pos = anchors.index_select(0, nearest_idx.reshape(-1)).view(h_g.shape[0], m, 2)
            rel_pos = pos_g.unsqueeze(1) - anchor_pos
            h_rep = h_g.unsqueeze(1).expand(-1, m, -1)
            logits = self.assignment_mlp(torch.cat([h_rep, rel_pos.to(dtype), anchor_pos.to(dtype)], dim=-1)).squeeze(-1)
            weights = torch.softmax(logits / max(temperature, self.min_assignment_temperature), dim=1)

            region_sum = h_g.new_zeros((k, self.hidden_dim))
            flat_idx = nearest_idx.reshape(-1)
            flat_w = weights.reshape(-1).to(dtype)
            weighted_h = (h_rep.reshape(-1, self.hidden_dim) * flat_w.unsqueeze(-1))
            region_sum.index_add_(0, flat_idx, weighted_h)
            area = h_g.new_zeros((k,))
            area.index_add_(0, flat_idx, flat_w)
            region_h = region_sum / area.clamp_min(self.eps).unsqueeze(-1)
            region_h = torch.where(area.unsqueeze(-1) > self.eps, region_h, h_g.mean(dim=0, keepdim=True).expand_as(region_h))

            entropy = -(weights * weights.clamp_min(self.eps).log()).sum(dim=1)
            entropies.append(entropy / max(float(torch.log(torch.tensor(float(m), device=device))), self.eps))
            compactness_terms.append((weights * nearest_dist.pow(2)).sum(dim=1).mean())
            area_rows.append(area)
            all_regions.append(region_h)
            all_pos.append(anchors.to(dtype=dtype))
            all_batch.append(torch.full((k,), out_graph_idx, device=device, dtype=torch.long))

            if self.base_region_edge_index.numel() > 0:
                all_edges.append(self.base_region_edge_index.to(device) + out_graph_idx * k)
            if bool(self.save_visualization if save_visualization is None else save_visualization):
                assignment_payload.append(
                    {
                        "graph_id": torch.tensor(int(graph_id), device=device),
                        "pixel_index": torch.nonzero(mask, as_tuple=False).flatten().detach(),
                        "anchor_index": nearest_idx.detach(),
                        "weights": weights.detach(),
                    }
                )

        if not all_regions:
            raise ValueError("LocalAssignmentPool received an empty batch")
        h_region = torch.cat(all_regions, dim=0)
        region_pos = torch.cat(all_pos, dim=0)
        region_batch = torch.cat(all_batch, dim=0)
        region_edge_index = (
            torch.cat(all_edges, dim=1)
            if all_edges
            else torch.empty(2, 0, device=device, dtype=torch.long)
        )
        area_mat = torch.stack(area_rows, dim=0)
        entropy_vec = torch.cat(entropies, dim=0)
        area_mean = area_mat.mean(dim=1, keepdim=True)
        area_std_per_graph = area_mat.std(dim=1, unbiased=False)
        target_area = float(self.target_area) if self.target_area is not None else float(h_pixel.shape[0] / max(len(all_regions), 1) / k)
        uniform_area = area_mean.detach().clamp_min(self.eps)
        balance_loss = ((area_mat - uniform_area) / uniform_area).pow(2).mean()
        area_loss = ((area_mat - area_mean) / area_mean.clamp_min(self.eps)).abs().mean()
        compactness_loss = torch.stack(compactness_terms).mean()
        entropy_loss = entropy_vec.mean()
        effective = area_mat.sum(dim=1).pow(2) / area_mat.pow(2).sum(dim=1).clamp_min(self.eps)
        aux: Dict[str, Any] = {
            "assignment_entropy": entropy_loss,
            "effective_regions": effective.mean(),
            "empty_region_ratio": (area_mat <= self.eps).to(dtype).mean(),
            "region_area_min": area_mat.amin(),
            "region_area_mean": area_mat.mean(),
            "region_area_max": area_mat.amax(),
            "region_area_std": area_std_per_graph.mean(),
            "balance_loss": balance_loss,
            "compactness_loss": compactness_loss,
            "entropy_loss": entropy_loss,
            "area_loss": area_loss,
            "target_area": h_region.new_tensor(target_area),
            "assignment_temperature": h_region.new_tensor(temperature),
        }
        if assignment_payload:
            aux["assignment_maps"] = assignment_payload
        return h_region, region_edge_index, region_batch, {"region_pos": region_pos, **aux}
