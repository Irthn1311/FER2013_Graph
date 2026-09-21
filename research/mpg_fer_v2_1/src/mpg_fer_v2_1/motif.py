"""Differentiable multiscale spatial motif composition for MPG-FER v2.1."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import motif_mutual_information_loss


def scheduled_motif_temperature(
    epoch: int,
    tau_start: float = 0.70,
    tau_final: float = 0.30,
    anneal_end_epoch: int = 35,
) -> float:
    """Cosine-anneal motif temperature, then hold the registered floor."""
    if epoch < 1:
        raise ValueError("epoch must be >= 1")
    if anneal_end_epoch < 2:
        raise ValueError("anneal_end_epoch must be >= 2")
    if not 0.0 < tau_final <= tau_start:
        raise ValueError("temperature bounds must satisfy 0 < final <= start")
    progress = min(max((epoch - 1) / (anneal_end_epoch - 1), 0.0), 1.0)
    return tau_final + 0.5 * (tau_start - tau_final) * (
        1.0 + math.cos(math.pi * progress)
    )


def _reflect_index(index: int, size: int) -> int:
    """Map an arbitrary integer to [0,size) using reflection without edge repeat."""
    if size < 2:
        return 0
    period = 2 * (size - 1)
    value = index % period
    return value if value < size else period - value


def build_aligned_support_indices(
    img_size: int = 48,
    anchor_size: int = 12,
    stride: int = 6,
    scales: tuple[int, ...] = (8, 12, 16),
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Build reflection-mapped supports sharing the 12x12 anchor centers."""
    starts = list(range(0, img_size - anchor_size + 1, stride))
    if len(starts) != 7:
        raise ValueError("MPG-FER v2.1 requires a 7x7 anchor grid")
    centers = []
    supports: dict[int, list[list[int]]] = {scale: [] for scale in scales}
    for top in starts:
        for left in starts:
            center_y = top + (anchor_size - 1) / 2.0
            center_x = left + (anchor_size - 1) / 2.0
            centers.append((center_x, center_y))
            for scale in scales:
                start_y = int(round(center_y - (scale - 1) / 2.0))
                start_x = int(round(center_x - (scale - 1) / 2.0))
                indices = [
                    _reflect_index(row, img_size) * img_size
                    + _reflect_index(column, img_size)
                    for row in range(start_y, start_y + scale)
                    for column in range(start_x, start_x + scale)
                ]
                supports[scale].append(indices)
    return (
        {scale: torch.tensor(values, dtype=torch.long) for scale, values in supports.items()},
        torch.tensor(centers, dtype=torch.float32),
    )


def aligned_logical_starts(
    img_size: int = 48,
    anchor_size: int = 12,
    stride: int = 6,
    scales: tuple[int, ...] = (8, 12, 16),
) -> dict[int, torch.Tensor]:
    """Return pre-reflection [top,left] starts for alignment audits."""
    starts = list(range(0, img_size - anchor_size + 1, stride))
    result: dict[int, list[tuple[int, int]]] = {scale: [] for scale in scales}
    for top in starts:
        for left in starts:
            center_y = top + (anchor_size - 1) / 2.0
            center_x = left + (anchor_size - 1) / 2.0
            for scale in scales:
                result[scale].append(
                    (
                        int(round(center_y - (scale - 1) / 2.0)),
                        int(round(center_x - (scale - 1) / 2.0)),
                    )
                )
    return {scale: torch.tensor(values, dtype=torch.long) for scale, values in result.items()}


class SpatialMotifComposer(nn.Module):
    """Soft TYPE assignment and aligned 8/12/16 occurrence fusion into 49 nodes."""

    def __init__(
        self,
        d_pixel: int = 96,
        num_motifs: int = 48,
        motif_window_sizes: tuple[int, int, int] = (8, 12, 16),
        motif_stride: int = 6,
        img_size: int = 48,
        d_type: int = 32,
        d_motif: int = 192,
        tau_start: float = 0.70,
        tau_final: float = 0.30,
        tau_anneal_end_epoch: int = 35,
        mi_beta: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        scheduled_motif_temperature(
            1, tau_start=tau_start, tau_final=tau_final,
            anneal_end_epoch=tau_anneal_end_epoch,
        )
        self.d_pixel = d_pixel
        self.num_motifs = num_motifs
        self.window_sizes = tuple(motif_window_sizes)
        self.stride = motif_stride
        self.img_size = img_size
        self.d_type = d_type
        self.d_motif = d_motif
        self.tau_start = float(tau_start)
        self.tau_final = float(tau_final)
        self.tau_anneal_end_epoch = int(tau_anneal_end_epoch)
        self.mi_beta = float(mi_beta)
        self.eps = eps

        self.prototypes = nn.Parameter(torch.randn(num_motifs, d_pixel) / math.sqrt(d_pixel))
        self.assignment_query = nn.Linear(d_pixel, d_pixel, bias=False)
        self.prototype_key = nn.Linear(d_pixel, d_pixel, bias=False)
        self.register_buffer(
            "current_tau", torch.tensor(self.tau_start, dtype=torch.float32)
        )

        self.scale_saliency = nn.ModuleDict(
            {str(scale): nn.Linear(d_pixel, 1) for scale in self.window_sizes}
        )
        self.scale_confidence = nn.Parameter(torch.ones(len(self.window_sizes)))
        self.type_proj = nn.Sequential(
            nn.Linear(num_motifs, d_type), nn.LayerNorm(d_type), nn.GELU()
        )
        self.occurrence_proj = nn.Sequential(
            nn.Linear(d_pixel + d_type + 5, d_motif),
            nn.LayerNorm(d_motif),
            nn.GELU(),
        )
        self.scale_gate = nn.Linear(d_motif, 1)

        coords = torch.linspace(-1.0, 1.0, img_size)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        self.register_buffer("grid_x", grid_x.reshape(-1))
        self.register_buffer("grid_y", grid_y.reshape(-1))
        supports, centers = build_aligned_support_indices(
            img_size=img_size,
            anchor_size=12,
            stride=motif_stride,
            scales=self.window_sizes,
        )
        self.register_buffer("anchor_centers_pixel", centers)
        for scale, indices in supports.items():
            self.register_buffer(f"support_idx_{scale}", indices)

    @property
    def temperature(self) -> torch.Tensor:
        return self.current_tau

    @torch.no_grad()
    def set_epoch_temperature(self, epoch: int) -> float:
        value = scheduled_motif_temperature(
            epoch,
            tau_start=self.tau_start,
            tau_final=self.tau_final,
            anneal_end_epoch=self.tau_anneal_end_epoch,
        )
        self.current_tau.fill_(value)
        return value

    def _pool_scale(
        self, h_pixel: torch.Tensor, assignments: torch.Tensor,
        confidence: torch.Tensor, scale: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = h_pixel.shape[0]
        indices = getattr(self, f"support_idx_{scale}")
        occurrences, support_size = indices.shape
        gather_idx = indices.unsqueeze(0).expand(batch, -1, -1)
        h_support = h_pixel[:, indices, :]
        a_support = assignments[:, indices, :]
        conf_support = confidence[:, indices, :]
        x_support = self.grid_x[indices].view(1, occurrences, support_size, 1).expand(batch, -1, -1, -1)
        y_support = self.grid_y[indices].view(1, occurrences, support_size, 1).expand(batch, -1, -1, -1)

        saliency = self.scale_saliency[str(scale)](h_support)
        scale_position = self.window_sizes.index(scale)
        weights = F.softmax(saliency + self.scale_confidence[scale_position] * conf_support, dim=2)
        what = (weights * h_support).sum(dim=2)
        type_distribution = (weights * a_support).sum(dim=2)
        motif_type = self.type_proj(type_distribution)
        cx = (weights * x_support).sum(dim=2)
        cy = (weights * y_support).sum(dim=2)
        sx = torch.sqrt((weights * (x_support - cx.unsqueeze(2)).square()).sum(dim=2) + self.eps)
        sy = torch.sqrt((weights * (y_support - cy.unsqueeze(2)).square()).sum(dim=2) + self.eps)
        mass = (weights * conf_support).sum(dim=2)
        where = torch.cat([cx, cy, sx, sy, mass], dim=-1)
        candidate = self.occurrence_proj(torch.cat([what, motif_type, where], dim=-1))
        centers = torch.cat([cx, cy], dim=-1)
        return candidate, centers, weights.squeeze(-1), type_distribution

    def forward(self, h_pixel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch, nodes, dimension = h_pixel.shape
        if nodes != self.img_size * self.img_size or dimension != self.d_pixel:
            raise ValueError("Unexpected pixel embedding shape")

        queries = F.normalize(self.assignment_query(h_pixel), dim=-1)
        keys = F.normalize(self.prototype_key(self.prototypes), dim=-1)
        assignments = F.softmax(queries @ keys.t() / self.temperature, dim=-1)
        confidence = assignments.max(dim=-1, keepdim=True).values

        candidates, centers, weights, type_distributions = [], [], {}, {}
        for scale in self.window_sizes:
            candidate, center, scale_weights, scale_types = self._pool_scale(
                h_pixel, assignments, confidence, scale
            )
            candidates.append(candidate)
            centers.append(center)
            weights[str(scale)] = scale_weights
            type_distributions[str(scale)] = scale_types
        candidate_stack = torch.stack(candidates, dim=2)  # [B,49,3,192]
        center_stack = torch.stack(centers, dim=2)  # [B,49,3,2]
        alpha = F.softmax(self.scale_gate(candidate_stack).squeeze(-1), dim=-1)
        h_motif = (alpha.unsqueeze(-1) * candidate_stack).sum(dim=2)
        fused_centers = (alpha.unsqueeze(-1) * center_stack).sum(dim=2)

        p_norm = F.normalize(self.prototypes, dim=-1)
        prototype_cosine = p_norm @ p_norm.t()
        offdiag_mask = ~torch.eye(self.num_motifs, dtype=torch.bool, device=h_pixel.device)
        offdiag = prototype_cosine[offdiag_mask]
        diversity = F.relu(offdiag).square().mean()
        mi_loss, mi = motif_mutual_information_loss(assignments, beta=self.mi_beta)
        utilization = mi["prototype_utilization"]
        top2 = assignments.topk(2, dim=-1).values
        mean_entropy = mi["H_local_raw"]
        diagnostics = {
            "loss_diversity": diversity,
            "loss_mi": mi_loss,
            "tau": self.temperature,
            "H_local_raw": mi["H_local_raw"],
            "H_local_normalized": mi["H_local_normalized"],
            "H_global_raw": mi["H_global_raw"],
            "H_global_normalized": mi["H_global_normalized"],
            "L_MI": mi["L_MI"],
            "mean_entropy": mean_entropy,
            "effective_motif_count": mean_entropy.exp(),
            "min_utilization": utilization.min(),
            "max_utilization": utilization.max(),
            "std_utilization": utilization.std(unbiased=False),
            "mean_top1_probability": top2[..., 0].mean(),
            "mean_top2_probability": top2[..., 1].mean(),
            "mean_top1_top2_margin": (top2[..., 0] - top2[..., 1]).mean(),
            "mean_offdiag_prototype_cosine": offdiag.mean(),
            "learned_centers_x": fused_centers[..., 0],
            "learned_centers_y": fused_centers[..., 1],
            "scale_weights": alpha,
            "mean_alpha_8": alpha[..., 0].mean(),
            "mean_alpha_12": alpha[..., 1].mean(),
            "mean_alpha_16": alpha[..., 2].mean(),
            "std_alpha_8": alpha[..., 0].std(unbiased=False),
            "std_alpha_12": alpha[..., 1].std(unbiased=False),
            "std_alpha_16": alpha[..., 2].std(unbiased=False),
            "occurrence_weights_8": weights["8"],
            "occurrence_weights_12": weights["12"],
            "occurrence_weights_16": weights["16"],
            "type_distributions": sum(
                alpha[..., index].unsqueeze(-1) * type_distributions[str(scale)]
                for index, scale in enumerate(self.window_sizes)
            ),
        }
        return h_motif, assignments, diagnostics
