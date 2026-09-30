"""Preregistered MPG-FER Table VI ablation model and registry (Issue #101)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from mpg_fer_v2_3.config import MPGConfig
from mpg_fer_v2_3.losses import motif_mutual_information_loss
from mpg_fer_v2_3.model import MPGFER, compute_motif_geometry
from mpg_fer_v2_3.motif import build_aligned_support_indices


class AblationMode(str, Enum):
    NO_PIXEL_GNN = "NO_PIXEL_GNN"
    FIXED_POOL = "FIXED_POOL"
    NO_MOTIF_GNN = "NO_MOTIF_GNN"
    SINGLE_SCALE_12 = "SINGLE_SCALE_12"
    NO_GEOM_BIAS = "NO_GEOM_BIAS"
    DENSE_MOTIF = "DENSE_MOTIF"
    NO_PIXEL_FUSION = "NO_PIXEL_FUSION"
    FULL = "FULL"


@dataclass(frozen=True)
class AblationSpec:
    internal_id: str
    paper_name: str
    scientific_question: str
    exact_intervention: str
    active_modules: dict[str, bool]
    applicable_losses: dict[str, bool]
    applicable_diagnostics: dict[str, bool]
    expected_tensor_invariants: dict[str, list[int]]


_ALL_LOSSES = {
    "final_ce": True,
    "pixel_aux": True,
    "motif_aux": True,
    "prototype_mi": True,
    "prototype_diversity": True,
    "consistency": True,
    "supcon": True,
}
_ALL_DIAGNOSTICS = {
    "motif_prototype_diagnostics": True,
    "motif_routing_diagnostics": True,
    "scale_gate": True,
}
_ALL_MODULES = {
    "pixel_descriptor": True,
    "pixel_projection": True,
    "pixel_gnn": True,
    "pixel_readout": True,
    "spatial_motif_composer": True,
    "fixed_spatial_pool": False,
    "motif_gnn": True,
    "motif_readout": True,
    "pixel_aux_head": True,
    "motif_aux_head": True,
    "supcon_head": True,
    "final_classifier": True,
}
_INVARIANTS = {
    "input": [-1, 1, 48, 48],
    "pixel_nodes": [-1, 2304, 96],
    "occurrence_nodes": [-1, 49, 192],
    "pixel_readout": [-1, 128],
    "motif_readout": [-1, 384],
    "fusion": [-1, 512],
    "logits": [-1, 7],
}


def _with(mapping: dict[str, bool], **changes: bool) -> dict[str, bool]:
    return {**mapping, **changes}


def _spec(
    mode: AblationMode,
    paper_name: str,
    question: str,
    intervention: str,
    *,
    modules: dict[str, bool] | None = None,
    losses: dict[str, bool] | None = None,
    diagnostics: dict[str, bool] | None = None,
) -> AblationSpec:
    return AblationSpec(
        internal_id=mode.value,
        paper_name=paper_name,
        scientific_question=question,
        exact_intervention=intervention,
        active_modules=modules or dict(_ALL_MODULES),
        applicable_losses=losses or dict(_ALL_LOSSES),
        applicable_diagnostics=diagnostics or dict(_ALL_DIAGNOSTICS),
        expected_tensor_invariants=dict(_INVARIANTS),
    )


ABLATION_REGISTRY: dict[AblationMode, AblationSpec] = {
    AblationMode.NO_PIXEL_GNN: _spec(
        AblationMode.NO_PIXEL_GNN,
        "w/o Pixel GNN",
        "Does contextual pixel relational reasoning contribute before spatial occurrence composition?",
        "Keep the 32D descriptor and 32-to-96 projection; bypass all four Pixel-GNN forward passes.",
        modules=_with(_ALL_MODULES, pixel_gnn=False),
    ),
    AblationMode.FIXED_POOL: _spec(
        AblationMode.FIXED_POOL,
        "Fixed spatial pooling",
        "Is learned Spatial Motif Composition useful beyond reducing 2304 contextual pixels to the same 49-node motif graph?",
        "Replace learned SMC with aligned stride-6 12x12 mean pooling followed by Linear 96-to-192, LayerNorm, GELU; use the same support centers.",
        modules=_with(
            _ALL_MODULES, spatial_motif_composer=False, fixed_spatial_pool=True
        ),
        losses=_with(_ALL_LOSSES, prototype_mi=False, prototype_diversity=False),
        diagnostics=_with(
            _ALL_DIAGNOSTICS,
            motif_prototype_diagnostics=False,
            scale_gate=False,
        ),
    ),
    AblationMode.NO_MOTIF_GNN: _spec(
        AblationMode.NO_MOTIF_GNN,
        "w/o Motif Graph",
        "Are the composed 49 occurrences already sufficient, or does relational reasoning among occurrences contribute?",
        "Keep the full SMC and feed its 49 occurrence nodes directly to Motif Readout; bypass all five motif blocks.",
        modules=_with(_ALL_MODULES, motif_gnn=False),
        diagnostics=_with(_ALL_DIAGNOSTICS, motif_routing_diagnostics=False),
    ),
    AblationMode.SINGLE_SCALE_12: _spec(
        AblationMode.SINGLE_SCALE_12,
        "Single-scale Composer (12x12)",
        "Does multiscale 8/12/16 composition contribute relative to a single 12x12 SMC?",
        "Retain learned prototypes, saliency, WHAT/TYPE/WHERE and 133-to-192 projection, but activate only the 12x12 support and no learned scale gate.",
        diagnostics=_with(_ALL_DIAGNOSTICS, scale_gate=False),
    ),
    AblationMode.NO_GEOM_BIAS: _spec(
        AblationMode.NO_GEOM_BIAS,
        "w/o geometry bias",
        "Does explicit relative geometry contribute beyond learned content similarity?",
        "Remove geometry bias from every motif relation score while preserving geometry tensors for diagnostics.",
    ),
    AblationMode.DENSE_MOTIF: _spec(
        AblationMode.DENSE_MOTIF,
        "Dense motif attention",
        "Does selected dynamic support help compared with all-pairs non-self relational support?",
        "Replace [8,16,16,16,24] support with [48,48,48,48,48], retaining geometry bias and all other motif-block computations.",
    ),
    AblationMode.NO_PIXEL_FUSION: _spec(
        AblationMode.NO_PIXEL_FUSION,
        "w/o pixel-level fusion",
        "After the motif path is learned, does direct pixel-level readout still contribute to final classification?",
        "Keep the pixel path and auxiliary head active; replace only the first 128 fusion dimensions with zeros.",
    ),
    AblationMode.FULL: _spec(
        AblationMode.FULL,
        "Full MPG-FER",
        "What is the frozen full-model control under the same seed-42 protocol?",
        "No intervention; execute the frozen MPG-FER v2.3 forward path unchanged.",
    ),
}


TABLE_VI_ORDER = tuple(ABLATION_REGISTRY)


def registry_document() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "issue": 101,
        "method": "MPG-FER",
        "seed": 42,
        "semantic_order": [mode.value for mode in TABLE_VI_ORDER],
        "variants": [asdict(ABLATION_REGISTRY[mode]) for mode in TABLE_VI_ORDER],
    }


class FixedSpatialPoolComposer(nn.Module):
    """The preregistered aligned 12x12 fixed spatial-pooling reference."""

    def __init__(
        self, d_pixel: int = 96, d_motif: int = 192, img_size: int = 48
    ) -> None:
        super().__init__()
        supports, centers = build_aligned_support_indices(
            img_size=img_size,
            anchor_size=12,
            stride=6,
            scales=(12,),
        )
        self.register_buffer("support_idx_12", supports[12])
        normalized_centers = 2.0 * centers / float(img_size - 1) - 1.0
        self.register_buffer("fixed_centers", normalized_centers)
        self.projection = nn.Sequential(
            nn.Linear(d_pixel, d_motif), nn.LayerNorm(d_motif), nn.GELU()
        )

    def forward(self, h_pixel: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        pooled = h_pixel[:, self.support_idx_12, :].mean(dim=2)
        occurrences = self.projection(pooled)
        batch = h_pixel.shape[0]
        centers = self.fixed_centers.unsqueeze(0).expand(batch, -1, -1)
        diagnostics: dict[str, Any] = {
            "learned_centers_x": centers[..., 0],
            "learned_centers_y": centers[..., 1],
            "fixed_support_mean": pooled,
            "motif_prototype_diagnostics_applicable": False,
            "motif_routing_diagnostics_applicable": True,
            "scale_gate_applicable": False,
        }
        return occurrences, diagnostics


def _routing_block_forward(
    layer: nn.Module,
    h_motif: torch.Tensor,
    geom_edges: torch.Tensor,
    *,
    use_geometry_bias: bool,
    topk: int,
    return_diagnostics: bool,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Execute a frozen motif block with only the registered score/support delta."""
    batch, nodes, dimension = h_motif.shape
    normalized = layer.norm1(h_motif)

    def reshape(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(batch, nodes, layer.num_heads, layer.head_dim).permute(
            0, 2, 1, 3
        )

    q = reshape(layer.q_proj(normalized))
    k = reshape(layer.k_proj(normalized))
    v = reshape(layer.v_proj(normalized))
    scores = q @ k.transpose(-1, -2) / (layer.head_dim**0.5)
    if use_geometry_bias:
        scores = scores + layer.geom_proj(geom_edges).permute(0, 3, 1, 2)
    if topk > nodes - 1:
        raise ValueError(f"topk={topk} exceeds the {nodes - 1} non-self keys")
    self_mask = torch.eye(nodes, device=h_motif.device, dtype=torch.bool).view(
        1, 1, nodes, nodes
    )
    self_mask = self_mask.expand(batch, layer.num_heads, nodes, nodes)
    masked_scores = scores.masked_fill(self_mask, torch.finfo(scores.dtype).min)
    if topk < nodes - 1:
        result = torch.topk(masked_scores, k=topk, dim=-1)
        selected_mask = torch.zeros_like(masked_scores, dtype=torch.bool)
        selected_mask.scatter_(-1, result.indices, True)
        cutoff = result.values[..., -1:]
        greater_count = (masked_scores > cutoff).sum(dim=-1)
        equal_count = (masked_scores == cutoff).sum(dim=-1)
        boundary_tie_count = (equal_count > (topk - greater_count)).sum()
        masked_scores = masked_scores.masked_fill(
            ~selected_mask, torch.finfo(scores.dtype).min
        )
    else:
        selected_mask = ~self_mask
        boundary_tie_count = torch.zeros((), dtype=torch.long, device=h_motif.device)
    attention_pre_dropout = F.softmax(masked_scores, dim=-1).masked_fill(
        ~selected_mask, 0.0
    )
    attention = layer.attn_dropout(attention_pre_dropout)
    message = (attention @ v).permute(0, 2, 1, 3).reshape(batch, nodes, dimension)
    h_motif = h_motif + layer.residual_scale * layer.drop_path1(layer.out_proj(message))
    h_motif = h_motif + layer.residual_scale * layer.drop_path2(
        layer.ffn(layer.norm2(h_motif))
    )
    if not return_diagnostics:
        return h_motif

    diagnostic_attention = attention_pre_dropout.detach()
    diagnostic_support = selected_mask.detach()
    p = diagnostic_attention.clamp(min=1e-12)
    entropy = -(diagnostic_attention * p.log()).sum(dim=-1).mean()
    grid_width = math.isqrt(nodes)
    if grid_width * grid_width != nodes:
        raise ValueError("routing diagnostics require a square occurrence grid")
    occurrence = torch.arange(nodes, device=h_motif.device)
    row = occurrence.div(grid_width, rounding_mode="floor")
    column = occurrence.remainder(grid_width)
    chebyshev = torch.maximum(
        (row[:, None] - row[None, :]).abs(),
        (column[:, None] - column[None, :]).abs(),
    ).view(1, 1, nodes, nodes)
    selected_count = diagnostic_support.sum().clamp_min(1)
    return h_motif, {
        "topk": torch.tensor(topk, device=h_motif.device),
        "residual_scale": torch.tensor(
            layer.residual_scale, dtype=h_motif.dtype, device=h_motif.device
        ),
        "selected_mask": diagnostic_support,
        "attention_pre_dropout": diagnostic_attention,
        "entropy": entropy,
        "top1_mass": diagnostic_attention.max(dim=-1).values.mean(),
        "boundary_tie_count": boundary_tie_count.detach(),
        "local_share": (diagnostic_support & (chebyshev == 1)).sum() / selected_count,
        "meso_share": (diagnostic_support & ((chebyshev == 2) | (chebyshev == 3))).sum()
        / selected_count,
        "far_share": (diagnostic_support & (chebyshev >= 4)).sum() / selected_count,
        "edge_universe_coverage": diagnostic_support.any(dim=(0, 1)).sum()
        / (nodes * (nodes - 1)),
    }


class AblationMPGFER(MPGFER):
    """One canonical MPG-FER source implementing all seven deltas plus FULL."""

    def __init__(
        self,
        config: MPGConfig | None = None,
        mode: AblationMode | str = AblationMode.FULL,
    ) -> None:
        self.ablation_mode = AblationMode(mode)
        super().__init__(config)
        if self.ablation_mode is AblationMode.FIXED_POOL:
            # Common/full modules were initialized first so shared modules retain
            # the frozen seed trajectory. The learned SMC is then removed rather
            # than retained as fake inactive prototype machinery.
            del self.motif_composer
            self.fixed_pool_composer = FixedSpatialPoolComposer(
                d_pixel=self.config.d_pixel,
                d_motif=self.config.d_motif,
                img_size=self.config.img_size,
            )

    @property
    def specification(self) -> AblationSpec:
        return ABLATION_REGISTRY[self.ablation_mode]

    @property
    def loss_applicability(self) -> dict[str, bool]:
        return dict(self.specification.applicable_losses)

    @property
    def diagnostic_applicability(self) -> dict[str, bool]:
        return dict(self.specification.applicable_diagnostics)

    def model_summary(
        self,
        *,
        source_sha256: str | None = None,
        source_git_commit: str | None = None,
    ) -> dict[str, Any]:
        summary = super().model_summary(
            source_sha256=source_sha256,
            source_git_commit=source_git_commit,
        )
        summary.update(
            {
                "ablation_mode": self.ablation_mode.value,
                "paper_name": self.specification.paper_name,
                "loss_applicability": self.loss_applicability,
                "diagnostic_applicability": self.diagnostic_applicability,
            }
        )
        return summary

    def set_epoch_temperature(self, epoch: int) -> float | None:
        if self.ablation_mode is AblationMode.FIXED_POOL:
            return None
        return super().set_epoch_temperature(epoch)

    def _single_scale_12(
        self, h_pixel: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        composer = self.motif_composer
        queries = F.normalize(composer.assignment_query(h_pixel), dim=-1)
        keys = F.normalize(composer.prototype_key(composer.prototypes), dim=-1)
        assignments = F.softmax(queries @ keys.t() / composer.temperature, dim=-1)
        confidence = assignments.max(dim=-1, keepdim=True).values
        candidate, centers, weights, type_distribution = composer._pool_scale(
            h_pixel, assignments, confidence, 12
        )

        p_norm = F.normalize(composer.prototypes, dim=-1)
        prototype_cosine = p_norm @ p_norm.t()
        offdiag_mask = ~torch.eye(
            composer.num_motifs, dtype=torch.bool, device=h_pixel.device
        )
        offdiag = prototype_cosine[offdiag_mask]
        diversity = F.relu(offdiag).square().mean()
        mi_loss, mi = motif_mutual_information_loss(assignments, beta=composer.mi_beta)
        utilization = mi["prototype_utilization"]
        top2 = assignments.topk(2, dim=-1).values
        alpha = h_pixel.new_zeros((h_pixel.shape[0], candidate.shape[1], 3))
        alpha[..., 1] = 1.0
        diagnostics: dict[str, Any] = {
            "loss_diversity": diversity,
            "loss_mi": mi_loss,
            "tau": composer.temperature,
            "H_local_raw": mi["H_local_raw"],
            "H_local_normalized": mi["H_local_normalized"],
            "H_global_raw": mi["H_global_raw"],
            "H_global_normalized": mi["H_global_normalized"],
            "L_MI": mi["L_MI"],
            "mean_entropy": mi["H_local_raw"],
            "effective_motif_count": mi["H_local_raw"].exp(),
            "min_utilization": utilization.min(),
            "max_utilization": utilization.max(),
            "std_utilization": utilization.std(unbiased=False),
            "mean_top1_probability": top2[..., 0].mean(),
            "mean_top2_probability": top2[..., 1].mean(),
            "mean_top1_top2_margin": (top2[..., 0] - top2[..., 1]).mean(),
            "mean_offdiag_prototype_cosine": offdiag.mean(),
            "learned_centers_x": centers[..., 0],
            "learned_centers_y": centers[..., 1],
            "scale_weights": alpha,
            "mean_alpha_8": h_pixel.new_zeros(()),
            "mean_alpha_12": h_pixel.new_ones(()),
            "mean_alpha_16": h_pixel.new_zeros(()),
            "std_alpha_8": h_pixel.new_zeros(()),
            "std_alpha_12": h_pixel.new_zeros(()),
            "std_alpha_16": h_pixel.new_zeros(()),
            "occurrence_weights_8": None,
            "occurrence_weights_12": weights,
            "occurrence_weights_16": None,
            "type_distributions": type_distribution,
            "motif_prototype_diagnostics_applicable": True,
            "motif_routing_diagnostics_applicable": True,
            "scale_gate_applicable": False,
        }
        return candidate, assignments, diagnostics

    def forward(
        self, x: torch.Tensor, return_routing_supports: bool = False
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.ablation_mode is AblationMode.FULL:
            return super().forward(x, return_routing_supports=return_routing_supports)

        batch = x.shape[0]
        h = self.pixel_proj(self.pixel_extractor(x))
        h_projected = h
        intensities = x.reshape(batch, self.config.num_pixels, 1)
        edges = self.pixel_topology.compute_edge_features(intensities)
        if self.ablation_mode is not AblationMode.NO_PIXEL_GNN:
            for layer in self.pixel_gnn:
                h = layer(
                    h,
                    self.pixel_topology.neighbor_idx,
                    self.pixel_topology.neighbor_mask,
                    edges,
                )
        p_mean, p_max = h.mean(dim=1), h.max(dim=1).values
        p_attention = (F.softmax(self.pixel_attn_pool(h), dim=1) * h).sum(dim=1)
        pixel_readout = self.pixel_readout_proj(
            torch.cat([p_mean, p_max, p_attention], dim=-1)
        )
        pixel_logits = self.aux_pixel_head(pixel_readout)

        if self.ablation_mode is AblationMode.FIXED_POOL:
            h_motif, diagnostics = self.fixed_pool_composer(h)
            assignments = None
        elif self.ablation_mode is AblationMode.SINGLE_SCALE_12:
            h_motif, assignments, diagnostics = self._single_scale_12(h)
        else:
            h_motif, assignments, diagnostics = self.motif_composer(h)
        h_motif_initial = h_motif
        geometry = compute_motif_geometry(
            diagnostics["learned_centers_x"], diagnostics["learned_centers_y"]
        )

        routing_diagnostics: dict[str, Any] = {}
        routing_applicable = self.ablation_mode is not AblationMode.NO_MOTIF_GNN
        if routing_applicable:
            for l_idx, layer in enumerate(self.motif_gnn):
                topk = (
                    48 if self.ablation_mode is AblationMode.DENSE_MOTIF else layer.topk
                )
                h_motif, layer_diag = _routing_block_forward(
                    layer,
                    h_motif,
                    geometry,
                    use_geometry_bias=self.ablation_mode
                    is not AblationMode.NO_GEOM_BIAS,
                    topk=topk,
                    return_diagnostics=True,
                )
                prefix = f"motif_l{l_idx + 1}"
                routing_diagnostics[f"{prefix}_selected_k"] = layer_diag["topk"]
                routing_diagnostics[f"{prefix}_residual_scale"] = layer_diag[
                    "residual_scale"
                ]
                for field in (
                    "entropy",
                    "top1_mass",
                    "boundary_tie_count",
                    "local_share",
                    "meso_share",
                    "far_share",
                    "edge_universe_coverage",
                ):
                    routing_diagnostics[f"{prefix}_{field}"] = layer_diag[field]
                if return_routing_supports:
                    selected = layer_diag["selected_mask"]
                    routing_diagnostics[f"{prefix}_selected_indices"] = (
                        selected.nonzero(as_tuple=False)[:, -1]
                        .reshape(*selected.shape[:-1], topk)
                        .to(torch.int16)
                    )

        m_mean, m_max = h_motif.mean(dim=1), h_motif.max(dim=1).values
        m_attention = (F.softmax(self.motif_attn_pool(h_motif), dim=1) * h_motif).sum(
            dim=1
        )
        motif_readout = self.motif_readout_proj(
            torch.cat([m_mean, m_max, m_attention], dim=-1)
        )
        motif_logits = self.aux_motif_head(motif_readout)
        pixel_fusion = (
            torch.zeros_like(pixel_readout)
            if self.ablation_mode is AblationMode.NO_PIXEL_FUSION
            else pixel_readout
        )
        fusion = torch.cat([pixel_fusion, motif_readout], dim=-1)
        supcon_embeddings = F.normalize(self.supcon_head(fusion), dim=-1)
        logits = self.classifier(fusion)
        applicability = self.diagnostic_applicability
        diagnostics.update(
            {
                "motif_prototype_diagnostics_applicable": applicability[
                    "motif_prototype_diagnostics"
                ],
                "motif_routing_diagnostics_applicable": applicability[
                    "motif_routing_diagnostics"
                ],
                "scale_gate_applicable": applicability["scale_gate"],
            }
        )
        outputs: dict[str, Any] = {
            "final_logits": logits,
            "pixel_logits": pixel_logits,
            "motif_logits": motif_logits,
            "h_pixel_projected": h_projected,
            "h_pixel_nodes": h,
            "h_motif_nodes_initial": h_motif_initial,
            "h_motif_nodes_final": h_motif,
            "h_pixel_readout": pixel_readout,
            "h_motif_readout": motif_readout,
            "fusion_representation": fusion,
            "supcon_embeddings": supcon_embeddings,
            "motif_assignments": assignments,
            "motif_geometry": geometry,
            "diagnostic_applicability": applicability,
            "loss_applicability": self.loss_applicability,
            **diagnostics,
            **routing_diagnostics,
        }
        return logits, outputs
