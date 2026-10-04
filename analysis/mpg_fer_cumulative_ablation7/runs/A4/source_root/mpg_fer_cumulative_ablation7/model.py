"""Cumulative MPG-FER 7-configuration ablation ladder."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from mpg_fer_v2_3.config import MPGConfig
from mpg_fer_v2_3.losses import motif_mutual_information_loss
from mpg_fer_v2_3.model import MPGFER, compute_motif_geometry
from mpg_fer_table_vi.model import _routing_block_forward


class CumulativeAblationMode(str, Enum):
    A0 = "A0"
    A1 = "A1"
    A2 = "A2"
    A3 = "A3"
    A4 = "A4"
    A5 = "A5"
    A6 = "A6"


@dataclass(frozen=True)
class CumulativeAblationSpec:
    internal_id: str
    paper_name: str
    vietnamese_name: str
    cumulative_description: str
    pixel_gnn: bool
    composer: str | None
    composer_scales: list[int]
    motif_gnn: bool
    geometry_bias: bool
    relation_mode: str
    motif_topk_schedule: list[int]
    motif_pooling: str
    direct_pixel_fusion: bool
    classifier: str
    applicable_losses: dict[str, float]
    notes: str


CUMULATIVE_REGISTRY: dict[CumulativeAblationMode, CumulativeAblationSpec] = {
    CumulativeAblationMode.A0: CumulativeAblationSpec(
        internal_id="A0",
        paper_name="Pixel Graph baseline",
        vietnamese_name="Pixel Graph baseline",
        cumulative_description="Baseline: full Pixel GNN and Pixel Readout only; zero motif representation.",
        pixel_gnn=True,
        composer=None,
        composer_scales=[],
        motif_gnn=False,
        geometry_bias=False,
        relation_mode="none",
        motif_topk_schedule=[],
        motif_pooling="none",
        direct_pixel_fusion=True,
        classifier="full_concat_512",
        applicable_losses={
            "final_ce": 1.0,
            "pixel_aux": 0.05,
            "motif_aux": 0.0,
            "lambda_mi": 0.0,
            "lambda_div": 0.0,
            "lambda_consistency": 0.15,
            "lambda_supcon": 0.05,
        },
        notes="Zero tensor for motif readout preserves 512D classifier input shape.",
    ),
    CumulativeAblationMode.A1: CumulativeAblationSpec(
        internal_id="A1",
        paper_name="+ Spatial Motif Composer",
        vietnamese_name="+ Spatial Motif Composer",
        cumulative_description="A0 + single-scale (12x12) Spatial Motif Composer with fixed motif pooling.",
        pixel_gnn=True,
        composer="single_scale_12",
        composer_scales=[12],
        motif_gnn=False,
        geometry_bias=False,
        relation_mode="none",
        motif_topk_schedule=[],
        motif_pooling="fixed",
        direct_pixel_fusion=True,
        classifier="full_concat_512",
        applicable_losses={
            "final_ce": 1.0,
            "pixel_aux": 0.05,
            "motif_aux": 0.2,
            "lambda_mi": 0.025,
            "lambda_div": 0.01,
            "lambda_consistency": 0.15,
            "lambda_supcon": 0.05,
        },
        notes="Composed 49 occurrence nodes fed directly to fixed motif readout (no Motif GNN).",
    ),
    CumulativeAblationMode.A2: CumulativeAblationSpec(
        internal_id="A2",
        paper_name="+ Motif Graph",
        vietnamese_name="+ Motif Graph",
        cumulative_description="A1 + dense relational reasoning among occurrence nodes (Motif GNN, all-pairs non-self).",
        pixel_gnn=True,
        composer="single_scale_12",
        composer_scales=[12],
        motif_gnn=True,
        geometry_bias=False,
        relation_mode="dense",
        motif_topk_schedule=[48, 48, 48, 48, 48],
        motif_pooling="fixed",
        direct_pixel_fusion=True,
        classifier="full_concat_512",
        applicable_losses={
            "final_ce": 1.0,
            "pixel_aux": 0.05,
            "motif_aux": 0.2,
            "lambda_mi": 0.025,
            "lambda_div": 0.01,
            "lambda_consistency": 0.15,
            "lambda_supcon": 0.05,
        },
        notes="Full 5-layer Motif GNN enabled with dense (all-pairs) support; geometry bias disabled.",
    ),
    CumulativeAblationMode.A3: CumulativeAblationSpec(
        internal_id="A3",
        paper_name="+ Geometry Bias",
        vietnamese_name="+ Geometry Bias",
        cumulative_description="A2 + relative geometry bias term added to motif relation attention scores.",
        pixel_gnn=True,
        composer="single_scale_12",
        composer_scales=[12],
        motif_gnn=True,
        geometry_bias=True,
        relation_mode="dense",
        motif_topk_schedule=[48, 48, 48, 48, 48],
        motif_pooling="fixed",
        direct_pixel_fusion=True,
        classifier="full_concat_512",
        applicable_losses={
            "final_ce": 1.0,
            "pixel_aux": 0.05,
            "motif_aux": 0.2,
            "lambda_mi": 0.025,
            "lambda_div": 0.01,
            "lambda_consistency": 0.15,
            "lambda_supcon": 0.05,
        },
        notes="Geometry projection added to pairwise attention scores in Motif GNN.",
    ),
    CumulativeAblationMode.A4: CumulativeAblationSpec(
        internal_id="A4",
        paper_name="+ Multi-scale Composition",
        vietnamese_name="+ Multi-scale Composition",
        cumulative_description="A3 + multi-scale (8, 12, 16) Spatial Motif Composition with learned scale gate.",
        pixel_gnn=True,
        composer="multi_scale_8_12_16",
        composer_scales=[8, 12, 16],
        motif_gnn=True,
        geometry_bias=True,
        relation_mode="dense",
        motif_topk_schedule=[48, 48, 48, 48, 48],
        motif_pooling="fixed",
        direct_pixel_fusion=True,
        classifier="full_concat_512",
        applicable_losses={
            "final_ce": 1.0,
            "pixel_aux": 0.05,
            "motif_aux": 0.2,
            "lambda_mi": 0.025,
            "lambda_div": 0.01,
            "lambda_consistency": 0.15,
            "lambda_supcon": 0.05,
        },
        notes="Full multiscale support 8/12/16 replacing single scale 12x12.",
    ),
    CumulativeAblationMode.A5: CumulativeAblationSpec(
        internal_id="A5",
        paper_name="+ Dynamic Top-K Relations",
        vietnamese_name="+ Dynamic Top-K Relations",
        cumulative_description="A4 + dynamic Top-K relation schedule [8, 16, 16, 16, 24] replacing dense relations.",
        pixel_gnn=True,
        composer="multi_scale_8_12_16",
        composer_scales=[8, 12, 16],
        motif_gnn=True,
        geometry_bias=True,
        relation_mode="dynamic_topk",
        motif_topk_schedule=[8, 16, 16, 16, 24],
        motif_pooling="fixed",
        direct_pixel_fusion=True,
        classifier="full_concat_512",
        applicable_losses={
            "final_ce": 1.0,
            "pixel_aux": 0.05,
            "motif_aux": 0.2,
            "lambda_mi": 0.025,
            "lambda_div": 0.01,
            "lambda_consistency": 0.15,
            "lambda_supcon": 0.05,
        },
        notes="Dynamic sparse routing with preregistered topk schedule; fixed motif pooling retained.",
    ),
    CumulativeAblationMode.A6: CumulativeAblationSpec(
        internal_id="A6",
        paper_name="Full Model",
        vietnamese_name="Full Model",
        cumulative_description="A5 + learnable attention pooling in motif readout. Complete canonical MPG-FER v2.3 architecture.",
        pixel_gnn=True,
        composer="multi_scale_8_12_16",
        composer_scales=[8, 12, 16],
        motif_gnn=True,
        geometry_bias=True,
        relation_mode="dynamic_topk",
        motif_topk_schedule=[8, 16, 16, 16, 24],
        motif_pooling="learnable_attention",
        direct_pixel_fusion=True,
        classifier="full_concat_512",
        applicable_losses={
            "final_ce": 1.0,
            "pixel_aux": 0.05,
            "motif_aux": 0.2,
            "lambda_mi": 0.025,
            "lambda_div": 0.01,
            "lambda_consistency": 0.15,
            "lambda_supcon": 0.05,
        },
        notes="Canonical FULL model executing the unchanged forward path.",
    ),
}

CUMULATIVE_ABLATION_ORDER = tuple(CUMULATIVE_REGISTRY)


def cumulative_registry_document() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "method": "MPG-FER",
        "seed": 42,
        "configurations": [asdict(CUMULATIVE_REGISTRY[m]) for m in CUMULATIVE_ABLATION_ORDER],
    }


class CumulativeAblationMPGFER(MPGFER):
    """Subclass of frozen MPGFER implementing strictly nested cumulative ladder A0..A6."""

    def __init__(
        self,
        config: MPGConfig | None = None,
        mode: CumulativeAblationMode | str = CumulativeAblationMode.A6,
    ) -> None:
        self.cumulative_mode = CumulativeAblationMode(mode)
        super().__init__(config)

    @property
    def spec(self) -> CumulativeAblationSpec:
        return CUMULATIVE_REGISTRY[self.cumulative_mode]

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
        alpha = h_pixel.new_zeros((h_pixel.shape[0], candidate.shape[1], 3))
        alpha[..., 1] = 1.0
        diagnostics: dict[str, Any] = {
            "loss_diversity": diversity,
            "loss_mi": mi_loss,
            "tau": composer.temperature,
            "learned_centers_x": centers[..., 0],
            "learned_centers_y": centers[..., 1],
            "scale_weights": alpha,
            "type_distributions": type_distribution,
        }
        return candidate, assignments, diagnostics

    def forward(
        self, x: torch.Tensor, return_routing_supports: bool = False
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # A6 executes the exact unchanged canonical FULL forward path
        if self.cumulative_mode is CumulativeAblationMode.A6:
            return super().forward(x, return_routing_supports=return_routing_supports)

        batch = x.shape[0]
        # Pixel GNN path is identical across all configurations
        h = self.pixel_proj(self.pixel_extractor(x))
        h_projected = h
        intensities = x.reshape(batch, self.config.num_pixels, 1)
        edges = self.pixel_topology.compute_edge_features(intensities)
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

        # A0: Pixel Graph baseline (no motif composer, no motif GNN, zero motif readout)
        if self.cumulative_mode is CumulativeAblationMode.A0:
            motif_readout = torch.zeros(
                batch,
                self.config.d_motif_readout,
                device=x.device,
                dtype=pixel_readout.dtype,
            )
            motif_logits = torch.zeros(
                batch,
                self.config.num_classes,
                device=x.device,
                dtype=pixel_readout.dtype,
            )
            full_fusion = torch.cat([pixel_readout, motif_readout], dim=-1)
            supcon_embeddings = F.normalize(self.supcon_head(full_fusion), dim=-1)
            logits = self.classifier(full_fusion)
            outputs: dict[str, Any] = {
                "final_logits": logits,
                "pixel_logits": pixel_logits,
                "motif_logits": motif_logits,
                "h_pixel_projected": h_projected,
                "h_pixel_nodes": h,
                "h_pixel_readout": pixel_readout,
                "h_motif_readout": motif_readout,
                "fusion_representation": full_fusion,
                "supcon_source_representation": full_fusion,
                "supcon_embeddings": supcon_embeddings,
                "loss_diversity": torch.zeros((), device=x.device),
                "loss_mi": torch.zeros((), device=x.device),
                "scale_weights": None,
                "motif_assignments": None,
                "motif_geometry": None,
            }
            return logits, outputs

        # A1..A5: Motif Composer
        if self.spec.composer == "single_scale_12":
            h_motif, assignments, diagnostics = self._single_scale_12(h)
        else:
            h_motif, assignments, diagnostics = self.motif_composer(h)
        h_motif_initial = h_motif

        geometry = compute_motif_geometry(
            diagnostics["learned_centers_x"], diagnostics["learned_centers_y"]
        )

        # Motif GNN reasoning
        routing_diagnostics: dict[str, Any] = {}
        if self.spec.motif_gnn:
            use_geom = self.spec.geometry_bias
            for l_idx, layer in enumerate(self.motif_gnn):
                topk = (
                    48
                    if self.spec.relation_mode == "dense"
                    else self.spec.motif_topk_schedule[l_idx]
                )
                h_motif, layer_diag = _routing_block_forward(
                    layer,
                    h_motif,
                    geometry,
                    use_geometry_bias=use_geom,
                    topk=topk,
                    return_diagnostics=True,
                )
                prefix = f"motif_l{l_idx + 1}"
                routing_diagnostics[f"{prefix}_selected_k"] = layer_diag["topk"]
                routing_diagnostics[f"{prefix}_residual_scale"] = layer_diag[
                    "residual_scale"
                ]

        # Motif Readout
        m_mean, m_max = h_motif.mean(dim=1), h_motif.max(dim=1).values
        if self.spec.motif_pooling == "fixed":
            # Fixed uniform mean pooling over occurrences
            m_pool = h_motif.mean(dim=1)
        else:
            m_pool = (F.softmax(self.motif_attn_pool(h_motif), dim=1) * h_motif).sum(dim=1)

        motif_readout = self.motif_readout_proj(
            torch.cat([m_mean, m_max, m_pool], dim=-1)
        )
        motif_logits = self.aux_motif_head(motif_readout)

        full_fusion = torch.cat([pixel_readout, motif_readout], dim=-1)
        supcon_embeddings = F.normalize(self.supcon_head(full_fusion), dim=-1)
        logits = self.classifier(full_fusion)

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
            "fusion_representation": full_fusion,
            "supcon_source_representation": full_fusion,
            "supcon_embeddings": supcon_embeddings,
            "motif_assignments": assignments,
            "motif_geometry": geometry,
            **diagnostics,
            **routing_diagnostics,
        }
        return logits, outputs
