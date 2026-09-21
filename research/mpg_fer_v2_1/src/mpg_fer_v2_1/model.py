"""Pure-GNN MPG-FER v2.1 model: pixel graph, motifs, and motif graph."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MPGConfig
from .features import PixelFeatureExtractor
from .graph import PixelGraphTopology
from .motif import SpatialMotifComposer


class DropPath(nn.Module):
    """Per-sample stochastic depth applied to residual branches, never nodes."""

    def __init__(self, drop_probability: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= drop_probability < 1.0:
            raise ValueError("drop_probability must be in [0,1)")
        self.drop_probability = float(drop_probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_probability == 0.0:
            return x
        keep = 1.0 - self.drop_probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep)
        return x * mask / keep


def masked_neighbor_softmax(scores: torch.Tensor, neighbor_mask: torch.Tensor) -> torch.Tensor:
    mask = neighbor_mask.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    weights = F.softmax(scores.masked_fill(~mask, torch.finfo(scores.dtype).min), dim=-1)
    return weights.masked_fill(~mask, 0.0)


def compute_motif_geometry(cx: torch.Tensor, cy: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    dx = cx.unsqueeze(1) - cx.unsqueeze(2)
    dy = cy.unsqueeze(1) - cy.unsqueeze(2)
    dist_sq = dx.square() + dy.square()
    dist = torch.sqrt(dist_sq + eps)
    nodes = cx.shape[1]
    self_mask = torch.eye(nodes, device=cx.device, dtype=torch.bool).unsqueeze(0)
    dist = dist.masked_fill(self_mask, 0.0)
    sin_theta = (dy / (dist + eps)).masked_fill(self_mask, 0.0)
    cos_theta = (dx / (dist + eps)).masked_fill(self_mask, 0.0)
    return torch.stack([dx, dy, dist, dist_sq, sin_theta, cos_theta], dim=-1)


class EdgeAwarePixelGNNLayer(nn.Module):
    """8-neighbor attention with K/V projected once per node before gather."""

    def __init__(
        self, d_pixel: int = 96, edge_dim: int = 5, num_heads: int = 4,
        dropout: float = 0.15, drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        if d_pixel % num_heads:
            raise ValueError("d_pixel must be divisible by num_heads")
        self.d_pixel = d_pixel
        self.num_heads = num_heads
        self.head_dim = d_pixel // num_heads
        self.norm1 = nn.LayerNorm(d_pixel)
        self.q_proj = nn.Linear(d_pixel, d_pixel)
        self.k_proj = nn.Linear(d_pixel, d_pixel)
        self.v_proj = nn.Linear(d_pixel, d_pixel)
        self.edge_bias = nn.Linear(edge_dim, num_heads)
        self.edge_val = nn.Linear(edge_dim, d_pixel)
        self.out_proj = nn.Linear(d_pixel, d_pixel)
        self.attn_dropout = nn.Dropout(dropout)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(d_pixel)
        self.ffn = nn.Sequential(
            nn.Linear(d_pixel, 2 * d_pixel), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d_pixel, d_pixel), nn.Dropout(dropout),
        )
        self.drop_path2 = DropPath(drop_path)

    def forward(
        self, h: torch.Tensor, neighbor_idx: torch.Tensor,
        neighbor_mask: torch.Tensor, edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        batch, nodes, dimension = h.shape
        heads, head_dim = self.num_heads, self.head_dim
        normalized = self.norm1(h)
        q = self.q_proj(normalized).reshape(batch, nodes, heads, head_dim).unsqueeze(3)
        k_all = self.k_proj(normalized)
        v_all = self.v_proj(normalized)
        k = k_all[:, neighbor_idx, :].reshape(batch, nodes, 8, heads, head_dim).permute(0, 1, 3, 2, 4)
        v = (v_all[:, neighbor_idx, :] + self.edge_val(edge_feats)).reshape(
            batch, nodes, 8, heads, head_dim
        ).permute(0, 1, 3, 2, 4)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (head_dim**0.5)
        scores = scores + self.edge_bias(edge_feats).permute(0, 1, 3, 2).unsqueeze(3)
        attention = self.attn_dropout(masked_neighbor_softmax(scores, neighbor_mask))
        message = torch.matmul(attention, v).squeeze(3).reshape(batch, nodes, dimension)
        h = h + self.drop_path1(self.out_proj(message))
        return h + self.drop_path2(self.ffn(self.norm2(h)))


class GeometryAwareMotifTransformerBlock(nn.Module):
    def __init__(
        self, d_motif: int = 192, geom_dim: int = 6, num_heads: int = 6,
        dropout: float = 0.15, drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        if d_motif % num_heads:
            raise ValueError("d_motif must be divisible by num_heads")
        self.d_motif = d_motif
        self.num_heads = num_heads
        self.head_dim = d_motif // num_heads
        self.norm1 = nn.LayerNorm(d_motif)
        self.q_proj = nn.Linear(d_motif, d_motif)
        self.k_proj = nn.Linear(d_motif, d_motif)
        self.v_proj = nn.Linear(d_motif, d_motif)
        self.geom_proj = nn.Linear(geom_dim, num_heads)
        self.out_proj = nn.Linear(d_motif, d_motif)
        self.attn_dropout = nn.Dropout(dropout)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(d_motif)
        self.ffn = nn.Sequential(
            nn.Linear(d_motif, 2 * d_motif), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d_motif, d_motif), nn.Dropout(dropout),
        )
        self.drop_path2 = DropPath(drop_path)

    def forward(self, h_motif: torch.Tensor, geom_edges: torch.Tensor) -> torch.Tensor:
        batch, nodes, dimension = h_motif.shape
        normalized = self.norm1(h_motif)
        reshape = lambda value: value.reshape(batch, nodes, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k, v = reshape(self.q_proj(normalized)), reshape(self.k_proj(normalized)), reshape(self.v_proj(normalized))
        scores = q @ k.transpose(-1, -2) / (self.head_dim**0.5)
        scores = scores + self.geom_proj(geom_edges).permute(0, 3, 1, 2)
        self_mask = torch.eye(nodes, device=h_motif.device, dtype=torch.bool).view(1, 1, nodes, nodes)
        attention = self.attn_dropout(F.softmax(scores.masked_fill(self_mask, torch.finfo(scores.dtype).min), dim=-1))
        message = (attention @ v).permute(0, 2, 1, 3).reshape(batch, nodes, dimension)
        h_motif = h_motif + self.drop_path1(self.out_proj(message))
        return h_motif + self.drop_path2(self.ffn(self.norm2(h_motif)))


def _linear_rates(depth: int, maximum: float) -> list[float]:
    return torch.linspace(0.0, maximum, depth).tolist() if depth > 1 else [maximum]


class MPGFER(nn.Module):
    """MPG-FER v2.1; no convolutional or dense-image transformer backbone."""

    def __init__(self, config: MPGConfig | None = None) -> None:
        super().__init__()
        self.config = config or MPGConfig()
        cfg = self.config
        self.pixel_extractor = PixelFeatureExtractor(img_size=cfg.img_size)
        self.pixel_topology = PixelGraphTopology(img_size=cfg.img_size)
        self.pixel_proj = nn.Sequential(
            nn.Linear(cfg.raw_pixel_dim, cfg.d_pixel), nn.LayerNorm(cfg.d_pixel), nn.GELU()
        )
        self.pixel_gnn = nn.ModuleList([
            EdgeAwarePixelGNNLayer(
                cfg.d_pixel, cfg.pixel_edge_dim, cfg.num_pixel_heads,
                cfg.pixel_dropout, drop_path=rate,
            )
            for rate in _linear_rates(cfg.num_pixel_gnn_layers, cfg.pixel_drop_path_max)
        ])
        self.pixel_attn_pool = nn.Linear(cfg.d_pixel, 1)
        self.pixel_readout_proj = nn.Sequential(
            nn.Linear(cfg.d_pixel * 3, cfg.d_pixel_readout),
            nn.LayerNorm(cfg.d_pixel_readout), nn.GELU(),
        )
        self.aux_pixel_head = nn.Linear(cfg.d_pixel_readout, cfg.num_classes)
        self.motif_composer = SpatialMotifComposer(
            d_pixel=cfg.d_pixel, num_motifs=cfg.num_motifs,
            motif_window_sizes=cfg.motif_window_sizes, motif_stride=cfg.motif_stride,
            img_size=cfg.img_size, d_type=cfg.d_type, d_motif=cfg.d_motif,
            tau_start=cfg.tau_start, tau_final=cfg.tau_final,
            tau_anneal_end_epoch=cfg.tau_anneal_end_epoch,
            mi_beta=cfg.mi_beta,
        )
        self.motif_gnn = nn.ModuleList([
            GeometryAwareMotifTransformerBlock(
                cfg.d_motif, cfg.motif_geom_dim, cfg.num_motif_heads,
                cfg.motif_dropout, drop_path=rate,
            )
            for rate in _linear_rates(cfg.num_motif_layers, cfg.motif_drop_path_max)
        ])
        self.motif_attn_pool = nn.Linear(cfg.d_motif, 1)
        self.motif_readout_proj = nn.Sequential(
            nn.Linear(cfg.d_motif * 3, cfg.d_motif_readout),
            nn.LayerNorm(cfg.d_motif_readout), nn.GELU(),
        )
        self.aux_motif_head = nn.Linear(cfg.d_motif_readout, cfg.num_classes)
        self.supcon_head = nn.Sequential(
            nn.Linear(cfg.d_classifier_in, cfg.supcon_dim),
            nn.LayerNorm(cfg.supcon_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(cfg.d_classifier_in, cfg.d_classifier_hidden),
            nn.LayerNorm(cfg.d_classifier_hidden), nn.GELU(),
            nn.Dropout(cfg.classifier_dropout),
            nn.Linear(cfg.d_classifier_hidden, cfg.num_classes),
        )

    def set_epoch_temperature(self, epoch: int) -> float:
        """Set the deterministic serialized motif temperature for an epoch."""
        return self.motif_composer.set_epoch_temperature(epoch)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch = x.shape[0]
        h = self.pixel_proj(self.pixel_extractor(x))
        intensities = x.reshape(batch, self.config.num_pixels, 1)
        edges = self.pixel_topology.compute_edge_features(intensities)
        for layer in self.pixel_gnn:
            h = layer(h, self.pixel_topology.neighbor_idx, self.pixel_topology.neighbor_mask, edges)
        p_mean, p_max = h.mean(dim=1), h.max(dim=1).values
        p_attention = (F.softmax(self.pixel_attn_pool(h), dim=1) * h).sum(dim=1)
        pixel_readout = self.pixel_readout_proj(torch.cat([p_mean, p_max, p_attention], dim=-1))
        pixel_logits = self.aux_pixel_head(pixel_readout)

        h_motif, assignments, diagnostics = self.motif_composer(h)
        geometry = compute_motif_geometry(
            diagnostics["learned_centers_x"], diagnostics["learned_centers_y"]
        )
        for layer in self.motif_gnn:
            h_motif = layer(h_motif, geometry)
        m_mean, m_max = h_motif.mean(dim=1), h_motif.max(dim=1).values
        m_attention = (F.softmax(self.motif_attn_pool(h_motif), dim=1) * h_motif).sum(dim=1)
        motif_readout = self.motif_readout_proj(torch.cat([m_mean, m_max, m_attention], dim=-1))
        motif_logits = self.aux_motif_head(motif_readout)
        fusion = torch.cat([pixel_readout, motif_readout], dim=-1)
        supcon_embeddings = F.normalize(self.supcon_head(fusion), dim=-1)
        logits = self.classifier(fusion)
        outputs = {
            "final_logits": logits,
            "pixel_logits": pixel_logits,
            "motif_logits": motif_logits,
            "h_pixel_readout": pixel_readout,
            "h_motif_readout": motif_readout,
            "fusion_representation": fusion,
            "supcon_embeddings": supcon_embeddings,
            "motif_assignments": assignments,
            "motif_geometry": geometry,
            **diagnostics,
        }
        return logits, outputs
