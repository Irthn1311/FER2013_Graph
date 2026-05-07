"""Motif-relation classifier for D9 end-to-end experiments."""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def compute_motif_geometry(motif_maps: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute soft center, area, and hard y-band masses from motif maps."""

    if motif_maps.ndim == 3:
        bsz, num_motifs, num_nodes = motif_maps.shape
        side = int(math.sqrt(float(num_nodes)))
        if side * side != int(num_nodes):
            raise ValueError(f"Cannot infer square map from N={num_nodes}")
        motif_maps = motif_maps.view(bsz, num_motifs, side, side)
    if motif_maps.ndim != 4:
        raise ValueError(f"motif_maps must be [B, K, H, W] or [B, K, N], got {tuple(motif_maps.shape)}")
    bsz, num_motifs, height, width = motif_maps.shape
    maps = motif_maps.clamp_min(0.0)
    flat = maps.flatten(2)
    norm = flat / flat.sum(dim=2, keepdim=True).clamp_min(1e-8)
    ys = torch.linspace(0.0, 1.0, height, device=maps.device, dtype=maps.dtype)
    xs = torch.linspace(0.0, 1.0, width, device=maps.device, dtype=maps.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    cx = (norm * xx.reshape(1, 1, -1)).sum(dim=2)
    cy = (norm * yy.reshape(1, 1, -1)).sum(dim=2)
    area = flat.mean(dim=2)

    y_idx = torch.arange(height, device=maps.device)
    upper_mask = (y_idx < height // 3).to(dtype=maps.dtype).view(1, 1, height, 1)
    middle_mask = ((y_idx >= height // 3) & (y_idx < (2 * height) // 3)).to(dtype=maps.dtype).view(1, 1, height, 1)
    lower_mask = (y_idx >= (2 * height) // 3).to(dtype=maps.dtype).view(1, 1, height, 1)
    denom = maps.sum(dim=(2, 3)).clamp_min(1e-8)
    upper = (maps * upper_mask).sum(dim=(2, 3)) / denom
    middle = (maps * middle_mask).sum(dim=(2, 3)) / denom
    lower = (maps * lower_mask).sum(dim=(2, 3)) / denom
    return {
        "centers": torch.stack([cx, cy], dim=-1).view(bsz, num_motifs, 2),
        "area": area,
        "region_masses": torch.stack([upper, middle, lower], dim=-1),
        "maps_4d": motif_maps,
        "maps_norm_flat": norm,
    }


class MotifRelationLayer(nn.Module):
    def __init__(self, hidden_dim: int, pair_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.pair_mlp = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.msg_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.score = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.update = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, pair_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, num_motifs, hidden_dim = h.shape
        pair_emb = self.pair_mlp(pair_features)
        h_i = h.unsqueeze(2).expand(bsz, num_motifs, num_motifs, hidden_dim)
        h_j = h.unsqueeze(1).expand(bsz, num_motifs, num_motifs, hidden_dim)
        msg = self.msg_mlp(torch.cat([h_j, pair_emb], dim=-1))
        score = self.score(torch.cat([h_i, h_j, pair_emb], dim=-1)).squeeze(-1)
        eye = torch.eye(num_motifs, device=h.device, dtype=torch.bool).view(1, num_motifs, num_motifs)
        score = score.masked_fill(eye, -1e4)
        attn = torch.softmax(score, dim=2)
        agg = (attn.unsqueeze(-1) * msg).sum(dim=2)
        return self.norm(h + self.update(torch.cat([h, agg], dim=-1))), attn


class D9MotifRelationClassifier(nn.Module):
    """Classify emotion from motif embeddings plus explicit motif-pair relations."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 7,
        layers: int = 2,
        dropout: float = 0.2,
        pooling: str = "attention",
        use_pair_features: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.layers = int(layers)
        self.pooling = str(pooling or "attention").lower()
        self.use_pair_features = bool(use_pair_features)
        if self.pooling not in {"attention", "selection_weighted", "mean"}:
            raise ValueError(f"Unsupported pooling={self.pooling!r}")
        node_dim = self.embedding_dim + 1 + 2 + 1 + 3
        self.node_mlp = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.pair_dim = 13
        self.relation_layers = nn.ModuleList(
            [MotifRelationLayer(self.hidden_dim, self.pair_dim, dropout=dropout) for _ in range(max(self.layers, 1))]
        )
        self.pool_score = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def forward(
        self,
        motif_embeddings: torch.Tensor,
        motif_maps: torch.Tensor,
        selection_weights: torch.Tensor | None = None,
        centers: torch.Tensor | None = None,
        area: torch.Tensor | None = None,
        region_masses: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if motif_embeddings.ndim != 3:
            raise ValueError(f"motif_embeddings must be [B, K, H], got {tuple(motif_embeddings.shape)}")
        if int(motif_embeddings.shape[-1]) != self.embedding_dim:
            raise ValueError(f"Expected embedding_dim={self.embedding_dim}, got {int(motif_embeddings.shape[-1])}")
        bsz, num_motifs, _ = motif_embeddings.shape
        geometry = compute_motif_geometry(motif_maps)
        centers = geometry["centers"] if centers is None else centers
        area = geometry["area"] if area is None else area
        region_masses = geometry["region_masses"] if region_masses is None else region_masses
        if selection_weights is None:
            selection_weights = motif_embeddings.new_full((bsz, num_motifs), 1.0 / max(num_motifs, 1))
        selection_weights = selection_weights.to(device=motif_embeddings.device, dtype=motif_embeddings.dtype)
        selection_weights = selection_weights / selection_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

        node_features = torch.cat(
            [
                motif_embeddings,
                selection_weights.unsqueeze(-1),
                centers.to(dtype=motif_embeddings.dtype),
                area.to(dtype=motif_embeddings.dtype).unsqueeze(-1),
                region_masses.to(dtype=motif_embeddings.dtype),
            ],
            dim=-1,
        )
        h = self.node_mlp(node_features)
        pair_features = self._pair_features(motif_embeddings, geometry["maps_norm_flat"], selection_weights, centers, area)
        relation_attn = None
        for layer in self.relation_layers:
            h, relation_attn = layer(h, pair_features)

        if self.pooling == "attention":
            scores = self.pool_score(h).squeeze(-1) + selection_weights.clamp_min(1e-8).log()
            pool_weights = torch.softmax(scores, dim=1)
        elif self.pooling == "selection_weighted":
            pool_weights = selection_weights
        else:
            pool_weights = h.new_full((bsz, num_motifs), 1.0 / max(num_motifs, 1))
        weighted = (h * pool_weights.unsqueeze(-1)).sum(dim=1)
        max_pool = h.max(dim=1).values
        logits = self.classifier(torch.cat([weighted, max_pool], dim=-1))
        return {
            "logits": logits,
            "motif_node_features": node_features,
            "motif_relation_features": pair_features,
            "motif_relation_attention": relation_attn,
            "motif_pool_weights": pool_weights,
            "centers": centers,
            "area": area,
            "region_masses": region_masses,
        }

    def _pair_features(
        self,
        motif_embeddings: torch.Tensor,
        maps_norm_flat: torch.Tensor,
        selection_weights: torch.Tensor,
        centers: torch.Tensor,
        area: torch.Tensor,
    ) -> torch.Tensor:
        ci = centers.unsqueeze(2)
        cj = centers.unsqueeze(1)
        dx = cj[..., 0] - ci[..., 0]
        dy = cj[..., 1] - ci[..., 1]
        dist = torch.sqrt(dx.square() + dy.square() + 1e-8)
        sin_angle = dy / dist
        cos_angle = dx / dist
        area_i = area.unsqueeze(2)
        area_j = area.unsqueeze(1)
        area_ratio = (area_j / area_i.clamp_min(1e-8)).clamp(0.0, 10.0)
        sel_i = selection_weights.unsqueeze(2)
        sel_j = selection_weights.unsqueeze(1)
        emb_i = F.normalize(motif_embeddings.float(), dim=-1, eps=1e-8).unsqueeze(2)
        emb_j = F.normalize(motif_embeddings.float(), dim=-1, eps=1e-8).unsqueeze(1)
        emb_cos = (emb_i * emb_j).sum(dim=-1).to(dtype=motif_embeddings.dtype)
        overlap = torch.bmm(maps_norm_flat, maps_norm_flat.transpose(1, 2)).to(dtype=motif_embeddings.dtype)
        pair_features = torch.stack(
            [
                dx,
                dy,
                dist,
                sin_angle,
                cos_angle,
                area_i.expand_as(dx),
                area_j.expand_as(dx),
                area_ratio,
                sel_i.expand_as(dx),
                sel_j.expand_as(dx),
                (sel_i * sel_j).expand_as(dx),
                emb_cos,
                overlap,
            ],
            dim=-1,
        ).to(dtype=motif_embeddings.dtype)
        return torch.nan_to_num(pair_features, nan=0.0, posinf=10.0, neginf=-10.0)


class D9PooledMotifMLPClassifier(nn.Module):
    """Simpler classifier head for testing whether MR message passing causes collapse."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 7,
        dropout: float = 0.2,
        pooling: str = "selection_weighted",
        **_: Any,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.pooling = str(pooling or "selection_weighted").lower()
        if self.pooling not in {"selection_weighted", "mean"}:
            raise ValueError(f"Unsupported pooled MLP pooling={self.pooling!r}")
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embedding_dim * 2),
            nn.Dropout(float(dropout)),
            nn.Linear(self.embedding_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def forward(
        self,
        motif_embeddings: torch.Tensor,
        motif_maps: torch.Tensor,
        selection_weights: torch.Tensor | None = None,
        centers: torch.Tensor | None = None,
        area: torch.Tensor | None = None,
        region_masses: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        del motif_maps, centers, area, region_masses
        if motif_embeddings.ndim != 3:
            raise ValueError(f"motif_embeddings must be [B, K, H], got {tuple(motif_embeddings.shape)}")
        if int(motif_embeddings.shape[-1]) != self.embedding_dim:
            raise ValueError(f"Expected embedding_dim={self.embedding_dim}, got {int(motif_embeddings.shape[-1])}")
        bsz, num_motifs, _ = motif_embeddings.shape
        if selection_weights is None or self.pooling == "mean":
            pool_weights = motif_embeddings.new_full((bsz, num_motifs), 1.0 / max(num_motifs, 1))
        else:
            pool_weights = selection_weights.to(device=motif_embeddings.device, dtype=motif_embeddings.dtype)
            pool_weights = pool_weights / pool_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        weighted = (motif_embeddings * pool_weights.unsqueeze(-1)).sum(dim=1)
        max_pool = motif_embeddings.max(dim=1).values
        logits = self.classifier(torch.cat([weighted, max_pool], dim=-1))
        return {
            "logits": logits,
            "motif_pool_weights": pool_weights,
            "motif_relation_attention": None,
        }


class D9ResidualPairRelationClassifier(nn.Module):
    """Pooled MLP baseline plus a small residual pair-relation branch."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 7,
        layers: int = 1,
        dropout: float = 0.2,
        pooling: str = "selection_weighted",
        alpha_init: float = 0.1,
        **_: Any,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.layers = int(layers)
        self.pooling = str(pooling or "selection_weighted").lower()
        if self.pooling not in {"selection_weighted", "mean"}:
            raise ValueError(f"Unsupported residual relation pooling={self.pooling!r}")
        self.base_classifier = nn.Sequential(
            nn.LayerNorm(self.embedding_dim * 2),
            nn.Dropout(float(dropout)),
            nn.Linear(self.embedding_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.num_classes),
        )
        pair_layers = []
        pair_dim = 13
        for layer_idx in range(max(self.layers, 1)):
            in_dim = pair_dim if layer_idx == 0 else self.hidden_dim
            pair_layers.extend(
                [
                    nn.LayerNorm(in_dim),
                    nn.Linear(in_dim, self.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
        self.pair_encoder = nn.Sequential(*pair_layers)
        self.relation_classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.num_classes),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(
        self,
        motif_embeddings: torch.Tensor,
        motif_maps: torch.Tensor,
        selection_weights: torch.Tensor | None = None,
        centers: torch.Tensor | None = None,
        area: torch.Tensor | None = None,
        region_masses: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        del region_masses
        if motif_embeddings.ndim != 3:
            raise ValueError(f"motif_embeddings must be [B, K, H], got {tuple(motif_embeddings.shape)}")
        if int(motif_embeddings.shape[-1]) != self.embedding_dim:
            raise ValueError(f"Expected embedding_dim={self.embedding_dim}, got {int(motif_embeddings.shape[-1])}")
        bsz, num_motifs, _ = motif_embeddings.shape
        geometry = compute_motif_geometry(motif_maps)
        centers = geometry["centers"] if centers is None else centers
        area = geometry["area"] if area is None else area
        if selection_weights is None or self.pooling == "mean":
            pool_weights = motif_embeddings.new_full((bsz, num_motifs), 1.0 / max(num_motifs, 1))
        else:
            pool_weights = selection_weights.to(device=motif_embeddings.device, dtype=motif_embeddings.dtype)
            pool_weights = pool_weights / pool_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

        weighted = (motif_embeddings * pool_weights.unsqueeze(-1)).sum(dim=1)
        max_pool = motif_embeddings.max(dim=1).values
        base_logits = self.base_classifier(torch.cat([weighted, max_pool], dim=-1))

        pair_features = self._pair_features(
            motif_embeddings=motif_embeddings,
            maps_norm_flat=geometry["maps_norm_flat"],
            selection_weights=pool_weights,
            centers=centers,
            area=area,
        )
        pair_h = self.pair_encoder(pair_features)
        eye = torch.eye(num_motifs, device=motif_embeddings.device, dtype=torch.bool).view(1, num_motifs, num_motifs, 1)
        pair_mask = (~eye).to(dtype=pair_h.dtype)
        pair_weight = (pool_weights.unsqueeze(2) * pool_weights.unsqueeze(1)).unsqueeze(-1) * pair_mask
        relation_summary = (pair_h * pair_weight).sum(dim=(1, 2)) / pair_weight.sum(dim=(1, 2)).clamp_min(1e-8)
        relation_logits = self.relation_classifier(relation_summary)
        logits = base_logits + self.alpha.to(dtype=base_logits.dtype) * relation_logits
        return {
            "logits": logits,
            "base_logits": base_logits,
            "relation_logits": relation_logits,
            "relation_alpha": self.alpha.detach(),
            "motif_relation_features": pair_features,
            "motif_pool_weights": pool_weights,
            "motif_relation_attention": None,
        }

    def _pair_features(
        self,
        motif_embeddings: torch.Tensor,
        maps_norm_flat: torch.Tensor,
        selection_weights: torch.Tensor,
        centers: torch.Tensor,
        area: torch.Tensor,
    ) -> torch.Tensor:
        ci = centers.unsqueeze(2)
        cj = centers.unsqueeze(1)
        dx = cj[..., 0] - ci[..., 0]
        dy = cj[..., 1] - ci[..., 1]
        dist = torch.sqrt(dx.square() + dy.square() + 1e-8)
        sin_angle = dy / dist
        cos_angle = dx / dist
        area_i = area.unsqueeze(2)
        area_j = area.unsqueeze(1)
        area_ratio = (area_j / area_i.clamp_min(1e-8)).clamp(0.0, 10.0)
        sel_i = selection_weights.unsqueeze(2)
        sel_j = selection_weights.unsqueeze(1)
        emb_i = F.normalize(motif_embeddings.float(), dim=-1, eps=1e-8).unsqueeze(2)
        emb_j = F.normalize(motif_embeddings.float(), dim=-1, eps=1e-8).unsqueeze(1)
        emb_cos = (emb_i * emb_j).sum(dim=-1).to(dtype=motif_embeddings.dtype)
        overlap = torch.bmm(maps_norm_flat, maps_norm_flat.transpose(1, 2)).to(dtype=motif_embeddings.dtype)
        pair_features = torch.stack(
            [
                dx,
                dy,
                dist,
                sin_angle,
                cos_angle,
                area_i.expand_as(dx),
                area_j.expand_as(dx),
                area_ratio,
                sel_i.expand_as(dx),
                sel_j.expand_as(dx),
                (sel_i * sel_j).expand_as(dx),
                emb_cos,
                overlap,
            ],
            dim=-1,
        ).to(dtype=motif_embeddings.dtype)
        return torch.nan_to_num(pair_features, nan=0.0, posinf=10.0, neginf=-10.0)
