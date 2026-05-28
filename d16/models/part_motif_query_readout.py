"""Part-conditioned latent motif-query readout for D16 pixel-GNN embeddings.

The motifs here are learned readout queries over node embeddings. They are not
semantic motif detectors and do not support causal, evidence, or semantic-region
claims.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F


def _valid_num_heads(hidden_dim: int, requested_heads: int) -> int:
    requested = max(1, int(requested_heads))
    for heads in range(min(requested, hidden_dim), 0, -1):
        if hidden_dim % heads == 0:
            return heads
    return 1


class PartMotifQueryReadout(torch.nn.Module):
    """Read node embeddings with multiple MediaPipe-guided part queries."""

    GROUPS: Dict[str, List[str]] = {
        "mouth": ["mouth", "left_mouth_corner", "right_mouth_corner"],
        "eye": ["left_eye", "right_eye"],
        "brow": ["left_brow", "right_brow"],
        "nose_cheek": ["nose", "left_cheek", "right_cheek"],
    }

    def __init__(
        self,
        part_names: Iterable[str],
        hidden_dim: int,
        output_dim: int,
        motif_counts: Dict[str, int] | None = None,
        lambda_part: float = 1.0,
        eps: float = 1e-6,
        use_cls_token: bool = True,
        use_motif_type_embedding: bool = True,
        transformer_layers: int = 1,
        transformer_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.2,
        residual_concat: bool = True,
        diagnostics: bool = True,
    ) -> None:
        super().__init__()
        self.part_names = list(part_names)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.lambda_part = float(lambda_part)
        self.eps = float(eps)
        self.use_cls_token = bool(use_cls_token)
        self.residual_concat = bool(residual_concat)
        self.diagnostics = bool(diagnostics)
        self.part_order = ["mouth", "eye", "brow", "nose_cheek", "global"]
        counts = motif_counts or {"mouth": 3, "eye": 3, "brow": 3, "nose_cheek": 1, "global": 2}
        self.motif_counts = {name: int(counts.get(name, 0)) for name in self.part_order}
        missing_counts = [name for name in self.part_order if self.motif_counts[name] <= 0]
        if missing_counts:
            raise ValueError(f"part_motif_query motif_counts must be positive for {missing_counts}")

        self.group_indices = {
            group_name: self._indices(names) for group_name, names in self.GROUPS.items()
        }
        missing_groups = [name for name, indices in self.group_indices.items() if not indices]
        if missing_groups:
            raise ValueError(f"Cannot infer D16 part indices for groups: {missing_groups}")

        motif_parts: List[str] = []
        motif_names: List[str] = []
        for part in self.part_order:
            for idx in range(self.motif_counts[part]):
                motif_parts.append(part)
                motif_names.append(f"{part}_{idx}")
        self.motif_parts = motif_parts
        self.motif_names = motif_names
        self.num_motifs = len(motif_names)
        self.register_buffer(
            "motif_part_index",
            torch.tensor([self.part_order.index(name) for name in motif_parts], dtype=torch.long),
            persistent=False,
        )

        self.queries = torch.nn.Parameter(torch.randn(self.num_motifs, self.hidden_dim) * 0.02)
        self.key_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)

        heads = _valid_num_heads(self.hidden_dim, int(transformer_heads))
        ff_dim = max(self.hidden_dim, int(round(self.hidden_dim * float(mlp_ratio))))
        layer = torch.nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=max(1, int(transformer_layers)),
            enable_nested_tensor=False,
        )
        self.cls_token = (
            torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
            if self.use_cls_token
            else None
        )
        self.motif_type_embedding = (
            torch.nn.Parameter(torch.zeros(1, self.num_motifs, self.hidden_dim))
            if bool(use_motif_type_embedding)
            else None
        )
        projection_input_dim = self.hidden_dim
        if self.residual_concat:
            projection_input_dim += len(self.part_order) * self.hidden_dim
        self.projection = torch.nn.Sequential(
            torch.nn.LayerNorm(projection_input_dim),
            torch.nn.Linear(projection_input_dim, self.output_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(self.output_dim, self.output_dim),
        )

    def _indices(self, names: Iterable[str]) -> List[int]:
        return [self.part_names.index(name) for name in names if name in self.part_names]

    def _stack_parts(self, part_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        missing = [name for name in self.part_order if name not in part_embeddings]
        if missing:
            raise KeyError(f"Missing D16 part embeddings for A3 residual concat: {missing}")
        return torch.stack([part_embeddings[name] for name in self.part_order], dim=1)

    def _group_prior(self, part_soft: torch.Tensor, group_name: str) -> torch.Tensor:
        if group_name == "global":
            return torch.ones((part_soft.size(0),), device=part_soft.device, dtype=part_soft.dtype)
        indices = self.group_indices.get(group_name)
        if not indices:
            raise KeyError(f"Missing D16 part indices for group {group_name!r}")
        return part_soft[:, indices].max(dim=1).values

    def _valid_group_mask(
        self,
        valid_part_groups: Dict[str, torch.Tensor] | torch.Tensor | None,
        num_graphs: int,
        device: torch.device,
    ) -> torch.Tensor:
        if valid_part_groups is None:
            return torch.ones((num_graphs, len(self.part_order)), device=device, dtype=torch.bool)
        if isinstance(valid_part_groups, torch.Tensor):
            mask = valid_part_groups.to(device=device, dtype=torch.bool)
            if tuple(mask.shape) != (num_graphs, len(self.part_order)):
                raise ValueError(
                    f"valid_part_groups tensor shape {tuple(mask.shape)} does not match {(num_graphs, len(self.part_order))}"
                )
            return mask
        masks = []
        for name in self.part_order:
            value = valid_part_groups.get(name)
            if value is None:
                raise KeyError(f"Missing valid_part_groups entry for A3 readout: {name}")
            masks.append(value.to(device=device, dtype=torch.bool))
        return torch.stack(masks, dim=1)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        batch_index: torch.Tensor,
        part_soft: torch.Tensor,
        num_graphs: int,
        part_embeddings: Dict[str, torch.Tensor] | None = None,
        valid_part_groups: Dict[str, torch.Tensor] | torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if node_embeddings.dim() != 2:
            raise ValueError(f"node_embeddings must be [N, H], got {tuple(node_embeddings.shape)}")
        if node_embeddings.size(1) != self.hidden_dim:
            raise ValueError(f"node embedding dim {node_embeddings.size(1)} does not match hidden_dim={self.hidden_dim}")
        if part_soft.dim() != 2:
            raise ValueError(f"part_soft must be [N, num_parts], got {tuple(part_soft.shape)}")
        if part_soft.size(0) != node_embeddings.size(0):
            raise ValueError("part_soft and node_embeddings must have the same node count")

        device = node_embeddings.device
        dtype = node_embeddings.dtype
        motif_tokens = node_embeddings.new_zeros((int(num_graphs), self.num_motifs, self.hidden_dim))
        entropy = node_embeddings.new_zeros((int(num_graphs), self.num_motifs))
        peak = node_embeddings.new_zeros((int(num_graphs), self.num_motifs))
        part_mass = node_embeddings.new_zeros((int(num_graphs), self.num_motifs))
        valid_groups = self._valid_group_mask(valid_part_groups, int(num_graphs), device)

        queries = self.queries.to(device=device, dtype=dtype)
        scale = 1.0 / math.sqrt(float(self.hidden_dim))
        for graph_id in range(int(num_graphs)):
            node_mask = batch_index == graph_id
            h_g = node_embeddings[node_mask]
            part_g = part_soft[node_mask]
            if h_g.numel() == 0:
                continue
            keys = self.key_proj(h_g)
            values = self.value_proj(h_g)
            content_scores = torch.matmul(queries, keys.transpose(0, 1)) * scale
            motif_outputs = []
            for motif_idx, group_name in enumerate(self.motif_parts):
                group_idx = self.part_order.index(group_name)
                if not bool(valid_groups[graph_id, group_idx].item()):
                    motif_outputs.append(values.new_zeros((self.hidden_dim,)))
                    continue
                prior = self._group_prior(part_g, group_name).to(dtype=dtype).clamp_min(self.eps)
                scores = content_scores[motif_idx] + self.lambda_part * torch.log(prior)
                alpha = torch.softmax(scores, dim=0)
                motif_outputs.append(torch.sum(values * alpha.unsqueeze(1), dim=0))
                safe_alpha = alpha.clamp_min(self.eps)
                entropy[graph_id, motif_idx] = -(alpha * torch.log(safe_alpha)).sum()
                peak[graph_id, motif_idx] = alpha.max()
                part_mass[graph_id, motif_idx] = torch.sum(alpha * prior.clamp(0.0, 1.0))
            motif_tokens[graph_id] = torch.stack(motif_outputs, dim=0)

        tokens = motif_tokens
        if self.motif_type_embedding is not None:
            tokens = tokens + self.motif_type_embedding.to(device=device, dtype=dtype)
        if self.cls_token is not None:
            cls = self.cls_token.to(device=device, dtype=dtype).expand(int(num_graphs), -1, -1)
            tokens_in = torch.cat([cls, tokens], dim=1)
        else:
            tokens_in = tokens
        transformed = self.encoder(tokens_in)
        transformed_motifs = transformed[:, 1:, :] if self.cls_token is not None else transformed
        if self.use_cls_token:
            z_motif = transformed[:, 0, :]
        else:
            z_motif = transformed_motifs.mean(dim=1)

        if self.residual_concat:
            if part_embeddings is None:
                raise ValueError("A3 residual_concat requires part_embeddings")
            residual = self._stack_parts(part_embeddings).flatten(start_dim=1)
            fused = torch.cat([z_motif, residual], dim=1)
        else:
            fused = z_motif
        z_image = self.projection(fused)

        token_norm = torch.linalg.vector_norm(motif_tokens, dim=-1)
        normalized = F.normalize(motif_tokens, p=2, dim=-1, eps=self.eps)
        sim = torch.matmul(normalized, normalized.transpose(1, 2))
        usage = token_norm / token_norm.sum(dim=1, keepdim=True).clamp_min(self.eps)
        effective = 1.0 / usage.square().sum(dim=1).clamp_min(self.eps)
        return {
            "z_image": z_image,
            "motif_tokens": motif_tokens,
            "motif_transformed_tokens": transformed_motifs,
            "motif_usage": usage,
            "motif_attention_entropy": entropy,
            "motif_attention_peak": peak,
            "motif_part_mass": part_mass,
            "motif_similarity": sim,
            "effective_motif_count": effective,
            "motif_part_index": self.motif_part_index.to(device=device),
        }
