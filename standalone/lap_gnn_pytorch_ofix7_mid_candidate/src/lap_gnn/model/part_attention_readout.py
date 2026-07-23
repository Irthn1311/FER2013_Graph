"""Learned part-weighting readout for D16 pixel-GNN embeddings.

This module is a model readout only. It does not make motif, semantic-region,
causal-evidence, or interpretability claims.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import torch


class PartAttentionReadout(torch.nn.Module):
    """Fuse pooled D16 part embeddings with per-sample learned part weights."""

    def __init__(
        self,
        part_names: Iterable[str],
        hidden_dim: int,
        output_dim: int,
        attn_hidden_dim: int | None = None,
        dropout: float = 0.0,
        use_global_context: bool = True,
        fusion: str = "attn_plus_global",
        use_part_type_embedding: bool = True,
        return_attention: bool = True,
    ) -> None:
        super().__init__()
        self.part_names = list(part_names)
        if "global" not in self.part_names:
            raise ValueError("PartAttentionReadout requires a global part embedding")
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.use_global_context = bool(use_global_context)
        self.fusion = str(fusion)
        self.return_attention = bool(return_attention)

        score_input_dim = self.hidden_dim * (2 if self.use_global_context else 1)
        score_hidden_dim = int(attn_hidden_dim or hidden_dim)
        self.score_mlp = torch.nn.Sequential(
            torch.nn.Linear(score_input_dim, score_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(score_hidden_dim, 1),
        )
        self.part_type_embedding = (
            torch.nn.Parameter(torch.zeros(len(self.part_names), self.hidden_dim))
            if bool(use_part_type_embedding)
            else None
        )

        if self.fusion == "attn_only":
            projection_input_dim = self.hidden_dim
        elif self.fusion == "attn_plus_global":
            projection_input_dim = self.hidden_dim * 2
        elif self.fusion == "attn_plus_concat":
            projection_input_dim = self.hidden_dim * (len(self.part_names) + 1)
        else:
            raise ValueError(f"Unsupported part attention fusion={self.fusion!r}")

        self.projection = torch.nn.Sequential(
            torch.nn.LayerNorm(projection_input_dim),
            torch.nn.Linear(projection_input_dim, self.output_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(self.output_dim, self.output_dim),
        )

    def _stack_parts(self, part_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        missing = [name for name in self.part_names if name not in part_embeddings]
        if missing:
            raise KeyError(f"Missing D16 part embeddings for readout: {missing}")
        parts = [part_embeddings[name] for name in self.part_names]
        return torch.stack(parts, dim=1)

    def _valid_mask(
        self,
        valid_part_groups: Dict[str, torch.Tensor] | torch.Tensor | None,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        if valid_part_groups is None:
            return torch.ones(ref.shape[:2], device=ref.device, dtype=torch.bool)
        if isinstance(valid_part_groups, torch.Tensor):
            mask = valid_part_groups.to(device=ref.device, dtype=torch.bool)
            if tuple(mask.shape) != tuple(ref.shape[:2]):
                raise ValueError(f"valid_part_groups tensor shape {tuple(mask.shape)} does not match {tuple(ref.shape[:2])}")
            return mask
        masks: List[torch.Tensor] = []
        for name in self.part_names:
            value = valid_part_groups.get(name)
            if value is None:
                masks.append(torch.ones(ref.size(0), device=ref.device, dtype=torch.bool))
            else:
                masks.append(value.to(device=ref.device, dtype=torch.bool))
        return torch.stack(masks, dim=1)

    def forward(
        self,
        part_embeddings: Dict[str, torch.Tensor] | torch.Tensor,
        valid_part_groups: Dict[str, torch.Tensor] | torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(part_embeddings, torch.Tensor):
            parts = part_embeddings
            if parts.dim() != 3:
                raise ValueError(f"part_embeddings tensor must be [B, P, H], got {tuple(parts.shape)}")
            if parts.size(1) != len(self.part_names):
                raise ValueError(f"part_embeddings P={parts.size(1)} does not match {len(self.part_names)} part names")
        else:
            parts = self._stack_parts(part_embeddings)
        if parts.size(-1) != self.hidden_dim:
            raise ValueError(f"part embedding dim {parts.size(-1)} does not match hidden_dim={self.hidden_dim}")

        mask = self._valid_mask(valid_part_groups, parts)
        if self.part_type_embedding is not None:
            parts_for_score = parts + self.part_type_embedding.to(device=parts.device, dtype=parts.dtype).unsqueeze(0)
        else:
            parts_for_score = parts
        global_index = self.part_names.index("global")
        z_global = parts[:, global_index, :]
        if self.use_global_context:
            global_context = z_global.unsqueeze(1).expand(-1, parts.size(1), -1)
            score_input = torch.cat([parts_for_score, global_context], dim=-1)
        else:
            score_input = parts_for_score
        scores = self.score_mlp(score_input).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        alpha = torch.softmax(scores, dim=1)
        alpha = torch.where(mask, alpha, torch.zeros_like(alpha))
        alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-6)
        z_att = torch.sum(parts * alpha.unsqueeze(-1), dim=1)

        if self.fusion == "attn_only":
            fused = z_att
        elif self.fusion == "attn_plus_global":
            fused = torch.cat([z_att, z_global], dim=1)
        else:
            fused = torch.cat([z_att, parts.flatten(start_dim=1)], dim=1)
        z_image = self.projection(fused)
        return {
            "z_image": z_image,
            "part_attention_weights": alpha,
        }
