"""Part-token Transformer readout for D16 pixel-GNN embeddings.

This module learns interactions between pooled part tokens. It is a readout
mechanism only and does not make motif, causal-evidence, semantic-region, or
interpretability claims.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import torch


def _valid_num_heads(hidden_dim: int, requested_heads: int) -> int:
    requested = max(1, int(requested_heads))
    for heads in range(min(requested, hidden_dim), 0, -1):
        if hidden_dim % heads == 0:
            return heads
    return 1


class PartTokenTransformerReadout(torch.nn.Module):
    """Fuse D16 part tokens through a compact Transformer encoder."""

    def __init__(
        self,
        part_names: Iterable[str],
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.2,
        use_cls_token: bool = True,
        use_part_type_embedding: bool = True,
        pooling: str = "cls",
        residual_concat: bool = True,
    ) -> None:
        super().__init__()
        self.part_names = list(part_names)
        if "global" not in self.part_names:
            raise ValueError("PartTokenTransformerReadout requires a global part embedding")
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.use_cls_token = bool(use_cls_token)
        self.pooling = str(pooling)
        self.residual_concat = bool(residual_concat)
        if self.pooling == "cls" and not self.use_cls_token:
            raise ValueError("pooling='cls' requires use_cls_token=True")
        if self.pooling not in {"cls", "mean", "max"}:
            raise ValueError(f"Unsupported part-token pooling={self.pooling!r}")

        heads = _valid_num_heads(self.hidden_dim, int(num_heads))
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
            num_layers=max(1, int(num_layers)),
            enable_nested_tensor=False,
        )
        self.cls_token = (
            torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
            if self.use_cls_token
            else None
        )
        self.part_type_embedding = (
            torch.nn.Parameter(torch.zeros(1, len(self.part_names), self.hidden_dim))
            if bool(use_part_type_embedding)
            else None
        )
        projection_input_dim = self.hidden_dim
        if self.residual_concat:
            projection_input_dim += len(self.part_names) * self.hidden_dim
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
            raise KeyError(f"Missing D16 part embeddings for part-token readout: {missing}")
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
                raise ValueError(
                    f"valid_part_groups tensor shape {tuple(mask.shape)} does not match {tuple(ref.shape[:2])}"
                )
            return mask
        masks: List[torch.Tensor] = []
        for name in self.part_names:
            value = valid_part_groups.get(name)
            if value is None:
                raise KeyError(f"Missing valid_part_groups entry for part-token readout: {name}")
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
        valid_mask = self._valid_mask(valid_part_groups, parts)
        tokens = torch.where(valid_mask.unsqueeze(-1), parts, torch.zeros_like(parts))
        if self.part_type_embedding is not None:
            tokens = tokens + self.part_type_embedding.to(device=tokens.device, dtype=tokens.dtype)

        if self.cls_token is not None:
            cls = self.cls_token.to(device=tokens.device, dtype=tokens.dtype).expand(tokens.size(0), -1, -1)
            tokens_in = torch.cat([cls, tokens], dim=1)
            key_padding_mask = torch.cat(
                [torch.zeros((tokens.size(0), 1), device=tokens.device, dtype=torch.bool), ~valid_mask],
                dim=1,
            )
        else:
            tokens_in = tokens
            key_padding_mask = ~valid_mask

        transformed = self.encoder(tokens_in, src_key_padding_mask=key_padding_mask)
        transformed_parts = transformed[:, 1:, :] if self.cls_token is not None else transformed
        transformed_parts = torch.where(valid_mask.unsqueeze(-1), transformed_parts, torch.zeros_like(transformed_parts))
        if self.pooling == "cls":
            z_transformer = transformed[:, 0, :]
        elif self.pooling == "mean":
            denom = valid_mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=transformed_parts.dtype)
            z_transformer = transformed_parts.sum(dim=1) / denom
        else:
            masked = transformed_parts.masked_fill(~valid_mask.unsqueeze(-1), torch.finfo(transformed_parts.dtype).min)
            z_transformer = masked.max(dim=1).values

        if self.residual_concat:
            fused = torch.cat([z_transformer, parts.flatten(start_dim=1)], dim=1)
        else:
            fused = z_transformer
        z_image = self.projection(fused)
        return {
            "z_image": z_image,
            "part_token_original_tokens": parts,
            "part_token_transformed_tokens": transformed_parts,
            "part_token_valid_mask": valid_mask,
        }
