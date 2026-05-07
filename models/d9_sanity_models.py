"""Small D9 sanity classifiers that reuse the graph loader and feature masks."""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn


class D9SanityGlobalPoolClassifier(nn.Module):
    """Node MLP plus global mean/max pooling for feature-B sanity checks."""

    def __init__(
        self,
        node_dim: int = 3,
        hidden_dim: int = 64,
        num_classes: int = 7,
        dropout: float = 0.2,
        **_: Any,
    ) -> None:
        super().__init__()
        self.node_dim = int(node_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.node_mlp = nn.Sequential(
            nn.LayerNorm(self.node_dim),
            nn.Linear(self.node_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "D9SanityGlobalPoolClassifier":
        return cls(**dict(config))

    def forward(self, batch_or_x) -> Dict[str, torch.Tensor]:
        if isinstance(batch_or_x, dict):
            x = batch_or_x.get("x", batch_or_x.get("node_features"))
            node_mask = batch_or_x.get("node_mask")
        else:
            x = batch_or_x
            node_mask = None
        if x is None:
            raise KeyError("D9SanityGlobalPoolClassifier needs 'x' or 'node_features'")
        if x.ndim != 3:
            raise ValueError(f"x must be [B, N, D], got {tuple(x.shape)}")
        if int(x.shape[-1]) != self.node_dim:
            raise ValueError(f"Input node dim={int(x.shape[-1])} does not match model node_dim={self.node_dim}")

        h = self.node_mlp(x)
        if node_mask is not None:
            mask = node_mask.to(device=h.device, dtype=h.dtype).unsqueeze(-1)
            h_masked = h.masked_fill(mask <= 0, 0.0)
            denom = mask.sum(dim=1).clamp_min(1.0)
            mean_pool = h_masked.sum(dim=1) / denom
            max_pool = h.masked_fill(mask <= 0, -1e4).max(dim=1).values
        else:
            mean_pool = h.mean(dim=1)
            max_pool = h.max(dim=1).values
        logits = self.classifier(torch.cat([mean_pool, max_pool], dim=-1))
        return {"logits": logits, "h_pixel": h}
