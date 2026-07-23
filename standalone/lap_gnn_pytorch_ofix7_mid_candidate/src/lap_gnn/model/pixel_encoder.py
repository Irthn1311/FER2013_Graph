"""Pixel feature encoder for D16."""

from __future__ import annotations

import torch


class PixelEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
