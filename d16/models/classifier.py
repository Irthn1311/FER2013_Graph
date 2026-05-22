"""D16 classifier head."""

from __future__ import annotations

import torch


class D16Classifier(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 7, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
