"""Additive residual modules over the frozen NO_PIXEL_FUSION backbone."""

from __future__ import annotations

from enum import Enum
from typing import Any

import torch
import torch.nn as nn


class ResidualVariant(str, Enum):
    R0 = "R0"
    R1 = "R1"
    R2 = "R2"


class PixelDelta(nn.Module):
    def __init__(self, pixel_dim: int = 128, hidden_dim: int = 64, classes: int = 7) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(pixel_dim),
            nn.Linear(pixel_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, classes),
        )
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, pixel_readout: torch.Tensor) -> torch.Tensor:
        return self.network(pixel_readout)


class ConditionalGate(nn.Module):
    def __init__(
        self,
        pixel_dim: int = 128,
        motif_dim: int = 384,
        hidden_dim: int = 64,
        classes: int = 7,
    ) -> None:
        super().__init__()
        fusion_dim = pixel_dim + motif_dim
        self.network = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, classes),
            nn.Sigmoid(),
        )

    def forward(
        self, pixel_readout: torch.Tensor, motif_readout: torch.Tensor
    ) -> torch.Tensor:
        return self.network(torch.cat((pixel_readout, motif_readout), dim=-1))


class ResidualBranch(nn.Module):
    """R0/R1/R2 computation operating only on existing NPF outputs."""

    def __init__(self, variant: ResidualVariant | str) -> None:
        super().__init__()
        self.variant = ResidualVariant(variant)
        if self.variant is ResidualVariant.R0:
            self.pixel_delta = None
            self.gate = None
        else:
            self.pixel_delta = PixelDelta()
            self.gate = ConditionalGate() if self.variant is ResidualVariant.R2 else None

    def forward(
        self,
        base_logits: torch.Tensor,
        pixel_readout: torch.Tensor,
        motif_readout: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        if self.variant is ResidualVariant.R0:
            zeros = torch.zeros_like(base_logits)
            return base_logits, {"delta": zeros, "gate": None}
        assert self.pixel_delta is not None
        delta = self.pixel_delta(pixel_readout)
        if self.variant is ResidualVariant.R1:
            return base_logits + delta, {"delta": delta, "gate": None}
        assert self.gate is not None
        gate = self.gate(pixel_readout, motif_readout)
        return base_logits + gate * delta, {"delta": delta, "gate": gate}


class NPFResidualModel(nn.Module):
    """One-forward-model wrapper used for final frozen evaluation."""

    def __init__(self, frozen_npf: nn.Module, variant: ResidualVariant | str) -> None:
        super().__init__()
        self.backbone = frozen_npf
        self.residual = ResidualBranch(variant)
        self.freeze_backbone()

    def freeze_backbone(self) -> None:
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> "NPFResidualModel":
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        with torch.no_grad():
            base_logits, outputs = self.backbone(images)
        final_logits, residual = self.residual(
            base_logits,
            outputs["h_pixel_readout"],
            outputs["h_motif_readout"],
        )
        return final_logits, {
            "base_logits": base_logits,
            "final_logits": final_logits,
            "h_pixel_readout": outputs["h_pixel_readout"],
            "h_motif_readout": outputs["h_motif_readout"],
            **residual,
        }

    def allowed_trainable_names(self) -> set[str]:
        return {
            f"residual.{name}"
            for name, parameter in self.residual.named_parameters()
            if parameter.requires_grad
        }
