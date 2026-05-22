"""D16 regularizer stubs kept disabled for v0."""

from __future__ import annotations

import torch


def zero_evidence_regularizer(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_tensor(0.0)
