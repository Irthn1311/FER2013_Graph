"""External Step-10 learned local residual-slot candidate."""

from .model import (
    LearnedLocalResidualSlotLapGNN,
    LearnedLocalResidualSlotPool,
    build_candidate_model,
)

__all__ = [
    "LearnedLocalResidualSlotLapGNN",
    "LearnedLocalResidualSlotPool",
    "build_candidate_model",
]
