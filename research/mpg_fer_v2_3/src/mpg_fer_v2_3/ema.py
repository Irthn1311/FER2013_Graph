"""Complete model-state exponential moving average."""

from __future__ import annotations

import copy
import torch
import torch.nn as nn


class ModelEMA:
    """Track parameters and buffers and expose deterministic evaluation state."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = float(decay)
        self.num_updates = 0
        self.module = copy.deepcopy(model).eval()
        self.module.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = model.state_dict()
        target = self.module.state_dict()
        parameter_names = set(dict(model.named_parameters()))
        if source.keys() != target.keys():
            raise RuntimeError("EMA/model state keys differ")
        for name, ema_value in target.items():
            value = source[name].detach()
            if name in parameter_names and torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(value, alpha=1.0 - self.decay)
            else:
                # Scheduled state such as current_tau is authoritative state,
                # not a quantity that EMA is allowed to numerically average.
                ema_value.copy_(value)
        self.num_updates += 1

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "model_state_dict": self.module.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state["decay"])
        self.num_updates = int(state["num_updates"])
        self.module.load_state_dict(state["model_state_dict"], strict=True)
