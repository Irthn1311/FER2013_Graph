"""PyTorch-compatible ReduceLROnPlateau state machine."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class PlateauState:
    best: float
    num_bad_epochs: int = 0
    cooldown_counter: int = 0
    last_epoch: int = 0


class TorchCompatibleReduceLROnPlateau:
    def __init__(
        self, optimizer, mode="min", factor=0.5, patience=5, threshold=1e-4,
        threshold_mode="rel", cooldown=0, min_lr=3e-5, eps=1e-8,
    ):
        self.optimizer = optimizer
        self.mode = str(mode)
        self.factor = float(factor)
        self.patience = int(patience)
        self.threshold = float(threshold)
        self.threshold_mode = str(threshold_mode)
        self.cooldown = int(cooldown)
        self.min_lr = float(min_lr)
        self.eps = float(eps)
        self.current_lr = float(self.optimizer.learning_rate.numpy())
        self.state = PlateauState(best=float("inf") if self.mode == "min" else -float("inf"))

    def _better(self, value: float) -> bool:
        best = self.state.best
        if self.mode == "min" and self.threshold_mode == "rel":
            return value < best * (1.0 - self.threshold)
        if self.mode == "min":
            return value < best - self.threshold
        if self.threshold_mode == "rel":
            return value > best * (1.0 + self.threshold)
        return value > best + self.threshold

    def step(self, metric: float) -> float:
        self.state.last_epoch += 1
        value = float(metric)
        if self._better(value):
            self.state.best = value
            self.state.num_bad_epochs = 0
        else:
            self.state.num_bad_epochs += 1
        if self.state.cooldown_counter > 0:
            self.state.cooldown_counter -= 1
            self.state.num_bad_epochs = 0
        if self.state.num_bad_epochs > self.patience:
            old_lr = self.current_lr
            new_lr = max(old_lr * self.factor, self.min_lr)
            if old_lr - new_lr > self.eps:
                self.optimizer.learning_rate.assign(new_lr)
                self.current_lr = new_lr
            self.state.cooldown_counter = self.cooldown
            self.state.num_bad_epochs = 0
        return self.current_lr

    def get_state(self) -> dict:
        return asdict(self.state)

    def set_state(self, state: dict) -> None:
        self.state = PlateauState(**state)
        self.current_lr = float(self.optimizer.learning_rate.numpy())
