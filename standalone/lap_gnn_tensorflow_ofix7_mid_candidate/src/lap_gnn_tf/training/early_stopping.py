"""Exact OFIX7-mid validation-loss early stopping."""

from dataclasses import asdict, dataclass


@dataclass
class EarlyStoppingState:
    best: float
    epochs_without_improvement: int = 0


class ValidationLossEarlyStopping:
    def __init__(self, min_epochs: int = 30, patience: int = 15):
        self.min_epochs = int(min_epochs)
        self.patience = int(patience)
        self.state = EarlyStoppingState(best=float("inf"))

    def update(self, epoch: int, value: float) -> bool:
        value = float(value)
        if value < self.state.best:
            self.state.best = value
            self.state.epochs_without_improvement = 0
        else:
            self.state.epochs_without_improvement += 1
        return int(epoch) >= self.min_epochs and self.state.epochs_without_improvement >= self.patience

    def get_state(self) -> dict:
        return asdict(self.state)

    def set_state(self, state: dict) -> None:
        self.state = EarlyStoppingState(**state)

