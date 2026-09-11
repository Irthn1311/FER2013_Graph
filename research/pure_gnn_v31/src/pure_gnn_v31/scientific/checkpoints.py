"""Historical checkpointing and model selection contracts for Pure-GNN scientific runs."""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import tensorflow as tf


class CheckpointSelector:
    """Manages checkpoint selection according to historical earliest strict improvement rule."""

    def __init__(
        self,
        monitor: str = "val_accuracy",
        mode: str = "max",
        output_dir: Optional[Union[str, Path]] = None,
    ):
        self.monitor = monitor
        self.mode = mode.lower()
        if self.mode not in ("max", "min"):
            raise ValueError(f"Unknown mode: {mode}. Expected 'max' or 'min'.")

        self.output_dir = Path(output_dir) if output_dir else None
        self.best_value = -float("inf") if self.mode == "max" else float("inf")
        self.selected_epoch: Optional[int] = None
        self.history: List[float] = []

    def is_better(self, value: float) -> bool:
        if self.mode == "max":
            return value > self.best_value
        else:
            return value < self.best_value

    def update(self, epoch: int, metrics: Dict[str, float], model: Optional[tf.keras.Model] = None) -> bool:
        """Evaluates epoch metric and updates best checkpoint if strict improvement is achieved."""
        if self.monitor not in metrics:
            raise KeyError(f"Monitored metric '{self.monitor}' not found in provided metrics: {list(metrics.keys())}")

        val = float(metrics[self.monitor])
        if not math.isfinite(val):
            raise ValueError(f"Monitored metric '{self.monitor}' is not finite: {val}")

        self.history.append(val)
        improved = self.is_better(val)

        if improved:
            self.best_value = val
            self.selected_epoch = epoch
            if self.output_dir and model:
                ckpt_path = self.output_dir / f"best_{self.monitor}.keras"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                model.save(ckpt_path)

        return improved


class EarliestStrictCheckpoint(tf.keras.callbacks.Callback):
    """Keras Callback implementing historical earliest-strict-maximum checkpoint selection."""

    def __init__(
        self,
        output_dir: Union[str, Path],
        monitor: str = "val_accuracy",
        mode: str = "max",
    ):
        super().__init__()
        self.selector = CheckpointSelector(monitor=monitor, mode=mode, output_dir=output_dir)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None):
        if logs is None:
            return
        self.selector.update(epoch, logs, self.model)
