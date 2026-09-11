"""Historical checkpointing and model selection contracts for Pure-GNN scientific runs."""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import tensorflow as tf


class CheckpointSelector:
    """Manages checkpoint selection according to historical earliest strict improvement rule."""

    def __init__(
        self,
        monitor: str,
        mode: str,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        if not monitor or not isinstance(monitor, str):
            raise ValueError(f"Explicit monitor metric string is required, got: {monitor}")
        if not mode or not isinstance(mode, str):
            raise ValueError(f"Explicit mode string ('max' or 'min') is required, got: {mode}")

        self.monitor = monitor
        self.mode = mode.lower()
        if self.mode not in ("max", "min"):
            raise ValueError(f"Unknown mode: {mode}. Expected 'max' or 'min'.")

        self.output_dir = Path(output_dir) if output_dir else None
        self.best_value = -float("inf") if self.mode == "max" else float("inf")
        self.selected_epoch_index_zero_based: Optional[int] = None
        self.selected_epoch_number_one_based: Optional[int] = None
        self.history: List[float] = []

    def is_better(self, value: float) -> bool:
        if self.mode == "max":
            return value > self.best_value
        else:
            return value < self.best_value

    def update(
        self,
        epoch_zero_based: int,
        metrics: Dict[str, float],
        model: Optional[tf.keras.Model] = None,
    ) -> bool:
        """Evaluates epoch metric and updates best weights checkpoint if strict improvement is achieved."""
        if self.monitor not in metrics:
            raise KeyError(f"Monitored metric '{self.monitor}' not found in provided metrics: {list(metrics.keys())}")

        val = float(metrics[self.monitor])
        if not math.isfinite(val):
            raise ValueError(f"Monitored metric '{self.monitor}' is not finite: {val}")

        self.history.append(val)
        improved = self.is_better(val)

        if improved:
            self.best_value = val
            self.selected_epoch_index_zero_based = epoch_zero_based
            self.selected_epoch_number_one_based = epoch_zero_based + 1
            if self.output_dir and model:
                ckpt_path = self.output_dir / f"best_{self.monitor}.weights.h5"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                model.save_weights(str(ckpt_path))

        return improved


class EarliestStrictCheckpoint(tf.keras.callbacks.Callback):
    """Keras Callback implementing historical earliest-strict-maximum checkpoint selection."""

    def __init__(
        self,
        output_dir: Union[str, Path],
        monitor: str,
        mode: str,
    ):
        super().__init__()
        self.selector = CheckpointSelector(monitor=monitor, mode=mode, output_dir=output_dir)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None):
        if logs is None:
            return
        self.selector.update(epoch, logs, self.model)
