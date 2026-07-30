"""Atomic single validation-accuracy Keras checkpoint."""

from __future__ import annotations

import json
import os
from pathlib import Path


class CheckpointPolicy:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_macro = -float("inf")
        self.best_accuracy = -float("inf")
        self.best_macro_epoch = -1
        self.best_accuracy_epoch = -1

    def _save(self, model, optimizer, stem: str, metadata: dict) -> None:
        model_path = self.checkpoint_dir / f"{stem}.keras"
        weights_path = self.checkpoint_dir / f"{stem}.weights.h5"
        temp_model = self.checkpoint_dir / f".{stem}.keras"
        temp_weights = self.checkpoint_dir / f".{stem}.weights.h5"
        model.save(temp_model, include_optimizer=True)
        model.save_weights(temp_weights)
        os.replace(temp_model, model_path)
        os.replace(temp_weights, weights_path)
        metadata_path = self.checkpoint_dir / f"{stem}.metadata.json"
        temp_metadata = metadata_path.with_suffix(".json.tmp")
        temp_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temp_metadata, metadata_path)

    def update_best(self, model, optimizer, epoch: int, metrics: dict, metadata: dict) -> dict:
        saved = []
        macro = float(metrics["macro_f1"])
        accuracy = float(metrics["accuracy"])
        common = {**metadata, "epoch": int(epoch), "validation_metrics": metrics}
        if macro > self.best_macro:
            self.best_macro = macro
            self.best_macro_epoch = int(epoch)
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_accuracy_epoch = int(epoch)
            self._save(model, optimizer, "best_val_accuracy", common)
            saved.append("best_val_accuracy")
        return {"saved": saved, "best_macro_epoch": self.best_macro_epoch, "best_accuracy_epoch": self.best_accuracy_epoch}

    def update(self, model, optimizer, epoch: int, metrics: dict, metadata: dict) -> dict:
        return self.update_best(model, optimizer, epoch, metrics, metadata)
