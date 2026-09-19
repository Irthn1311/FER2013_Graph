"""Ranked Checkpoint Manager for Pixel GNN (FER2013_SGU Parity).

Keeps the top-K model checkpoints ranked by validation metrics (e.g. highest val_accuracy or lowest val_loss).
Automatically purges inferior checkpoints when exceeded, preserving disk space and logging top_k_rankings.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


class RankedCheckpointManager:
    """Manages top-K checkpoints saved as weights files (.weights.h5) ranked by an external metric."""

    def __init__(
        self,
        model: Any,
        directory: str | Path,
        max_to_keep: int = 5,
        metric_name: str = "val_accuracy",
        mode: str = "max",
    ) -> None:
        if mode not in {"max", "min"}:
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.model = model
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max(int(max_to_keep), 0)
        self.metric_name = str(metric_name)
        self.mode = mode
        self.metadata_path = self.directory / "top_k_rankings.json"
        self.entries: List[Dict[str, Any]] = self._load_existing_entries()
        self._sort_and_trim()

    @property
    def latest_checkpoint(self) -> Optional[str]:
        """Return the path to the current rank-1 checkpoint."""
        if not self.entries:
            return None
        return str(self.directory / self.entries[0]["checkpoint"])

    @property
    def threshold(self) -> Optional[float]:
        """Current metric value of the worst retained checkpoint in top-K."""
        if len(self.entries) < self.max_to_keep or not self.entries:
            return None
        return float(self.entries[-1]["metric"])

    def consider(
        self,
        epoch: int,
        metric: float,
        additional_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Consider saving current model weights if metric qualifies into top-K."""
        epoch = int(epoch)
        metric = float(metric)

        if self.max_to_keep <= 0:
            return {"saved": False, "reason": "disabled", "removed": None, "threshold": None}
        if not math.isfinite(metric):
            raise ValueError(f"Non-finite {self.metric_name} at epoch {epoch}: {metric}")

        checkpoint_name = f"ckpt-{epoch:03d}.weights.h5"

        # Compare with existing table excluding current epoch if previously present
        comparison_entries = [e for e in self.entries if e["epoch"] != epoch]
        threshold = None
        if len(comparison_entries) >= self.max_to_keep:
            sorted_comp = self._sorted(comparison_entries)
            threshold = float(sorted_comp[self.max_to_keep - 1]["metric"])
            qualifies = metric > threshold if self.mode == "max" else metric < threshold
        else:
            qualifies = True

        if not qualifies:
            return {
                "saved": False,
                "reason": "not_better_than_rank_k",
                "removed": None,
                "threshold": threshold,
            }

        # Save weights to disk
        target_path = self.directory / checkpoint_name
        self.model.save_weights(str(target_path))

        entry: Dict[str, Any] = {
            "epoch": epoch,
            "checkpoint": checkpoint_name,
            "metric": metric,
            "path": str(target_path),
        }
        if additional_metrics:
            for k, v in additional_metrics.items():
                if isinstance(v, (int, float, str, bool)):
                    entry[k] = v

        self.entries = comparison_entries + [entry]
        self.entries = self._sorted(self.entries)

        removed = None
        if len(self.entries) > self.max_to_keep:
            removed = self.entries.pop()
            removed_path = self.directory / removed["checkpoint"]
            if removed_path.is_file():
                try:
                    removed_path.unlink()
                except Exception as e:
                    print(f"[CHECKPOINT] Warning: failed to delete purged checkpoint {removed_path}: {e}")

        self._write_metadata()

        rank = next(i for i, item in enumerate(self.entries, 1) if item["epoch"] == epoch)
        return {
            "saved": True,
            "reason": "entered_top_k",
            "rank": rank,
            "checkpoint": str(target_path),
            "removed": removed,
            "threshold": self.threshold,
        }

    def _sorted(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        reverse = self.mode == "max"
        return sorted(
            entries,
            key=lambda item: (float(item["metric"]), int(item["epoch"])),
            reverse=reverse,
        )

    def _sort_and_trim(self) -> None:
        self.entries = self._sorted(self.entries)
        while len(self.entries) > self.max_to_keep:
            removed = self.entries.pop()
            removed_path = self.directory / removed["checkpoint"]
            if removed_path.is_file():
                removed_path.unlink(missing_ok=True)
        self._write_metadata()

    def _write_metadata(self) -> None:
        payload = {
            "metric_name": self.metric_name,
            "mode": self.mode,
            "max_to_keep": self.max_to_keep,
            "checkpoints": self.entries,
        }
        self.metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_existing_entries(self) -> List[Dict[str, Any]]:
        if not self.metadata_path.exists():
            return []
        try:
            data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if data.get("metric_name") != self.metric_name or data.get("mode") != self.mode:
                return []
            raw_entries = data.get("checkpoints", [])
            valid_entries = []
            for e in raw_entries:
                ckpt_file = self.directory / e.get("checkpoint", "")
                if ckpt_file.is_file():
                    valid_entries.append(e)
            return valid_entries
        except Exception:
            return []
