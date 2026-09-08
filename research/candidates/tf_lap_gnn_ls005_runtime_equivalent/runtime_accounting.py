"""Technical-only Issue #60 log timing parser and lean-runtime projection."""

from __future__ import annotations

import re
import statistics
from pathlib import Path
from typing import Any


_TIMING = re.compile(
    r"time train=(?:(\d+)h)?(\d+)m(\d+)s "
    r"val=(?:(\d+)h)?(\d+)m(\d+)s "
    r"train_eval=(?:(\d+)h)?(\d+)m(\d+)s "
    r"total=(?:(\d+)h)?(\d+)m(\d+)s"
)


class RuntimeAccountingError(RuntimeError):
    """Raised when technical timing evidence is absent or malformed."""


def _seconds(groups: tuple[str | None, ...]) -> int:
    return int(groups[0] or 0) * 3600 + int(groups[1] or 0) * 60 + int(groups[2] or 0)


def analyze_issue60_runtime_log(log_path: str | Path) -> dict[str, Any]:
    """Use timing lines only; accuracy/F1 text is deliberately ignored."""

    source = Path(log_path)
    text = source.read_text(encoding="utf-8")
    rows = []
    for match in _TIMING.finditer(text):
        values = match.groups()
        train = _seconds(values[0:3])
        validation = _seconds(values[3:6])
        clean_train = _seconds(values[6:9])
        total = _seconds(values[9:12])
        rows.append(
            {
                "training_sec": train,
                "validation_sec": validation,
                "clean_train_evaluation_sec": clean_train,
                "total_sec": total,
                "other_overhead_sec": total - train - validation - clean_train,
            }
        )
    if not rows:
        raise RuntimeAccountingError("No completed-epoch timing lines found")

    def stats(key: str) -> dict[str, float]:
        values = [float(row[key]) for row in rows]
        return {
            "mean_sec": statistics.mean(values),
            "median_sec": statistics.median(values),
            "min_sec": min(values),
            "max_sec": max(values),
        }

    means = {
        key: statistics.mean(float(row[key]) for row in rows)
        for key in rows[0]
    }
    projected = {}
    for epochs in (32, 40, 45):
        seconds = epochs * (
            means["training_sec"]
            + means["validation_sec"]
            + means["other_overhead_sec"]
        )
        # Registered final endpoint: one clean-train pass plus one validation pass.
        seconds += means["clean_train_evaluation_sec"] + means["validation_sec"]
        projected[str(epochs)] = {
            "seconds": seconds,
            "hours": seconds / 3600.0,
        }
    return {
        "schema_version": 1,
        "source": "Issue #60 merged stdout/stderr timing lines only",
        "scientific_metrics_used": False,
        "completed_epochs_observed": len(rows),
        "training": stats("training_sec"),
        "clean_train_evaluation": stats("clean_train_evaluation_sec"),
        "validation": stats("validation_sec"),
        "other_lifecycle_overhead": stats("other_overhead_sec"),
        "projection_method": (
            "observed all-epoch mean train+validation+other per epoch, plus one "
            "final mean clean-train evaluation and one final mean validation"
        ),
        "projected_lean_wall_clock": projected,
    }
