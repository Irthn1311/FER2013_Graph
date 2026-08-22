"""Read-only learning-history audit for TensorFlow FER2013_Graph runs.

This tool deliberately does not import the model or trainer, does not load a
checkpoint, and does not read any test-set artifact. It only inspects training
history/configuration produced by an already completed (or partially completed)
TensorFlow run and writes the diagnosis to a separate output directory.

Primary question:
    Is the current ceiling more consistent with a generalization gap, an
    optimization/representation limitation, or a model-selection policy drift?

This is diagnostic evidence only; it must not be used to tune on the test set.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REQUIRED_HISTORY_KEYS = {
    "epoch",
    "train_loss",
    "val_loss",
    "val_accuracy",
    "val_macro_f1",
    "lr",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _argbest(rows: list[dict[str, Any]], key: str, mode: str) -> dict[str, Any] | None:
    candidates = [(row, _finite(row.get(key))) for row in rows]
    candidates = [(row, value) for row, value in candidates if value is not None]
    if not candidates:
        return None
    chooser = min if mode == "min" else max
    row, value = chooser(candidates, key=lambda item: item[1])
    return {"epoch": int(row["epoch"]), "value": float(value)}


def _row_for_epoch(rows: list[dict[str, Any]], epoch: int | None) -> dict[str, Any] | None:
    if epoch is None:
        return None
    matches = [row for row in rows if int(row["epoch"]) == int(epoch)]
    return matches[0] if len(matches) == 1 else None


def _pp(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return 100.0 * (left - right)


def _training_policy(config: dict[str, Any]) -> dict[str, Any]:
    training = dict(config.get("training", {}) or {})
    checkpoint_policy = dict(training.get("checkpoint_policy", {}) or {})
    scheduler = dict(training.get("scheduler", {}) or {})
    early = dict(training.get("early_stopping", {}) or {})
    return {
        "checkpoint_monitor": training.get("checkpoint_monitor"),
        "metric_for_best_model": training.get("metric_for_best_model"),
        "final_test_checkpoint": training.get("final_test_checkpoint"),
        "checkpoint_policy_type": checkpoint_policy.get("type"),
        "checkpoint_policy_monitor": checkpoint_policy.get("monitor"),
        "scheduler_monitor": scheduler.get("monitor"),
        "scheduler_type": scheduler.get("type"),
        "early_stopping_metric": early.get("metric"),
        "early_stopping_mode": early.get("mode"),
    }


def _policy_drift(policy: dict[str, Any]) -> dict[str, Any]:
    monitors = {
        "checkpoint": policy.get("checkpoint_monitor") or policy.get("checkpoint_policy_monitor"),
        "scheduler": policy.get("scheduler_monitor"),
        "early_stopping": policy.get("early_stopping_metric"),
    }
    active = {name: str(value) for name, value in monitors.items() if value not in (None, "")}
    unique = sorted(set(active.values()))
    return {
        "active_monitors": active,
        "unique_monitor_count": len(unique),
        "unique_monitors": unique,
        "multiple_objectives_present": len(unique) > 1,
        "scientific_primary_macro_f1_aligned": active.get("checkpoint") == "val_macro_f1",
    }


def _diagnose(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    best_loss = _argbest(rows, "val_loss", "min")
    best_acc = _argbest(rows, "val_accuracy", "max")
    best_macro = _argbest(rows, "val_macro_f1", "max")

    macro_row = _row_for_epoch(rows, None if best_macro is None else best_macro["epoch"])
    final_row = rows[-1]

    train_macro_at_best = _finite(None if macro_row is None else macro_row.get("train_macro_f1"))
    val_macro_at_best = _finite(None if macro_row is None else macro_row.get("val_macro_f1"))
    macro_gap_pp = _pp(train_macro_at_best, val_macro_at_best)

    final_train_macro = _finite(final_row.get("train_macro_f1"))
    final_val_macro = _finite(final_row.get("val_macro_f1"))
    final_gap_pp = _pp(final_train_macro, final_val_macro)

    epoch_spread = None
    if best_loss and best_acc and best_macro:
        epochs = [best_loss["epoch"], best_acc["epoch"], best_macro["epoch"]]
        epoch_spread = max(epochs) - min(epochs)

    policy = _training_policy(config)
    drift = _policy_drift(policy)

    warnings: list[str] = []
    if drift["multiple_objectives_present"]:
        warnings.append(
            "Checkpoint, scheduler, and early-stopping monitors are not aligned to one metric."
        )
    if not drift["scientific_primary_macro_f1_aligned"]:
        warnings.append(
            "Checkpoint selection is not aligned with validation macro-F1 as the scientific primary metric."
        )
    if epoch_spread is not None and epoch_spread >= 5:
        warnings.append(
            "Best validation loss, accuracy, and macro-F1 occur at materially different epochs."
        )
    if macro_gap_pp is not None and macro_gap_pp >= 10.0:
        warnings.append(
            "Train-validation macro-F1 gap at the best validation macro-F1 epoch is at least 10 percentage points."
        )

    if macro_gap_pp is None:
        learning_status = "UNKNOWN_TRAIN_EVAL_INCOMPLETE"
    elif macro_gap_pp >= 10.0:
        learning_status = "GENERALIZATION_GAP_SIGNAL"
    elif macro_gap_pp <= 5.0:
        learning_status = "SMALL_GENERALIZATION_GAP"
    else:
        learning_status = "MODERATE_GENERALIZATION_GAP"

    if drift["multiple_objectives_present"] and epoch_spread is not None and epoch_spread >= 5:
        checkpoint_status = "MATERIAL_POLICY_DRIFT"
    elif drift["multiple_objectives_present"]:
        checkpoint_status = "POLICY_DRIFT_PRESENT"
    else:
        checkpoint_status = "MONITORS_ALIGNED"

    return {
        "audit_version": "tf-learning-history-root-cause-v1",
        "epochs_observed": len(rows),
        "best_validation_loss": best_loss,
        "best_validation_accuracy": best_acc,
        "best_validation_macro_f1": best_macro,
        "best_epoch_spread": epoch_spread,
        "train_macro_f1_at_best_val_macro_f1": train_macro_at_best,
        "val_macro_f1_at_best_val_macro_f1": val_macro_at_best,
        "train_val_macro_f1_gap_pp_at_best_val_macro_f1": macro_gap_pp,
        "final_epoch": int(final_row["epoch"]),
        "final_train_macro_f1": final_train_macro,
        "final_val_macro_f1": final_val_macro,
        "final_train_val_macro_f1_gap_pp": final_gap_pp,
        "training_policy": policy,
        "policy_drift": drift,
        "learning_status": learning_status,
        "checkpoint_status": checkpoint_status,
        "warnings": warnings,
        "test_artifacts_read": False,
    }


def _write_epoch_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "epoch",
        "train_loss",
        "train_eval_loss",
        "train_accuracy",
        "train_macro_f1",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
        "train_val_macro_gap_pp",
        "lr",
        "prior_corruption_probability",
        "stop_requested",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fields})


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_markdown(path: Path, run_dir: Path, result: dict[str, Any]) -> None:
    lines = [
        "# TensorFlow Learning-History Audit",
        "",
        f"Run: `{run_dir}`",
        "",
        "This report is validation/training-history only. No test artifact was read.",
        "",
        "## Decision signals",
        "",
        f"- Learning status: **{result['learning_status']}**",
        f"- Checkpoint status: **{result['checkpoint_status']}**",
        f"- Epochs observed: {result['epochs_observed']}",
        f"- Best val-loss epoch: {_fmt((result['best_validation_loss'] or {}).get('epoch'))}",
        f"- Best val-accuracy epoch: {_fmt((result['best_validation_accuracy'] or {}).get('epoch'))}",
        f"- Best val-macro-F1 epoch: {_fmt((result['best_validation_macro_f1'] or {}).get('epoch'))}",
        f"- Best-epoch spread: {_fmt(result['best_epoch_spread'])}",
        f"- Train-val macro-F1 gap at best val macro-F1: {_fmt(result['train_val_macro_f1_gap_pp_at_best_val_macro_f1'])} pp",
        "",
        "## Monitor policy",
        "",
    ]
    for name, value in result["training_policy"].items():
        lines.append(f"- {name}: `{value}`")
    lines.extend(["", "## Warnings", ""])
    if result["warnings"]:
        lines.extend(f"- {warning}" for warning in result["warnings"])
    else:
        lines.append("- None from this bounded audit.")
    lines.extend(
        [
            "",
            "## Scope boundary",
            "",
            "This audit does not determine whether MediaPipe priors are correct, whether the model uses shortcut signals, or whether the graph representation is optimal. Those require separate diagnostics after this learning-history gate.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def audit(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    history_path = run_dir / "history.json"
    config_path = run_dir / "resolved_config.json"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing TensorFlow history: {history_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing resolved TensorFlow config: {config_path}")

    history_payload = _read_json(history_path)
    rows = history_payload.get("epochs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("history.json must contain a non-empty 'epochs' list")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"history row {index} is not an object")
        missing = sorted(REQUIRED_HISTORY_KEYS - set(row))
        if missing:
            raise ValueError(f"history row {index} missing required keys: {missing}")

    config = _read_json(config_path)
    result = _diagnose(rows, config)
    result["source_run_dir"] = str(run_dir.resolve())
    result["source_history"] = str(history_path.resolve())
    result["source_config"] = str(config_path.resolve())

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "learning_diagnosis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_epoch_csv(output_dir / "epoch_metrics_validation_only.csv", rows)
    _write_markdown(output_dir / "learning_diagnosis.md", run_dir, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.run_dir, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
