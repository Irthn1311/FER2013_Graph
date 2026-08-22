"""Audit TensorFlow training/validation history without touching test artifacts.

The audit is deliberately implemented with the Python standard library.  It
does not import TensorFlow, load a checkpoint, run inference, or trigger
training.  Only ``history.json`` and ``resolved_config.json`` are read from the
source run directory.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


AUDIT_VERSION = "1.0.0"
OUTPUT_FILENAMES = (
    "learning_diagnosis.json",
    "learning_diagnosis.md",
    "epoch_metrics_validation_only.csv",
)

# These are convenience bands for a bounded diagnostic.  They are not
# scientific truth and must never be used to alter training or select a model.
GENERALIZATION_GAP_THRESHOLDS_PP = {
    "small_max_inclusive": 5.0,
    "large_min_inclusive": 10.0,
}

REQUIRED_VALIDATION_METRICS = ("val_loss", "val_accuracy", "val_macro_f1")
MONITOR_TO_MEASUREMENT = {
    "val_loss": "best_validation_loss",
    "val_accuracy": "best_validation_accuracy",
    "val_macro_f1": "best_validation_macro_f1",
}


class AuditInputError(ValueError):
    """Raised when run evidence is malformed or scientifically ambiguous."""


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AuditInputError(f"Malformed {description} JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AuditInputError(f"{description} must be a JSON object: {path}")
    return payload


def _finite_number(value: Any, field: str, row_index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuditInputError(
            f"history row {row_index} field {field!r} must be a finite number"
        )
    number = float(value)
    if not math.isfinite(number):
        raise AuditInputError(
            f"history row {row_index} field {field!r} must be finite"
        )
    return number


def _score(value: Any, field: str, row_index: int) -> float:
    number = _finite_number(value, field, row_index)
    if not 0.0 <= number <= 1.0:
        raise AuditInputError(
            f"history row {row_index} field {field!r} must be in [0, 1]"
        )
    return number


def _normalise_history(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_rows = payload.get("epochs")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise AuditInputError("history.json must contain a non-empty 'epochs' list")

    rows: list[dict[str, Any]] = []
    seen_epochs: set[int] = set()
    previous_epoch: int | None = None
    for index, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, dict):
            raise AuditInputError(f"history row {index} must be a JSON object")
        if "epoch" not in raw_row:
            raise AuditInputError(f"history row {index} is missing required field 'epoch'")
        epoch_value = raw_row["epoch"]
        if isinstance(epoch_value, bool) or not isinstance(epoch_value, int):
            raise AuditInputError(f"history row {index} field 'epoch' must be an integer")
        epoch = int(epoch_value)
        if epoch < 1:
            raise AuditInputError(f"history row {index} field 'epoch' must be >= 1")
        if epoch in seen_epochs:
            raise AuditInputError(f"history contains duplicate epoch number: {epoch}")
        if previous_epoch is not None and epoch <= previous_epoch:
            raise AuditInputError("history epochs must be in strictly increasing order")
        seen_epochs.add(epoch)
        previous_epoch = epoch

        missing = [name for name in REQUIRED_VALIDATION_METRICS if name not in raw_row]
        if missing:
            raise AuditInputError(
                f"history row {index} is missing required validation fields: {missing}"
            )
        row = {
            "epoch": epoch,
            "val_loss": _finite_number(raw_row["val_loss"], "val_loss", index),
            "val_accuracy": _score(raw_row["val_accuracy"], "val_accuracy", index),
            "val_macro_f1": _score(raw_row["val_macro_f1"], "val_macro_f1", index),
        }
        train_macro = raw_row.get("train_macro_f1")
        row["train_macro_f1"] = (
            None if train_macro is None else _score(train_macro, "train_macro_f1", index)
        )
        rows.append(row)
    return rows


def _training_mapping(config: Mapping[str, Any]) -> Mapping[str, Any]:
    candidates: list[tuple[str, Mapping[str, Any]]] = []
    direct = config.get("training")
    if isinstance(direct, dict):
        candidates.append(("training", direct))
    for wrapper_name in ("config", "resolved_config"):
        wrapper = config.get(wrapper_name)
        if isinstance(wrapper, dict) and isinstance(wrapper.get("training"), dict):
            candidates.append((f"{wrapper_name}.training", wrapper["training"]))
    if not candidates:
        raise AuditInputError(
            "resolved_config.json must contain a training object, directly or under "
            "'config'/'resolved_config'"
        )
    if len(candidates) > 1:
        first = candidates[0][1]
        if any(candidate != first for _, candidate in candidates[1:]):
            locations = [location for location, _ in candidates]
            raise AuditInputError(
                f"resolved config contains conflicting training objects: {locations}"
            )
    return candidates[0][1]


def _nested_value(mapping: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = mapping
    for part in path:
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _resolve_aliases(
    training: Mapping[str, Any], aliases: Sequence[tuple[str, Sequence[str]]]
) -> dict[str, Any]:
    represented: dict[str, str] = {}
    for label, path in aliases:
        value = _nested_value(training, path)
        if value is not None and str(value).strip():
            represented[label] = str(value).strip()
    unique = sorted({value.lower() for value in represented.values()})
    return {
        "value": unique[0] if len(unique) == 1 else None,
        "represented_fields": represented,
        "conflict": len(unique) > 1,
    }


def _extract_policy(config: Mapping[str, Any]) -> dict[str, Any]:
    training = _training_mapping(config)
    checkpoint = _resolve_aliases(
        training,
        (
            ("training.checkpoint_monitor", ("checkpoint_monitor",)),
            ("training.checkpoint_policy.monitor", ("checkpoint_policy", "monitor")),
        ),
    )
    scheduler = _resolve_aliases(
        training,
        (("training.scheduler.monitor", ("scheduler", "monitor")),),
    )
    early_stopping = _resolve_aliases(
        training,
        (
            ("training.early_stopping.monitor", ("early_stopping", "monitor")),
            ("training.early_stopping.metric", ("early_stopping", "metric")),
        ),
    )
    model_selection = _resolve_aliases(
        training,
        (
            ("training.metric_for_best_model", ("metric_for_best_model",)),
            ("training.best_metric", ("best_metric",)),
        ),
    )
    final_test_checkpoint = _nested_value(training, ("final_test_checkpoint",))
    return {
        "checkpoint_monitor": checkpoint["value"],
        "checkpoint_monitor_mode": _nested_value(training, ("checkpoint_monitor_mode",)),
        "scheduler_monitor": scheduler["value"],
        "scheduler_monitor_mode": _nested_value(training, ("scheduler", "mode")),
        "early_stopping_monitor": early_stopping["value"],
        "early_stopping_monitor_mode": _nested_value(training, ("early_stopping", "mode")),
        "final_test_checkpoint": (
            None if final_test_checkpoint is None else str(final_test_checkpoint)
        ),
        "final_model_selection_metric": model_selection["value"],
        "represented_fields": {
            "checkpoint_monitor": checkpoint["represented_fields"],
            "scheduler_monitor": scheduler["represented_fields"],
            "early_stopping_monitor": early_stopping["represented_fields"],
            "final_model_selection_metric": model_selection["represented_fields"],
        },
        "configuration_conflicts": sorted(
            name
            for name, result in (
                ("checkpoint_monitor", checkpoint),
                ("scheduler_monitor", scheduler),
                ("early_stopping_monitor", early_stopping),
                ("final_model_selection_metric", model_selection),
            )
            if result["conflict"]
        ),
    }


def _best(rows: Sequence[Mapping[str, Any]], field: str, *, minimise: bool) -> dict[str, Any]:
    chooser = min if minimise else max
    row = chooser(rows, key=lambda item: float(item[field]))
    return {"epoch": int(row["epoch"]), "value": float(row[field])}


def _gap_pp(train_macro_f1: Any, val_macro_f1: Any) -> float | None:
    if train_macro_f1 is None:
        return None
    return 100.0 * (float(train_macro_f1) - float(val_macro_f1))


def _learning_label(gap_pp: float | None) -> str:
    if gap_pp is None:
        return "UNKNOWN_TRAIN_EVAL_INCOMPLETE"
    if gap_pp <= GENERALIZATION_GAP_THRESHOLDS_PP["small_max_inclusive"]:
        return "SMALL_GENERALIZATION_GAP"
    if gap_pp < GENERALIZATION_GAP_THRESHOLDS_PP["large_min_inclusive"]:
        return "MODERATE_GENERALIZATION_GAP"
    return "GENERALIZATION_GAP_SIGNAL"


def _policy_interpretation(
    policy: Mapping[str, Any], measurements: Mapping[str, Any]
) -> tuple[str, dict[str, int], list[str]]:
    names = ("checkpoint_monitor", "scheduler_monitor", "early_stopping_monitor")
    active = {name: policy.get(name) for name in names}
    notes: list[str] = []
    if policy.get("configuration_conflicts"):
        notes.append("Conflicting aliases prevent a reliable monitor-policy classification.")
        return "UNKNOWN_POLICY_INCOMPLETE", {}, notes
    if any(value is None for value in active.values()):
        notes.append("One or more checkpoint/scheduler/early-stopping monitors are absent.")
        return "UNKNOWN_POLICY_INCOMPLETE", {}, notes
    if len(set(active.values())) == 1:
        return "MONITORS_ALIGNED", {}, notes

    monitor_best_epochs: dict[str, int] = {}
    for name, monitor in active.items():
        measurement_name = MONITOR_TO_MEASUREMENT.get(str(monitor))
        if measurement_name is not None:
            monitor_best_epochs[name] = int(measurements[measurement_name]["epoch"])
    if len(monitor_best_epochs) == len(active) and len(set(monitor_best_epochs.values())) > 1:
        notes.append(
            "Configured monitor objectives select different best epochs in the available history."
        )
        return "MATERIAL_POLICY_DRIFT", monitor_best_epochs, notes
    notes.append(
        "Monitor objectives diverge in configuration; available metrics do not show distinct "
        "best epochs for every configured monitor."
    )
    return "POLICY_DRIFT_PRESENT", monitor_best_epochs, notes


def _diagnose(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    best_loss = _best(rows, "val_loss", minimise=True)
    best_accuracy = _best(rows, "val_accuracy", minimise=False)
    best_macro = _best(rows, "val_macro_f1", minimise=False)
    best_epochs = [best_loss["epoch"], best_accuracy["epoch"], best_macro["epoch"]]
    macro_row = next(row for row in rows if row["epoch"] == best_macro["epoch"])
    final_row = rows[-1]
    gap_at_best = _gap_pp(macro_row["train_macro_f1"], macro_row["val_macro_f1"])
    final_gap = _gap_pp(final_row["train_macro_f1"], final_row["val_macro_f1"])
    measurements = {
        "total_observed_epochs": len(rows),
        "first_observed_epoch": int(rows[0]["epoch"]),
        "final_observed_epoch": int(final_row["epoch"]),
        "best_validation_loss": best_loss,
        "best_validation_accuracy": best_accuracy,
        "best_validation_macro_f1": best_macro,
        "best_epoch_spread": max(best_epochs) - min(best_epochs),
        "train_macro_f1_at_best_validation_macro_f1": macro_row["train_macro_f1"],
        "validation_macro_f1_at_best_validation_macro_f1": macro_row["val_macro_f1"],
        "train_validation_macro_f1_gap_pp_at_best_validation_macro_f1": gap_at_best,
        "final_train_macro_f1": final_row["train_macro_f1"],
        "final_validation_macro_f1": final_row["val_macro_f1"],
        "final_train_validation_macro_f1_gap_pp": final_gap,
    }
    policy = _extract_policy(config)
    policy_label, monitor_best_epochs, policy_notes = _policy_interpretation(
        policy, measurements
    )
    learning_label = _learning_label(gap_at_best)
    evidence_notes = list(policy_notes)
    if gap_at_best is None:
        evidence_notes.append(
            "Train macro-F1 is absent at the best validation macro-F1 epoch; "
            "generalization-gap interpretation remains UNKNOWN."
        )
    return {
        "measurements": measurements,
        "policy": {
            **policy,
            "monitor_best_epochs_when_available": monitor_best_epochs,
        },
        "interpretation": {
            "learning_behavior": learning_label,
            "monitor_policy": policy_label,
            "evidence_notes": evidence_notes,
        },
    }


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    fields = (
        "epoch",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
        "train_macro_f1",
        "train_validation_macro_f1_gap_pp",
    )
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                **row,
                "train_validation_macro_f1_gap_pp": _gap_pp(
                    row["train_macro_f1"], row["val_macro_f1"]
                ),
            }
        )
    return stream.getvalue()


def _display(value: Any) -> str:
    if value is None:
        return "UNKNOWN"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _markdown_text(result: Mapping[str, Any]) -> str:
    provenance = result["provenance"]
    measurements = result["measurements"]
    policy = result["policy"]
    interpretation = result["interpretation"]
    lines = [
        "# TensorFlow Learning-History Audit",
        "",
        "This is a read-only training/validation diagnostic. No test artifact was read.",
        "",
        "## Provenance",
        "",
        f"- Audit version: `{result['audit_version']}`",
        f"- Source run: `{provenance['source_run_dir']}`",
        f"- History: `{provenance['source_history_path']}`",
        f"- Resolved config: `{provenance['source_config_path']}`",
        f"- Test artifacts read: `{str(provenance['test_artifacts_read']).lower()}`",
        "",
        "## Raw measurements",
        "",
        f"- Total observed epochs: {measurements['total_observed_epochs']}",
        "- Best validation loss: "
        f"epoch {measurements['best_validation_loss']['epoch']}, "
        f"value {_display(measurements['best_validation_loss']['value'])}",
        "- Best validation accuracy: "
        f"epoch {measurements['best_validation_accuracy']['epoch']}, "
        f"value {_display(measurements['best_validation_accuracy']['value'])}",
        "- Best validation macro-F1: "
        f"epoch {measurements['best_validation_macro_f1']['epoch']}, "
        f"value {_display(measurements['best_validation_macro_f1']['value'])}",
        f"- Spread between best epochs: {measurements['best_epoch_spread']}",
        "- Train macro-F1 at best validation macro-F1: "
        f"{_display(measurements['train_macro_f1_at_best_validation_macro_f1'])}",
        "- Train-validation macro-F1 gap there: "
        f"{_display(measurements['train_validation_macro_f1_gap_pp_at_best_validation_macro_f1'])} pp",
        f"- Final train macro-F1: {_display(measurements['final_train_macro_f1'])}",
        f"- Final validation macro-F1: {_display(measurements['final_validation_macro_f1'])}",
        "- Final train-validation macro-F1 gap: "
        f"{_display(measurements['final_train_validation_macro_f1_gap_pp'])} pp",
        "",
        "## Configured monitor policy",
        "",
        f"- Checkpoint monitor: `{_display(policy['checkpoint_monitor'])}`",
        f"- Scheduler monitor: `{_display(policy['scheduler_monitor'])}`",
        f"- Early-stopping monitor: `{_display(policy['early_stopping_monitor'])}`",
        f"- Final-test checkpoint: `{_display(policy['final_test_checkpoint'])}`",
        "- Final model-selection metric: "
        f"`{_display(policy['final_model_selection_metric'])}`",
        "",
        "## Heuristic interpretation",
        "",
        f"- Learning behavior: **{interpretation['learning_behavior']}**",
        f"- Monitor policy: **{interpretation['monitor_policy']}**",
    ]
    notes = interpretation["evidence_notes"]
    lines.extend(["", "### Evidence notes", ""])
    lines.extend(f"- {note}" for note in notes)
    if not notes:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Scope boundary",
            "",
            "The gap bands are heuristic diagnostics, not scientific truth. This audit does "
            "not prove overfitting, identify a causal failure mode, justify a training/model "
            "change, or select a final model. It does not evaluate MediaPipe priors, graph "
            "structure, loss, regularization, or architecture.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="")
    os.replace(temporary, path)


def _validate_paths(run_dir: Path, output_dir: Path) -> tuple[Path, Path, Path, Path]:
    run_dir = run_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"TensorFlow run directory does not exist: {run_dir}")
    if output_dir == run_dir or run_dir in output_dir.parents:
        raise AuditInputError(
            "Output directory must be outside the source run directory to preserve it"
        )
    if output_dir.exists():
        if not output_dir.is_dir():
            raise AuditInputError(f"Output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError(f"Audit output directory must be absent or empty: {output_dir}")
    history_path = run_dir / "history.json"
    config_path = run_dir / "resolved_config.json"
    if not history_path.is_file():
        raise FileNotFoundError(f"Missing TensorFlow training history: {history_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing TensorFlow resolved config: {config_path}")
    return run_dir, output_dir, history_path, config_path


def audit_learning_history(run_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Create deterministic audit artifacts from an existing TensorFlow run."""

    run_dir, output_dir, history_path, config_path = _validate_paths(
        Path(run_dir), Path(output_dir)
    )
    history_payload = _read_json_object(history_path, "history")
    config = _read_json_object(config_path, "resolved config")
    rows = _normalise_history(history_payload)
    diagnosis = _diagnose(rows, config)
    result = {
        "schema_version": 1,
        "audit_version": AUDIT_VERSION,
        "provenance": {
            "source_run_dir": str(run_dir),
            "source_history_path": str(history_path),
            "source_config_path": str(config_path),
            "artifacts_read": [str(history_path), str(config_path)],
            "test_artifacts_read": False,
        },
        "thresholds": {
            "generalization_gap_pp": {
                **GENERALIZATION_GAP_THRESHOLDS_PP,
                "status": "heuristic_diagnostic_only",
                "usage_boundary": "not for training changes or final-model selection",
            },
            "material_policy_drift": {
                "rule": (
                    "configured checkpoint, scheduler, and early-stopping monitors diverge "
                    "and select more than one best epoch"
                )
            },
        },
        **diagnosis,
    }

    json_text = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    markdown_text = _markdown_text(result)
    csv_text = _csv_text(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(output_dir / OUTPUT_FILENAMES[0], json_text)
    _atomic_write(output_dir / OUTPUT_FILENAMES[1], markdown_text)
    _atomic_write(output_dir / OUTPUT_FILENAMES[2], csv_text)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read only TensorFlow history.json and resolved_config.json; never load "
            "weights, run inference/training, or read test artifacts."
        )
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = audit_learning_history(args.run_dir, args.output_dir)
    except (AuditInputError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
