from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PACKAGE_ROOT / "tools" / "audit_learning_history.py"
SPEC = importlib.util.spec_from_file_location("audit_learning_history", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
audit_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_module)


def _divergent_training_config():
    return {
        "checkpoint_monitor": "val_accuracy",
        "checkpoint_monitor_mode": "max",
        "metric_for_best_model": "val_accuracy",
        "best_metric": "val_accuracy",
        "final_test_checkpoint": "best_val_accuracy",
        "checkpoint_policy": {"monitor": "val_accuracy", "mode": "max"},
        "scheduler": {"monitor": "val_loss", "mode": "min"},
        "early_stopping": {"metric": "val_loss", "mode": "min"},
    }


def _separated_history():
    return {
        "epochs": [
            {
                "epoch": 1,
                "val_loss": 0.90,
                "val_accuracy": 0.50,
                "val_macro_f1": 0.45,
                "train_macro_f1": 0.48,
            },
            {
                "epoch": 2,
                "val_loss": 0.60,
                "val_accuracy": 0.65,
                "val_macro_f1": 0.55,
                "train_macro_f1": 0.58,
            },
            {
                "epoch": 3,
                "val_loss": 0.65,
                "val_accuracy": 0.72,
                "val_macro_f1": 0.60,
                "train_macro_f1": 0.63,
            },
            {
                "epoch": 4,
                "val_loss": 0.70,
                "val_accuracy": 0.70,
                "val_macro_f1": 0.66,
                "train_macro_f1": 0.70,
            },
        ]
    }


def _write_run(run_dir: Path, *, history=None, training=None, wrapped=False):
    run_dir.mkdir()
    if history is not False:
        (run_dir / "history.json").write_text(
            json.dumps(_separated_history() if history is None else history),
            encoding="utf-8",
        )
    if training is not False:
        config = {"training": _divergent_training_config() if training is None else training}
        if wrapped:
            config = {"resolved_config": config}
        (run_dir / "resolved_config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
    return run_dir


def _audit(tmp_path: Path, *, history=None, training=None, wrapped=False, name="audit"):
    run_dir = _write_run(
        tmp_path / "run", history=history, training=training, wrapped=wrapped
    )
    output_dir = tmp_path / name
    result = audit_module.audit_learning_history(run_dir, output_dir)
    return run_dir, output_dir, result


def test_complete_history_reports_independent_best_epochs_and_artifacts(tmp_path):
    _, output_dir, result = _audit(tmp_path)

    measurements = result["measurements"]
    assert measurements["total_observed_epochs"] == 4
    assert measurements["best_validation_loss"] == {"epoch": 2, "value": 0.60}
    assert measurements["best_validation_accuracy"] == {"epoch": 3, "value": 0.72}
    assert measurements["best_validation_macro_f1"] == {"epoch": 4, "value": 0.66}
    assert measurements["best_epoch_spread"] == 2
    assert measurements["train_macro_f1_at_best_validation_macro_f1"] == 0.70
    assert measurements["train_validation_macro_f1_gap_pp_at_best_validation_macro_f1"] == pytest.approx(4.0)
    assert result["interpretation"]["learning_behavior"] == "SMALL_GENERALIZATION_GAP"
    assert result["provenance"]["test_artifacts_read"] is False
    assert {path.name for path in output_dir.iterdir()} == set(
        audit_module.OUTPUT_FILENAMES
    )
    markdown = (output_dir / "learning_diagnosis.md").read_text(encoding="utf-8")
    assert "Raw measurements" in markdown
    assert "Heuristic interpretation" in markdown
    assert "No test artifact was read" in markdown


def test_large_train_validation_macro_f1_gap_is_a_signal_not_proof(tmp_path):
    history = _separated_history()
    history["epochs"][-1]["train_macro_f1"] = 0.86
    _, _, result = _audit(tmp_path, history=history)

    assert result["measurements"][
        "train_validation_macro_f1_gap_pp_at_best_validation_macro_f1"
    ] == pytest.approx(20.0)
    assert result["interpretation"]["learning_behavior"] == "GENERALIZATION_GAP_SIGNAL"
    assert result["thresholds"]["generalization_gap_pp"]["status"] == "heuristic_diagnostic_only"


def test_missing_train_macro_f1_at_best_epoch_remains_unknown(tmp_path):
    history = _separated_history()
    history["epochs"][-1]["train_macro_f1"] = None
    _, _, result = _audit(tmp_path, history=history)

    measurements = result["measurements"]
    assert measurements["train_macro_f1_at_best_validation_macro_f1"] is None
    assert measurements[
        "train_validation_macro_f1_gap_pp_at_best_validation_macro_f1"
    ] is None
    assert result["interpretation"]["learning_behavior"] == "UNKNOWN_TRAIN_EVAL_INCOMPLETE"


def test_aligned_monitor_policy(tmp_path):
    training = _divergent_training_config()
    training.update(
        {
            "checkpoint_monitor": "val_loss",
            "metric_for_best_model": "val_loss",
            "best_metric": "val_loss",
            "final_test_checkpoint": "best_val_loss",
        }
    )
    training["checkpoint_policy"]["monitor"] = "val_loss"
    _, _, result = _audit(tmp_path, training=training)

    assert result["interpretation"]["monitor_policy"] == "MONITORS_ALIGNED"
    assert result["policy"]["checkpoint_monitor"] == "val_loss"
    assert result["policy"]["scheduler_monitor"] == "val_loss"
    assert result["policy"]["early_stopping_monitor"] == "val_loss"


def test_divergent_monitor_policy_is_material_when_best_epochs_differ(tmp_path):
    _, _, result = _audit(tmp_path)

    assert result["interpretation"]["monitor_policy"] == "MATERIAL_POLICY_DRIFT"
    assert result["policy"]["monitor_best_epochs_when_available"] == {
        "checkpoint_monitor": 3,
        "scheduler_monitor": 2,
        "early_stopping_monitor": 2,
    }
    assert result["policy"]["final_test_checkpoint"] == "best_val_accuracy"
    assert result["policy"]["final_model_selection_metric"] == "val_accuracy"


@pytest.mark.parametrize(
    ("history", "message"),
    [
        ({"epochs": []}, "non-empty 'epochs' list"),
        ({"epochs": ["not-an-object"]}, "must be a JSON object"),
        ({"epochs": [{"epoch": 1, "val_loss": 0.5, "val_accuracy": 0.5}]}, "missing required validation fields"),
        (
            {
                "epochs": [
                    {
                        "epoch": 1,
                        "val_loss": float("nan"),
                        "val_accuracy": 0.5,
                        "val_macro_f1": 0.5,
                    }
                ]
            },
            "must be finite",
        ),
        (
            {
                "epochs": [
                    {"epoch": 1, "val_loss": 0.6, "val_accuracy": 0.5, "val_macro_f1": 0.5},
                    {"epoch": 1, "val_loss": 0.5, "val_accuracy": 0.6, "val_macro_f1": 0.6},
                ]
            },
            "duplicate epoch",
        ),
    ],
)
def test_malformed_required_validation_history_fails_closed(tmp_path, history, message):
    run_dir = _write_run(tmp_path / "run", history=history)
    with pytest.raises(audit_module.AuditInputError, match=message):
        audit_module.audit_learning_history(run_dir, tmp_path / "audit")


def test_missing_run_history_and_config_fail_explicitly(tmp_path):
    with pytest.raises(FileNotFoundError, match="run directory does not exist"):
        audit_module.audit_learning_history(tmp_path / "missing", tmp_path / "audit")

    no_history = _write_run(tmp_path / "no-history", history=False)
    with pytest.raises(FileNotFoundError, match="Missing TensorFlow training history"):
        audit_module.audit_learning_history(no_history, tmp_path / "audit-history")

    no_config = _write_run(tmp_path / "no-config", training=False)
    with pytest.raises(FileNotFoundError, match="Missing TensorFlow resolved config"):
        audit_module.audit_learning_history(no_config, tmp_path / "audit-config")


def test_audit_reads_only_history_and_config_not_test_or_model_artifacts(tmp_path, monkeypatch):
    run_dir = _write_run(tmp_path / "run")
    (run_dir / "best_val_accuracy.keras").write_bytes(b"not a real model")
    (run_dir / "test_metrics_best_val_accuracy.json").write_text(
        '{"accuracy": 1.0}', encoding="utf-8"
    )
    (run_dir / "predictions.csv").write_text("test_label\n0\n", encoding="utf-8")
    allowed = {
        (run_dir / "history.json").resolve(),
        (run_dir / "resolved_config.json").resolve(),
    }
    original_read_text = Path.read_text
    reads = []

    def guarded_read_text(path, *args, **kwargs):
        resolved = path.resolve()
        reads.append(resolved)
        if resolved not in allowed:
            raise AssertionError(f"unexpected source artifact read: {resolved}")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    result = audit_module.audit_learning_history(run_dir, tmp_path / "audit")

    assert reads == [run_dir / "history.json", run_dir / "resolved_config.json"]
    assert result["provenance"]["artifacts_read"] == [str(path) for path in reads]
    source = TOOL_PATH.read_bytes().decode("utf-8")
    assert "import tensorflow" not in source
    assert "load_model(" not in source


def _snapshot(directory: Path):
    return {
        path.relative_to(directory).as_posix(): (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_mtime_ns,
        )
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def test_source_run_is_unchanged_and_nested_output_is_rejected(tmp_path):
    run_dir = _write_run(tmp_path / "run")
    (run_dir / "checkpoints").mkdir()
    (run_dir / "checkpoints" / "best_val_accuracy.keras").write_bytes(b"model")
    before = _snapshot(run_dir)

    audit_module.audit_learning_history(run_dir, tmp_path / "audit")
    assert _snapshot(run_dir) == before

    with pytest.raises(audit_module.AuditInputError, match="outside the source run"):
        audit_module.audit_learning_history(run_dir, run_dir / "audit")
    assert _snapshot(run_dir) == before


def test_outputs_are_byte_deterministic_for_identical_inputs(tmp_path):
    run_dir = _write_run(tmp_path / "run")
    first = tmp_path / "audit-one"
    second = tmp_path / "audit-two"

    audit_module.audit_learning_history(run_dir, first)
    audit_module.audit_learning_history(run_dir, second)

    assert {
        name: (first / name).read_bytes() for name in audit_module.OUTPUT_FILENAMES
    } == {
        name: (second / name).read_bytes() for name in audit_module.OUTPUT_FILENAMES
    }


def test_partially_completed_run_and_supported_wrapped_config_are_accepted(tmp_path):
    _, _, result = _audit(tmp_path, wrapped=True)

    assert result["measurements"]["total_observed_epochs"] == 4
    assert result["provenance"]["test_artifacts_read"] is False


def test_malformed_resolved_config_fails_closed(tmp_path):
    run_dir = _write_run(tmp_path / "run")
    (run_dir / "resolved_config.json").write_text(
        json.dumps({"training": ["not", "an", "object"]}), encoding="utf-8"
    )

    with pytest.raises(audit_module.AuditInputError, match="must contain a training object"):
        audit_module.audit_learning_history(run_dir, tmp_path / "audit")


def test_conflicting_monitor_aliases_report_unknown_policy(tmp_path):
    training = _divergent_training_config()
    training["checkpoint_policy"]["monitor"] = "val_macro_f1"
    _, _, result = _audit(tmp_path, training=training)

    assert result["policy"]["checkpoint_monitor"] is None
    assert result["policy"]["configuration_conflicts"] == ["checkpoint_monitor"]
    assert result["interpretation"]["monitor_policy"] == "UNKNOWN_POLICY_INCOMPLETE"


def test_nonempty_output_directory_is_not_overwritten(tmp_path):
    run_dir = _write_run(tmp_path / "run")
    output_dir = tmp_path / "audit"
    output_dir.mkdir()
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="must be absent or empty"):
        audit_module.audit_learning_history(run_dir, output_dir)
    assert sentinel.read_text(encoding="utf-8") == "keep"
