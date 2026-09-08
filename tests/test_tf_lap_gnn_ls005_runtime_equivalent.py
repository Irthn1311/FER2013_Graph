from __future__ import annotations

import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import h5py
import numpy as np
import pytest
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROOT = ROOT / "standalone" / "lap_gnn_tensorflow_ofix7_mid_candidate"
FROZEN_SRC = FROZEN_ROOT / "src"
if str(FROZEN_SRC) not in sys.path:
    sys.path.insert(0, str(FROZEN_SRC))

from lap_gnn_tf.config import load_config  # noqa: E402
from lap_gnn_tf.training import evaluator, execution  # noqa: E402
from lap_gnn_tf.training.losses import sparse_cross_entropy as frozen_hard_ce  # noqa: E402
from research.candidates.tf_lap_gnn_ls005_rescue.loss_adapter import (  # noqa: E402
    smoothed_sparse_cross_entropy,
)
from research.candidates.tf_lap_gnn_ls005_runtime_equivalent import (  # noqa: E402
    prove_synthetic_lifecycle_equivalence,
)
from research.candidates.tf_lap_gnn_ls005_runtime_equivalent import (  # noqa: E402
    resume_audit,
    runtime_accounting,
    train_validation_only as candidate,
)


EXPECTED_PARENT = "6d89d17b2d3c39b7bf57084f9de4b707a92eb73e"
EXPECTED_PAYLOAD = "286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e"


def test_exact_parent_and_frozen_package_unchanged():
    assert subprocess.run(
        ["git", "cat-file", "-e", f"{EXPECTED_PARENT}^{{commit}}"],
        cwd=ROOT,
        check=False,
    ).returncode == 0
    changed = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            EXPECTED_PARENT,
            "--",
            FROZEN_ROOT.relative_to(ROOT).as_posix(),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    assert changed == ""
    assert candidate.verify_frozen_guards()["scientific_payload_sha256"] == EXPECTED_PAYLOAD


def test_lean_config_diff_is_only_registered_runtime_change_plus_ls005_metadata():
    frozen = load_config(candidate.FROZEN_CONFIG_PATH)
    lean = candidate.verify_candidate_config()
    assert candidate._deep_diff(frozen, lean) == candidate.EXPECTED_CONFIG_DIFF
    assert lean["loss"]["label_smoothing"] == 0.05
    assert lean["training"]["eval_train_metrics"] is False


def test_scientific_training_scheduler_early_stop_checkpoint_and_graph_are_frozen():
    frozen = load_config(candidate.FROZEN_CONFIG_PATH)
    lean = candidate.verify_candidate_config()
    for path in (
        ("seed",),
        ("model",),
        ("graph",),
        ("loss", "mode"),
        ("training", "gradient_execution_mode"),
        ("training", "optimizer_execution_mode"),
        ("training", "optimizer"),
        ("training", "lr"),
        ("training", "weight_decay"),
        ("training", "scheduler"),
        ("training", "early_stopping"),
        ("training", "checkpoint_policy"),
        ("training", "checkpoint_monitor"),
        ("training", "final_test_checkpoint"),
        ("training", "max_epochs"),
        ("training", "batch_size"),
        ("resources",),
        ("locked",),
    ):
        left, right = frozen, lean
        for key in path:
            left, right = left[key], right[key]
        assert right == left, path


def test_synthetic_lifecycle_equivalence_is_exact_for_three_epochs():
    proof = prove_synthetic_lifecycle_equivalence()
    assert proof["status"] == "PASS"
    assert proof["epochs_compared"] == 3
    assert proof["exact_equality"] is True
    assert proof["tolerance"] == {"floating": 0.0, "discrete": "exact"}
    assert proof["differing_epochs"] == []
    required = {
        "trainable_model_variables",
        "non_trainable_model_variables",
        "optimizer_variables_and_slots",
        "optimizer_iteration",
        "current_learning_rate",
        "scheduler_bookkeeping",
        "validation_metrics",
        "checkpoint_selection_state",
        "early_stopping_state",
        "next_epoch_training_batch_order",
        "next_epoch_augmentation_sequence",
    }
    assert set(proof["compared_state"]) == required
    for epoch, state in enumerate(proof["epochs"], start=1):
        assert state["epoch"] == epoch
        assert state["optimizer_iterations"] == epoch * 3
        assert state["next_epoch_data_state"]["epoch"] == epoch + 1
        assert len(state["next_epoch_data_state"]["order"]) == 12
        assert len(state["next_epoch_data_state"]["augmentation_sha256"]) == 64


def test_synthetic_bookkeeping_uses_registered_validation_quantities():
    proof = prove_synthetic_lifecycle_equivalence()
    validation_losses = []
    validation_accuracies = []
    for epoch, state in enumerate(proof["epochs"], start=1):
        validation_losses.append(state["validation"]["loss"])
        validation_accuracies.append(state["validation"]["accuracy"])
        assert state["scheduler_state"]["last_epoch"] == epoch
        assert state["scheduler_state"]["best"] == min(validation_losses)
        assert state["early_stopping_state"]["best"] == min(validation_losses)
        best_accuracy = max(validation_accuracies)
        best_epoch = validation_accuracies.index(best_accuracy) + 1
        assert state["checkpoint_state"]["best_accuracy"] == best_accuracy
        assert state["checkpoint_state"]["best_epoch"] == best_epoch


def test_per_epoch_clean_evaluator_absent_and_final_evaluators_follow_binding_restore(
    tmp_path, monkeypatch
):
    observed = {}
    monkeypatch.setattr(
        candidate,
        "verify_registered_runtime",
        lambda: {"gpu_count": 2, "gpu_names": ["Tesla T4", "Tesla T4"]},
    )

    def fake_frozen_run(config_path, fer_csv, prior_root, output_root, controls, **kwargs):
        config = load_config(config_path)
        observed["eval_train_metrics"] = config["training"]["eval_train_metrics"]
        observed["execution_during_training"] = execution.sparse_cross_entropy
        observed["evaluator_during_training"] = evaluator.sparse_cross_entropy
        observed["frozen_kwargs"] = kwargs
        Path(output_root).mkdir(parents=True)
        return {
            "training_validation_completed": True,
            "final_test_skipped": True,
            "test_accessed": False,
            "test_data_constructed": False,
            "test_checkpoint_loaded": False,
        }

    def fake_final(config, prior_root, output_root, controls):
        observed["execution_at_final_eval"] = execution.sparse_cross_entropy
        observed["evaluator_at_final_eval"] = evaluator.sparse_cross_entropy
        return {
            "selected_checkpoint": "best_val_accuracy.keras",
            "selected_epoch": 3,
            "checkpoint_reloaded_compile_false": True,
            "final_clean_train_hard_ce_evaluations": 1,
            "final_validation_hard_ce_evaluations": 1,
            "clean_train_augmentation": False,
            "test_access": False,
        }

    monkeypatch.setattr(
        candidate,
        "_load_frozen_wrapper",
        lambda: SimpleNamespace(run_validation_only=fake_frozen_run),
    )
    monkeypatch.setattr(candidate, "evaluate_selected_checkpoint_once", fake_final)
    controls = SimpleNamespace(clean_graph_cache_dir=None)
    marker = candidate.run_validation_only(
        candidate.CANDIDATE_CONFIG_PATH,
        tmp_path / "train.csv",
        tmp_path / "priors",
        tmp_path / "fresh-output",
        controls,
        limit_epochs=3,
        limit_train_batches=2,
        limit_val_batches=1,
    )
    assert observed["eval_train_metrics"] is False
    assert observed["frozen_kwargs"]["limit_train_eval_batches"] is None
    assert observed["execution_during_training"] is smoothed_sparse_cross_entropy
    assert observed["evaluator_during_training"] is frozen_hard_ce
    assert observed["execution_at_final_eval"] is frozen_hard_ce
    assert observed["evaluator_at_final_eval"] is frozen_hard_ce
    assert marker["per_epoch_clean_train_evaluations"] == 0
    assert marker["validation_evaluation_every_epoch"] is True
    assert marker["selected_checkpoint_evaluation"]["final_clean_train_hard_ce_evaluations"] == 1
    assert marker["selected_checkpoint_evaluation"]["final_validation_hard_ce_evaluations"] == 1
    assert execution.sparse_cross_entropy is frozen_hard_ce
    assert evaluator.sparse_cross_entropy is frozen_hard_ce


def test_binding_restores_when_frozen_training_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(
        candidate,
        "verify_registered_runtime",
        lambda: {"gpu_count": 2, "gpu_names": ["Tesla T4", "Tesla T4"]},
    )
    def fail(*args, **kwargs):
        assert execution.sparse_cross_entropy is smoothed_sparse_cross_entropy
        assert evaluator.sparse_cross_entropy is frozen_hard_ce
        raise RuntimeError("synthetic training failure")

    monkeypatch.setattr(
        candidate,
        "_load_frozen_wrapper",
        lambda: SimpleNamespace(run_validation_only=fail),
    )
    with pytest.raises(RuntimeError, match="synthetic training failure"):
        candidate.run_validation_only(
            candidate.CANDIDATE_CONFIG_PATH,
            tmp_path / "train.csv",
            tmp_path / "priors",
            tmp_path / "output",
            SimpleNamespace(clean_graph_cache_dir=None),
        )
    assert execution.sparse_cross_entropy is frozen_hard_ce
    assert evaluator.sparse_cross_entropy is frozen_hard_ce


class _FakeVariable:
    def __init__(self, value):
        self._value = np.asarray(value, dtype=np.float32)

    def numpy(self):
        return self._value


class _FakeModel:
    def __init__(self):
        self.variables = [_FakeVariable([1.0, 2.0])]
        self.trainable_variables = [object()] * candidate.EXPECTED_TRAINABLE_VARIABLE_COUNT

    def count_params(self):
        return candidate.EXPECTED_PARAMETER_COUNT


class _FakeGenerator:
    created = []

    def __init__(self, prior_root, split, config, *args, **kwargs):
        self.split = split
        self.config = config
        self.dataset = list(
            range(
                candidate.EXPECTED_TRAIN_SAMPLES
                if split == "train"
                else candidate.EXPECTED_VALIDATION_SAMPLES
            )
        )
        _FakeGenerator.created.append(self)

    def as_dataset(self, epoch, prefetch):
        return self.split


def test_selected_checkpoint_reloaded_then_clean_train_and_validation_hard_ce_once(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "checkpoints" / "best_val_accuracy.keras"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic")
    metadata = {"epoch": 4, "validation_metrics": {"accuracy": 0.6}}
    call_order = []
    monkeypatch.setattr(
        candidate,
        "_selected_checkpoint_proof",
        lambda output: (checkpoint, 4, metadata),
    )
    monkeypatch.setattr(
        candidate.tf.keras.models,
        "load_model",
        lambda path, compile: call_order.append(("load", Path(path), compile)) or _FakeModel(),
    )
    _FakeGenerator.created = []
    monkeypatch.setattr(candidate, "GraphBatchGenerator", _FakeGenerator)
    monkeypatch.setattr(
        candidate,
        "build_compiled_evaluation_step",
        lambda model: call_order.append(("build_hard_eval",)) or object(),
    )

    def fake_evaluate(model, batches, evaluate_step):
        call_order.append(("evaluate", batches))
        count = (
            candidate.EXPECTED_TRAIN_SAMPLES
            if batches == "train"
            else candidate.EXPECTED_VALIDATION_SAMPLES
        )
        return {
            "accuracy": 0.64 if batches == "val" else 0.72,
            "macro_f1": 0.60 if batches == "val" else 0.70,
            "loss": 1.1 if batches == "val" else 0.8,
            "support_per_class": [count, 0, 0, 0, 0, 0, 0],
        }

    monkeypatch.setattr(candidate, "evaluate_batches", fake_evaluate)
    config = candidate.verify_candidate_config()
    result = candidate.evaluate_selected_checkpoint_once(
        config,
        tmp_path / "priors",
        tmp_path,
        SimpleNamespace(
            eval_batch_size=32,
            graph_cache_size=64,
            graph_workers=2,
            clean_graph_cache_dir=tmp_path / "cache",
            tf_data_prefetch=2,
        ),
    )
    assert call_order == [
        ("load", checkpoint, False),
        ("build_hard_eval",),
        ("evaluate", "train"),
        ("evaluate", "val"),
    ]
    assert _FakeGenerator.created[0].config["graph"]["prior_corruption"]["enabled"] is False
    assert result["final_clean_train_hard_ce_evaluations"] == 1
    assert result["final_validation_hard_ce_evaluations"] == 1
    assert result["model_variables_immutable"] is True


@pytest.mark.parametrize(
    "path",
    [
        "/data/test/train.csv",
        "/data/testing/priors",
        "/data/test_split/cache",
        "/data/test-split/cache",
        "test.csv",
        "test_predictions.csv",
    ],
)
def test_test_paths_fail_closed_lexically(path):
    with pytest.raises(candidate.RuntimeEquivalentError, match="test path"):
        candidate.reject_explicit_test_path(path, "synthetic")


def test_non_test_paths_are_accepted_and_cli_has_no_test_or_clean_eval_limit():
    candidate.reject_explicit_test_path("/data/train/train.csv", "fer_csv")
    candidate.reject_explicit_test_path("/data/validation/priors", "prior_root")
    parser = candidate.build_parser()
    destinations = {action.dest for action in parser._actions}
    assert "test_csv" not in destinations
    assert "limit_test_batches" not in destinations
    assert "limit_train_eval_batches" not in destinations
    assert "allow_cpu_training" not in destinations


def test_registered_runtime_requires_exactly_two_t4_devices(monkeypatch):
    devices = [object(), object()]
    monkeypatch.setattr(candidate.tf.config, "list_physical_devices", lambda kind: devices)
    monkeypatch.setattr(
        candidate.tf.config.experimental,
        "get_device_details",
        lambda device: {"device_name": "Tesla T4"},
    )
    assert candidate.verify_registered_runtime() == {
        "gpu_count": 2,
        "gpu_names": ["Tesla T4", "Tesla T4"],
    }


@pytest.mark.parametrize(
    "names",
    [[], ["Tesla T4"], ["Tesla T4", "Tesla P100"]],
)
def test_registered_runtime_rejects_wrong_gpu_class(names, monkeypatch):
    devices = [object() for _ in names]
    name_by_id = {id(device): name for device, name in zip(devices, names)}
    monkeypatch.setattr(candidate.tf.config, "list_physical_devices", lambda kind: devices)
    monkeypatch.setattr(
        candidate.tf.config.experimental,
        "get_device_details",
        lambda device: {"device_name": name_by_id[id(device)]},
    )
    with pytest.raises(candidate.RuntimeEquivalentError, match="exactly two Tesla T4"):
        candidate.verify_registered_runtime()


def _write_synthetic_partial_artifacts(root: Path) -> None:
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(parents=True)
    h5_path = root / "model.weights.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset("optimizer/vars/0", data=np.asarray(31, dtype=np.int64))
        handle.create_dataset("layers/dense/vars/0", data=np.ones((2, 2), np.float32))
    with zipfile.ZipFile(checkpoints / "best_val_accuracy.keras", "w") as archive:
        archive.writestr("config.json", "{}")
        archive.write(h5_path, "model.weights.h5")
    h5_path.replace(checkpoints / "best_val_accuracy.weights.h5")
    (checkpoints / "best_val_accuracy.metadata.json").write_text(
        json.dumps(
            {
                "epoch": 31,
                "optimizer_state": {"iterations": 55645, "variable_count": 255, "learning_rate": 3e-4},
                "scheduler_state": {"last_epoch": 30, "num_bad_epochs": 0},
                "early_stopping_state": {"best": 1.0, "epochs_without_improvement": 0},
            }
        ),
        encoding="utf-8",
    )
    (root / "history.json").write_text(
        json.dumps({"epochs": [{"epoch": epoch} for epoch in range(1, 33)]}),
        encoding="utf-8",
    )


def test_partial_checkpoint_resume_audit_fails_closed_without_loading_model(tmp_path):
    _write_synthetic_partial_artifacts(tmp_path)
    audit = resume_audit.audit_issue60_partial_resume(tmp_path)
    assert audit["status"] == "ISSUE60_CHECKPOINT_NOT_AUTHORIZED_FOR_EXACT_RESUME"
    assert audit["checkpoint_loaded"] is False
    assert audit["model_weights_present"] is True
    assert audit["optimizer_state"]["optimizer_datasets_present"] is True
    assert audit["checkpoint_epoch"] == 31
    assert audit["last_completed_epoch"] == 32
    assert audit["data_generator_state_present"] is False
    assert audit["augmentation_rng_state_present"] is False
    assert audit["global_tensorflow_rng_state_present"] is False
    assert audit["exact_resume_proven"] is False
    assert "checkpoint_epoch_is_not_latest_completed_epoch" in audit["reasons"]


def test_runtime_accounting_uses_only_timing_and_projects_lean_lifecycle(tmp_path):
    log = tmp_path / "issue60.log"
    log.write_text(
        "accuracy=SECRET must be ignored\n"
        "time train=14m40s val=0m49s train_eval=6m28s total=21m57s\n"
        "macro_f1=SECRET must be ignored\n"
        "time train=15m00s val=0m49s train_eval=6m29s total=22m18s\n",
        encoding="utf-8",
    )
    result = runtime_accounting.analyze_issue60_runtime_log(log)
    assert result["scientific_metrics_used"] is False
    assert result["completed_epochs_observed"] == 2
    assert result["training"]["mean_sec"] == 890.0
    assert result["validation"]["mean_sec"] == 49.0
    assert result["clean_train_evaluation"]["mean_sec"] == 388.5
    expected_32 = 32 * (890.0 + 49.0) + 388.5 + 49.0
    assert result["projected_lean_wall_clock"]["32"]["seconds"] == expected_32


def test_fresh_cli_help_outside_repository_without_pythonpath(tmp_path):
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(Path(candidate.__file__).resolve()), "--help"],
        cwd=tmp_path,
        env=environment,
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Runtime-equivalent LS0.05" in completed.stdout


def test_new_training_path_contains_no_issue60_partial_checkpoint_reference():
    source = Path(candidate.__file__).read_text(encoding="utf-8")
    assert ".git/issue60" not in source.replace("\\", "/")
    assert "ISSUE60_CHECKPOINT" not in source
    assert "issue60_partial_checkpoint_loaded" in source


def test_no_pytorch_runtime_import():
    source_files = [
        candidate.CANDIDATE_ROOT / "train_validation_only.py",
        candidate.CANDIDATE_ROOT / "equivalence.py",
        candidate.CANDIDATE_ROOT / "resume_audit.py",
        candidate.CANDIDATE_ROOT / "runtime_accounting.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in source_files)
    assert "import torch" not in text
    assert "from torch" not in text
