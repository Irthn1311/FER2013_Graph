from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROOT = ROOT / "standalone" / "lap_gnn_tensorflow_ofix7_mid_candidate"
FROZEN_SRC = FROZEN_ROOT / "src"
if str(FROZEN_SRC) not in sys.path:
    sys.path.insert(0, str(FROZEN_SRC))

from lap_gnn_tf.config import load_config  # noqa: E402
from lap_gnn_tf.graph.batch import load_golden_batch  # noqa: E402
from lap_gnn_tf.model import build_model  # noqa: E402
from lap_gnn_tf.signatures import scientific_payload_checksum  # noqa: E402
from lap_gnn_tf.training import evaluator, execution  # noqa: E402
from lap_gnn_tf.training.losses import sparse_cross_entropy as frozen_hard_ce  # noqa: E402
from research.candidates.tf_lap_gnn_ls005_rescue import (  # noqa: E402
    LABEL_SMOOTHING,
    NUM_CLASSES,
    hard_sparse_cross_entropy_compatibility,
    registered_smoothed_targets,
    smoothed_sparse_cross_entropy,
)
from research.candidates.tf_lap_gnn_ls005_rescue import (  # noqa: E402
    train_validation_only as candidate,
)
from research.candidates.tf_lap_gnn_ls005_rescue.loss_adapter import (  # noqa: E402
    training_loss_binding,
)


IMPLEMENTATION_BASE = "f8a183520ab1ef7d949e66b48ee7ce7e352f1cb6"
EXPECTED_PAYLOAD = "286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e"
TEST_FILE = Path(__file__).resolve()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_base_exists_and_frozen_package_has_zero_diff():
    assert subprocess.run(
        ["git", "cat-file", "-e", f"{IMPLEMENTATION_BASE}^{{commit}}"],
        cwd=ROOT,
        check=False,
    ).returncode == 0
    frozen_relative = FROZEN_ROOT.relative_to(ROOT).as_posix()
    changed = subprocess.run(
        ["git", "diff", "--name-only", IMPLEMENTATION_BASE, "--", frozen_relative],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    assert changed == ""
    assert scientific_payload_checksum(FROZEN_ROOT) == EXPECTED_PAYLOAD


def test_frozen_revision_and_payload_guards_pass():
    assert candidate.verify_frozen_guards() == {
        "trainer_sha256": candidate.EXPECTED_TRAINER_SHA256,
        "validation_only_wrapper_sha256": candidate.EXPECTED_FROZEN_WRAPPER_SHA256,
        "scientific_payload_sha256": EXPECTED_PAYLOAD,
    }


def test_loaded_config_diff_is_exactly_run_name_and_label_smoothing():
    frozen = load_config(candidate.FROZEN_CONFIG_PATH)
    rescue = candidate.verify_candidate_config()
    assert candidate._deep_diff(frozen, rescue) == candidate.EXPECTED_CONFIG_DIFF
    assert rescue["loss"]["label_smoothing"] == 0.05

    frozen_without_metadata = json.loads(json.dumps(frozen))
    rescue_without_metadata = json.loads(json.dumps(rescue))
    rescue_without_metadata["run_name"] = frozen_without_metadata["run_name"]
    rescue_without_metadata["loss"]["label_smoothing"] = frozen_without_metadata[
        "loss"
    ]["label_smoothing"]
    assert rescue_without_metadata == frozen_without_metadata


def test_nonregistered_config_path_fails_closed(tmp_path):
    copied = tmp_path / "copied.yaml"
    copied.write_bytes(candidate.CANDIDATE_CONFIG_PATH.read_bytes())
    with pytest.raises(candidate.LS005RescueError, match="Only the registered"):
        candidate.verify_candidate_config(copied)


def test_optimizer_scheduler_early_stop_checkpoint_and_prior_are_frozen():
    frozen = load_config(candidate.FROZEN_CONFIG_PATH)
    rescue = candidate.verify_candidate_config()
    for path in (
        ("seed",),
        ("training", "optimizer"),
        ("training", "lr"),
        ("training", "weight_decay"),
        ("training", "scheduler"),
        ("training", "early_stopping"),
        ("training", "checkpoint_policy"),
        ("training", "checkpoint_monitor"),
        ("training", "final_test_checkpoint"),
        ("training", "max_epochs"),
        ("graph", "prior_corruption"),
        ("locked",),
        ("resources",),
    ):
        frozen_value = frozen
        rescue_value = rescue
        for key in path:
            frozen_value = frozen_value[key]
            rescue_value = rescue_value[key]
        assert rescue_value == frozen_value, path


def test_registered_epsilon_classes_and_target_formula():
    assert LABEL_SMOOTHING == 0.05
    assert NUM_CLASSES == 7
    labels = tf.constant([0, 3, 6], dtype=tf.int32)
    actual = registered_smoothed_targets(labels).numpy()
    expected = (1.0 - 0.05) * np.eye(7, dtype=np.float32)[[0, 3, 6]] + 0.05 / 7
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(actual.sum(axis=1), np.ones(3), rtol=0, atol=2e-7)
    np.testing.assert_allclose(
        actual[np.arange(3), [0, 3, 6]],
        np.full(3, 1.0 - 0.05 + 0.05 / 7),
        rtol=0,
        atol=1e-7,
    )


def test_smoothed_ce_matches_independent_manual_calculation():
    labels = tf.constant([1, 5], dtype=tf.int32)
    logits = tf.constant(
        [[-0.5, 1.2, 0.3, -0.7, 0.0, 0.1, 0.8],
         [0.4, -0.1, 0.2, 0.5, -0.3, 1.1, -0.8]],
        dtype=tf.float16,
    )
    actual = float(smoothed_sparse_cross_entropy(labels, logits).numpy())
    logits64 = logits.numpy().astype(np.float32).astype(np.float64)
    shifted = logits64 - logits64.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    targets = (1.0 - 0.05) * np.eye(7)[[1, 5]] + 0.05 / 7
    expected = float(np.mean(-np.sum(targets * log_probs, axis=1)))
    assert actual == pytest.approx(expected, rel=0, abs=2e-7)


def test_epsilon_zero_control_matches_frozen_hard_ce():
    labels = tf.constant([0, 4, 2], dtype=tf.int32)
    logits = tf.constant(
        [[0.2, -0.1, 0.5, 0.7, -0.4, 0.0, 0.9],
         [-0.2, 0.8, 0.1, -0.6, 1.2, 0.4, 0.3],
         [0.5, 0.3, 1.0, -0.2, 0.1, -0.8, 0.7]],
        dtype=tf.float32,
    )
    expected = frozen_hard_ce(labels, logits)
    actual = hard_sparse_cross_entropy_compatibility(labels, logits)
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=0, atol=1e-7)


def test_only_execution_binding_is_replaced_and_normal_return_restores():
    original_execution = execution.sparse_cross_entropy
    original_evaluator = evaluator.sparse_cross_entropy
    with training_loss_binding():
        assert execution.sparse_cross_entropy is smoothed_sparse_cross_entropy
        assert evaluator.sparse_cross_entropy is original_evaluator
        assert evaluator.sparse_cross_entropy is frozen_hard_ce
    assert execution.sparse_cross_entropy is original_execution
    assert evaluator.sparse_cross_entropy is original_evaluator


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_execution_binding_restored_after_failure_or_interrupt(error_type):
    original_execution = execution.sparse_cross_entropy
    original_evaluator = evaluator.sparse_cross_entropy
    with pytest.raises(error_type):
        with training_loss_binding():
            assert execution.sparse_cross_entropy is smoothed_sparse_cross_entropy
            raise error_type("synthetic failure")
    assert execution.sparse_cross_entropy is original_execution
    assert evaluator.sparse_cross_entropy is original_evaluator


def test_candidate_wrapper_runs_frozen_lifecycle_under_patch_and_restores(
    tmp_path, monkeypatch
):
    observed: dict[str, object] = {}

    def fake_run(config_path, fer_csv, prior_root, output_root, controls, **kwargs):
        observed["execution"] = execution.sparse_cross_entropy
        observed["evaluator"] = evaluator.sparse_cross_entropy
        observed["args"] = (config_path, fer_csv, prior_root, output_root, controls)
        observed["kwargs"] = kwargs
        Path(output_root).mkdir(parents=True)
        return {"final_test_skipped": True, "test_accessed": False}

    fake_wrapper = SimpleNamespace(run_validation_only=fake_run)
    monkeypatch.setattr(candidate, "_load_frozen_wrapper", lambda: fake_wrapper)
    output_root = tmp_path / "output"
    controls = SimpleNamespace()
    marker = candidate.run_validation_only(
        candidate.CANDIDATE_CONFIG_PATH,
        tmp_path / "fer.csv",
        tmp_path / "priors",
        output_root,
        controls,
        no_resume=True,
        limit_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        limit_train_eval_batches=1,
    )

    assert observed["execution"] is smoothed_sparse_cross_entropy
    assert observed["evaluator"] is frozen_hard_ce
    assert execution.sparse_cross_entropy is frozen_hard_ce
    assert evaluator.sparse_cross_entropy is frozen_hard_ce
    assert marker["training_execution_binding_replaced"] is True
    assert marker["training_execution_binding_restored"] is True
    assert marker["evaluator_hard_ce_binding_unchanged"] is True
    assert marker["test_accessed"] is False
    saved = json.loads(
        (output_root / candidate.CANDIDATE_MARKER_NAME).read_text(encoding="utf-8")
    )
    assert saved == marker


def test_candidate_wrapper_failure_restores_without_marker(tmp_path, monkeypatch):
    def fail(*_args, **_kwargs):
        assert execution.sparse_cross_entropy is smoothed_sparse_cross_entropy
        assert evaluator.sparse_cross_entropy is frozen_hard_ce
        raise RuntimeError("synthetic frozen-wrapper failure")

    monkeypatch.setattr(
        candidate,
        "_load_frozen_wrapper",
        lambda: SimpleNamespace(run_validation_only=fail),
    )
    output_root = tmp_path / "output"
    with pytest.raises(RuntimeError, match="synthetic frozen-wrapper failure"):
        candidate.run_validation_only(
            candidate.CANDIDATE_CONFIG_PATH,
            tmp_path / "fer.csv",
            tmp_path / "priors",
            output_root,
            SimpleNamespace(),
        )
    assert execution.sparse_cross_entropy is frozen_hard_ce
    assert evaluator.sparse_cross_entropy is frozen_hard_ce
    assert not (output_root / candidate.CANDIDATE_MARKER_NAME).exists()


def test_model_identity_is_frozen():
    batch = load_golden_batch(
        str(FROZEN_ROOT / "validation_assets" / "golden" / "graph_batch.npz")
    )
    model = build_model(batch)
    assert model.count_params() == candidate.EXPECTED_PARAMETER_COUNT
    assert len(model.trainable_variables) == candidate.EXPECTED_TRAINABLE_VARIABLE_COUNT


def test_synthetic_forward_backward_is_finite():
    tf.keras.utils.set_random_seed(42)
    inputs = tf.constant(
        [[-0.3, 0.2, 0.7, -0.1], [0.4, -0.5, 0.1, 0.9]], tf.float32
    )
    labels = tf.constant([2, 6], tf.int32)
    layer = tf.keras.layers.Dense(7)
    with tf.GradientTape() as tape:
        logits = layer(inputs)
        loss = smoothed_sparse_cross_entropy(labels, logits)
    gradients = tape.gradient(loss, layer.trainable_variables)
    assert bool(tf.math.is_finite(loss).numpy())
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(bool(tf.reduce_all(tf.math.is_finite(g)).numpy()) for g in gradients)


def test_cli_exposes_no_test_path_or_test_limit():
    parser = candidate.build_parser()
    destinations = {action.dest for action in parser._actions}
    assert "test_csv" not in destinations
    assert "test_path" not in destinations
    assert "limit_test_batches" not in destinations
    source = candidate.__file__ and Path(candidate.__file__).read_text(encoding="utf-8")
    assert "GraphBatchGenerator(" not in source
    assert "resolve_final_checkpoint" not in source


@pytest.mark.parametrize("path", ["test", "test.csv", "test_predictions.csv"])
def test_explicit_test_paths_fail_closed(path):
    with pytest.raises(candidate.LS005RescueError, match="must not identify a test"):
        candidate.reject_explicit_test_path(path, "synthetic")


def test_fresh_absolute_cli_help_without_pythonpath(tmp_path):
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
    assert "epsilon=0.05" in completed.stdout


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        (
            {"val_accuracy": 0.65, "val_macro_f1": 0.62, "clean_train_macro_f1": 0.70},
            "LAP_LS005_STRETCH_RESCUE",
        ),
        (
            {
                "val_accuracy": candidate.BASELINE_VAL_ACCURACY,
                "val_macro_f1": candidate.BASELINE_VAL_MACRO_F1,
                "clean_train_macro_f1": candidate.BASELINE_VAL_MACRO_F1 + 0.10,
            },
            "LAP_LS005_RESCUE_PASS",
        ),
        (
            {
                "val_accuracy": candidate.BASELINE_VAL_ACCURACY - 0.01,
                "val_macro_f1": candidate.BASELINE_VAL_MACRO_F1 - 0.01,
                "clean_train_macro_f1": candidate.BASELINE_VAL_MACRO_F1 + 0.04,
            },
            "LAP_LS005_GENERALIZATION_RESCUE_WITH_SMALL_ACCURACY_COST",
        ),
        (
            {
                "val_accuracy": candidate.BASELINE_VAL_ACCURACY - 0.011,
                "val_macro_f1": candidate.BASELINE_VAL_MACRO_F1 - 0.011,
                "clean_train_macro_f1": candidate.BASELINE_VAL_MACRO_F1 + 0.04,
            },
            "LAP_LS005_OVERREGULARIZED",
        ),
        (
            {
                "val_accuracy": candidate.BASELINE_VAL_ACCURACY + 0.001,
                "val_macro_f1": candidate.BASELINE_VAL_MACRO_F1 - 0.001,
                "clean_train_macro_f1": candidate.BASELINE_VAL_MACRO_F1 + 0.20,
            },
            "LAP_LS005_NO_CLEAR_RESCUE",
        ),
        (
            {
                "val_accuracy": candidate.BASELINE_VAL_ACCURACY - 0.01,
                "val_macro_f1": candidate.BASELINE_VAL_MACRO_F1 - 0.01,
                "clean_train_macro_f1": candidate.BASELINE_VAL_MACRO_F1 + 0.20,
            },
            "LAP_LS005_REGRESSION",
        ),
        (
            {"val_accuracy": 0.645, "val_macro_f1": 0.615, "clean_train_macro_f1": 0.80},
            "LAP_LS005_INCONCLUSIVE",
        ),
    ],
)
def test_future_decision_precedence_and_boundaries(metrics, expected):
    assert candidate.classify_outcome(**metrics) == expected


def test_frozen_modules_are_not_reloaded_or_modified_by_candidate_import():
    before = {
        "execution": execution.sparse_cross_entropy,
        "evaluator": evaluator.sparse_cross_entropy,
        "trainer_sha": _sha256(candidate.FROZEN_TRAINER_PATH),
        "wrapper_sha": _sha256(candidate.FROZEN_WRAPPER_PATH),
    }
    importlib.reload(importlib.import_module(
        "research.candidates.tf_lap_gnn_ls005_rescue.loss_adapter"
    ))
    after = {
        "execution": execution.sparse_cross_entropy,
        "evaluator": evaluator.sparse_cross_entropy,
        "trainer_sha": _sha256(candidate.FROZEN_TRAINER_PATH),
        "wrapper_sha": _sha256(candidate.FROZEN_WRAPPER_PATH),
    }
    assert after == before
