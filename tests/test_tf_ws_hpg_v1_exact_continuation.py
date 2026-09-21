"""Issue #71 exact continuation and planned-segmentation regressions."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import tensorflow as tf

from research.candidates.tf_ws_hpg_v1_continuation import (
    IMPLEMENTATION_STATUS,
    PLANNED_PAUSE_STATUS,
)
from research.candidates.tf_ws_hpg_v1_continuation import continuation
from research.candidates.tf_ws_hpg_v1_continuation import continuation_equivalence
from research.candidates.tf_ws_hpg_v1_continuation import capsule_benchmark
from research.candidates.tf_ws_hpg_v1_continuation import data_order
from research.candidates.tf_ws_hpg_v1_continuation import train_validation_only as runtime
from research.candidates.tf_ws_hpg_v1_training import augmentation, data
from research.candidates.tf_ws_hpg_v1_training import train_validation_only as accepted
from research.candidates.tf_ws_hpg_v1_weak_support.model import build_ws_hpg_v1_weak_support


ROOT = Path(__file__).resolve().parents[1]
EXACT_PARENT = "ab7c7a49e923764b6192d9e774352fddf8f1df8b"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def complete_capsules(tmp_path_factory):
    root = tmp_path_factory.mktemp("ws-complete-capsules")
    continuation_equivalence.run_worker("plan", root, root / "plan.json")
    continuation_equivalence.run_worker("save", root, root / "save.json")
    return root / "source-capsules"


def _plan():
    return data_order.AcceptedShufflePlan.materialize(8, 5)


def _manager(root: Path, *, identity=None):
    with np.load(root / continuation.IMMUTABLE_PLAN_NAME, allow_pickle=False) as arrays:
        plan = data_order.AcceptedShufflePlan.from_orders(arrays["orders"])
    identity = identity or dict(
        continuation_equivalence.SYNTHETIC_IDENTITY_BASE,
        accepted_shuffle_plan_sha256=plan.sha256,
    )
    return continuation.EpochBoundaryContinuationManager(root, identity, plan)


def _copy_capsules(source: Path, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    target = root / "capsules"
    shutil.copytree(source, target)
    return target


def _rewrite_latest_hash(root: Path, capsule: Path) -> None:
    latest_path = root / "LATEST.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    manifest = json.loads((capsule / "manifest.json").read_text(encoding="utf-8"))
    latest.update(
        completed_epoch=manifest["completed_epoch"],
        capsule_directory=capsule.name,
        capsule_sha256=manifest["capsule_sha256"],
        manifest_sha256=sha(capsule / "manifest.json"),
    )
    latest_path.write_text(json.dumps(latest), encoding="utf-8")


def _resign_manifest(root: Path, capsule: Path) -> None:
    manifest_path = capsule / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    basis = {
        "completed_epoch": manifest["completed_epoch"],
        "parent_capsule_sha256": manifest["parent_capsule_sha256"],
        "scientific_identity_sha256": manifest["scientific_identity_sha256"],
        "immutable_shuffle_plan_sha256": manifest["immutable_shuffle_plan_sha256"],
        "state_inventory": list(continuation.REQUIRED_STATE_INVENTORY),
        "members": manifest["members"],
    }
    manifest["capsule_sha256"] = continuation.canonical_sha256(basis)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _rewrite_latest_hash(root, capsule)


def test_locked_architecture_sources_identity_and_training_contract():
    assert runtime.EXACT_PARENT == EXACT_PARENT
    assert runtime.verify_locked_sources() == {
        "model.py": runtime.EXPECTED_MODEL_SHA256,
        "support.py": runtime.EXPECTED_SUPPORT_SHA256,
        "accepted_training": runtime.EXPECTED_TRAINING_SOURCE_SHA256,
    }
    model = build_ws_hpg_v1_weak_support()
    assert runtime.validate_model_identity(model) == {
        "parameters": 707_213,
        "trainable_variables": 118,
        "keras_variables": 138,
    }
    assert len(model.non_trainable_variables) == 20
    assert all("seed_generator" in getattr(variable, "path", variable.name)
               for variable in model.non_trainable_variables)
    assert accepted.TRAINING_CONFIG["max_epochs"] == 100
    assert accepted.TRAINING_CONFIG["training_label_smoothing"] == 0.05
    assert accepted.TRAINING_CONFIG["checkpoint"] == "earliest_strict_max_val_accuracy"
    identity = runtime.scientific_identity(_plan(), 8)
    assert set(identity["continuation_runtime_source_sha256"]) == {
        "__init__.py", "continuation.py", "data_order.py", "train_validation_only.py"
    }


def test_materialized_orders_equal_the_accepted_tf_data_stream_exactly():
    count, epochs = 12, 4
    images = np.arange(count, dtype=np.float32)[:, None]
    supports = np.arange(count, dtype=np.float32)[:, None]
    labels = np.arange(count, dtype=np.int32)
    accepted_records = data._training_records(images, supports, labels)
    accepted_orders = []
    for _ in range(epochs):
        accepted_orders.append(
            [int(record[0].numpy()) for _, record in accepted_records]
        )
    raw = data_order.materialize_raw_shuffle_iterations(count, epochs)
    np.testing.assert_array_equal(raw, np.asarray(accepted_orders))


def test_plan_locks_accepted_keras_315_odd_shuffle_iterator_epochs():
    count, epochs = 12, 4
    raw = data_order.materialize_raw_shuffle_iterations(count, 2 * epochs)
    plan = data_order.AcceptedShufflePlan.materialize(count, epochs)
    np.testing.assert_array_equal(plan.orders, raw[0::2])


def test_replayed_epoch_preserves_original_ids_enumeration_and_parameters():
    count = 8
    plan = data_order.AcceptedShufflePlan.materialize(count, 3)
    stream = data_order.next_epoch_stream(plan, 1)
    np.testing.assert_array_equal(stream["original_sample_order"], plan.order(2))
    np.testing.assert_array_equal(stream["enumeration_indices"], np.arange(count))
    expected = []
    for index in range(count):
        values = augmentation.sample_parameters(tf.constant(index))
        expected.append([
            float(tf.cast(values[key], tf.float32)) for key in (
                "flip", "rotation_degrees", "translation_x", "translation_y",
                "contrast", "brightness", "erase", "erase_area_fraction", "erase_aspect",
            )
        ])
    np.testing.assert_array_equal(stream["augmentation_parameters"], np.asarray(expected, np.float32))


def test_capsule_contract_and_complete_state_inventory(complete_capsules):
    capsule, manifest = _manager(complete_capsules).verify_latest()
    assert manifest["status"] == continuation.CAPSULE_STATUS
    assert manifest["completed_epoch"] == 2
    assert manifest["capsule_contract_sha256"] == continuation.CAPSULE_CONTRACT_SHA256
    assert manifest["state_inventory"] == list(continuation.REQUIRED_STATE_INVENTORY)
    assert manifest["invalid_issue70_state_used"] is False
    assert manifest["member_count"] == len(manifest["members"])
    assert (complete_capsules / continuation.IMMUTABLE_PLAN_NAME).is_file()
    assert (complete_capsules / continuation.IMMUTABLE_PLAN_MANIFEST_NAME).is_file()
    assert (complete_capsules / continuation.IMMUTABLE_AUGMENTATION_NAME).is_file()
    assert {"runtime_state.index", "explicit_state.npz", "state.json"} <= set(manifest["members"])
    for name in continuation.REQUIRED_SELECTED_CHECKPOINT_FILES:
        assert f"selected_checkpoint/{name}" in manifest["members"]
    state = json.loads((capsule / "state.json").read_text(encoding="utf-8"))
    assert state["model_variable_counts"] == {"trainable": 118, "non_trainable": 20, "keras_total": 138}
    assert state["optimizer_iteration"] == 4
    assert state["next_epoch_stream"]["next_epoch"] == 3
    assert state["warmup_cosine_state"]["position"] == 4
    assert state["invalid_issue70_state_used"] is False
    assert state["test_access"] is False
    with np.load(capsule / "explicit_state.npz", allow_pickle=False) as arrays:
        assert "accepted_shuffle_plan_orders" not in arrays
        assert "next_epoch_augmentation_parameters" not in arrays


def test_fresh_process_four_epoch_equivalence_is_exact(tmp_path):
    proof = continuation_equivalence.prove_fresh_process_exact_continuation(
        tmp_path, sys.executable
    )
    assert proof["status"] == "PASS", proof
    assert proof["worker_pids_distinct"] is True
    assert proof["exact_equality"] is True
    assert proof["floating_tolerance"] == 0.0
    assert proof["tensorflow_op_determinism_enabled"] is False
    assert proof["per_epoch_exact"] == {3: True, 4: True}
    assert proof["aggregate_state_sha_equal"] == {3: True, 4: True}
    assert all(
        values["uninterrupted"] == values["restored"]
        for values in proof["aggregate_state_sha256"].values()
    )
    assert proof["variable_counts"] == {"trainable": 118, "non_trainable": 20, "keras": 138}
    assert proof["invalid_issue70_state_used"] is False


def test_accepted_issue70_lifecycle_matches_new_non_resume_exactly(tmp_path):
    proof = continuation_equivalence.prove_accepted_lifecycle_equivalence(
        tmp_path, sys.executable
    )
    assert proof["status"] == "PASS", proof
    assert proof["exact_equality"] is True
    assert proof["floating_tolerance"] == 0.0
    assert proof["optimizer_batches_per_epoch"] == 2
    assert proof["full_model"] is True
    assert proof["path_a_initialization"] == {
        "new_build_runtime_called": False,
        "optimizer_eager_build_called": False,
        "tensorflow_global_generator_explicitly_set": False,
    }
    assert proof["path_b_initialization"] == {
        "new_build_runtime_called": True,
        "optimizer_eager_build_called": True,
        "tensorflow_global_generator_explicitly_set": True,
    }
    assert proof["tensorflow_global_generator_audit"] == {
        "absent_before_accepted_initialization": True,
        "absent_after_four_accepted_epochs": True,
        "scientifically_consumed": False,
    }
    assert proof["tensorflow_global_generator_excluded_from_accepted_vs_new_aggregate"] is True
    assert proof["keras_dropout_seed_generators_included_in_aggregate"] is True
    assert proof["accepted_plan_worker"]["accepted_training_records_used"] is True
    assert proof["accepted_plan_worker"]["raw_successive_traversals"] == 10
    assert proof["accepted_plan_worker"]["keras_one_based_odd_traversals_selected"] == [
        1, 3, 5, 7, 9
    ]
    assert proof["per_epoch_exact"] == {1: True, 2: True, 3: True, 4: True}
    assert all(
        values["accepted"] == values["new_non_resume"]
        for values in proof["aggregate_state_sha256"].values()
    )


def test_proof1_path_a_is_exact_accepted_initialization_and_path_b_is_production():
    accepted_source = inspect.getsource(
        continuation_equivalence._run_accepted_lifecycle
    )
    assert "_build_runtime(" not in accepted_source
    assert "optimizer.build(" not in accepted_source
    assert "set_global_generator(" not in accepted_source
    assert accepted_source.index("random.seed(42)") < accepted_source.index(
        "build_ws_hpg_v1_weak_support()"
    )
    assert accepted_source.index("np.random.seed(42)") < accepted_source.index(
        "build_ws_hpg_v1_weak_support()"
    )
    assert accepted_source.index("tf.keras.utils.set_random_seed(42)") < accepted_source.index(
        "build_ws_hpg_v1_weak_support()"
    )
    assert "optimizer=accepted.build_optimizer(2)" in accepted_source
    assert "loss=accepted.training_loss" in accepted_source
    assert "accepted.EarliestStrictMaximumCheckpoint" in accepted_source
    assert "accepted_data.build_dataset" in accepted_source
    assert accepted_source.count("model.fit(") == 1
    production_source = inspect.getsource(continuation_equivalence._run_epochs)
    assert "_build_runtime" in production_source
    assert "EpochBoundaryContinuationManager" in production_source
    assert "_EpochBoundaryCapsuleCallback" in production_source
    assert "build_segment_training_dataset" in production_source


def test_missing_and_corrupt_immutable_plan_fail_closed(complete_capsules, tmp_path):
    reference = _manager(complete_capsules)
    missing = _copy_capsules(complete_capsules, tmp_path / "missing-plan")
    (missing / continuation.IMMUTABLE_PLAN_NAME).unlink()
    with pytest.raises(continuation.ExactContinuationError):
        continuation.EpochBoundaryContinuationManager(
            missing, reference.scientific_identity, reference.order_plan
        ).verify_latest()
    corrupt = _copy_capsules(complete_capsules, tmp_path / "corrupt-plan")
    plan_path = corrupt / continuation.IMMUTABLE_PLAN_NAME
    plan_path.write_bytes(plan_path.read_bytes() + b"corrupt")
    with pytest.raises(continuation.ExactContinuationError, match="plan member hash"):
        _manager(corrupt).verify_latest()


def test_missing_and_corrupt_immutable_augmentation_fail_closed(complete_capsules, tmp_path):
    reference = _manager(complete_capsules)
    missing = _copy_capsules(complete_capsules, tmp_path / "missing-augmentation")
    (missing / continuation.IMMUTABLE_AUGMENTATION_NAME).unlink()
    with pytest.raises(continuation.ExactContinuationError, match="augmentation member missing"):
        continuation.EpochBoundaryContinuationManager(
            missing, reference.scientific_identity, reference.order_plan
        ).verify_latest()
    corrupt = _copy_capsules(complete_capsules, tmp_path / "corrupt-augmentation")
    path = corrupt / continuation.IMMUTABLE_AUGMENTATION_NAME
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(continuation.ExactContinuationError, match="augmentation member hash"):
        _manager(corrupt).verify_latest()


def test_production_dimension_benchmark_contract_is_locked():
    assert capsule_benchmark.PRODUCTION_SAMPLE_COUNT == 28_709
    assert capsule_benchmark.PRODUCTION_PLAN_EPOCHS == 101
    assert capsule_benchmark.MEASURED_EPOCHS == (1, 30, 60)
    source = inspect.getsource(capsule_benchmark.run_benchmark)
    assert "optimizer_training_steps_executed" in source
    assert '"training_executed": False' in source
    assert '"test_access": False' in source


def test_missing_and_corrupt_members_fail_closed(complete_capsules, tmp_path):
    missing = _copy_capsules(complete_capsules, tmp_path / "missing")
    missing_manager = _manager(missing)
    (missing / "epoch_000002" / "explicit_state.npz").unlink()
    with pytest.raises(continuation.ExactContinuationError, match="inventory drift"):
        missing_manager.verify_latest()
    corrupt = _copy_capsules(complete_capsules, tmp_path / "corrupt")
    state = corrupt / "epoch_000002" / "state.json"
    state.write_bytes(state.read_bytes() + b"\n")
    with pytest.raises(continuation.ExactContinuationError, match="hash mismatch"):
        _manager(corrupt).verify_latest()


def test_incomplete_newer_staging_never_falls_back(complete_capsules, tmp_path):
    copied = _copy_capsules(complete_capsules, tmp_path)
    (copied / ".epoch_000003.interrupted.tmp").mkdir()
    with pytest.raises(continuation.ExactContinuationError, match="Incomplete newer"):
        _manager(copied).verify_latest()


def test_missing_required_state_inventory_fails_closed(complete_capsules, tmp_path):
    copied = _copy_capsules(complete_capsules, tmp_path)
    capsule = copied / "epoch_000002"
    state_path = capsule / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["state_inventory"].remove("adamw_variables_and_slots")
    state_path.write_text(json.dumps(state), encoding="utf-8")
    manifest_path = capsule / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["members"]["state.json"] = sha(state_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _resign_manifest(copied, capsule)
    with pytest.raises(continuation.ExactContinuationError, match="state inventory"):
        _manager(copied).verify_latest()


def test_stale_latest_and_lineage_gap_fail_closed(complete_capsules, tmp_path):
    stale = _copy_capsules(complete_capsules, tmp_path / "stale")
    _rewrite_latest_hash(stale, stale / "epoch_000001")
    with pytest.raises(continuation.ExactContinuationError, match="stale"):
        _manager(stale).verify_latest()
    gap = _copy_capsules(complete_capsules, tmp_path / "gap")
    shutil.rmtree(gap / "epoch_000001")
    with pytest.raises(continuation.ExactContinuationError, match="lineage gap"):
        _manager(gap).verify_latest()


def test_parent_capsule_identity_mismatch_fails_closed(complete_capsules, tmp_path):
    copied = _copy_capsules(complete_capsules, tmp_path)
    capsule = copied / "epoch_000002"
    manifest_path = capsule / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["parent_capsule_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _resign_manifest(copied, capsule)
    with pytest.raises(continuation.ExactContinuationError, match="parent lineage"):
        _manager(copied).verify_latest()


def test_source_or_config_identity_drift_fails_closed(complete_capsules):
    with pytest.raises(continuation.ExactContinuationError, match="identity drift"):
        _manager(complete_capsules, identity={"drift": True}).verify_latest()


def test_partial_epoch_cannot_be_persisted(tmp_path):
    plan = _plan()
    identity = dict(
        continuation_equivalence.SYNTHETIC_IDENTITY_BASE,
        accepted_shuffle_plan_sha256=plan.sha256,
    )
    manager = continuation.EpochBoundaryContinuationManager(
        tmp_path / "capsules", identity, plan
    )
    manager._ensure_new_root()
    manager._prepared = True
    with pytest.raises(continuation.ExactContinuationError, match="fully completed"):
        manager.persist_epoch_boundary(
            completed_epoch=2,
            model=None,
            optimizer=None,
            early_stop=None,
            checkpoint_callback=None,
            history=[{"epoch": 1}],
            output_root=tmp_path,
        )


def test_planned_pause_requires_verified_epoch60_and_has_no_scientific_result(monkeypatch):
    capsule = Path("/capsules/epoch_000060")
    manager = type("Manager", (), {"verify_latest": lambda self: (
        capsule, {"completed_epoch": 60, "status": continuation.CAPSULE_STATUS}
    )})()
    result = runtime._planned_pause_result(manager, 60)
    assert result["status"] == PLANNED_PAUSE_STATUS
    assert result["scientific_result_valid"] is False
    assert result["scientific_interpretation"] is None
    assert result["final_selected_checkpoint_evaluation_performed"] is False
    assert "decision" not in result
    with pytest.raises(runtime.WSContinuationError, match="only for epoch 60"):
        runtime._planned_pause_result(manager, 59)


def test_epoch_bookkeeping_and_complete_capsule_precede_planned_pause():
    source = inspect.getsource(runtime._EpochBoundaryCapsuleCallback.on_epoch_end)
    assert source.index("persist_epoch_boundary") < source.index("PLANNED_PAUSE_EPOCH")
    assert source.index("not self.model.stop_training") < source.index("PLANNED_PAUSE_EPOCH")
    lifecycle_source = inspect.getsource(runtime.run_continuable_lifecycle)
    assert lifecycle_source.count("model.fit(") == 1
    assert "build_segment_training_dataset" in lifecycle_source
    assert lifecycle_source.index("[checkpoint, early_stop]") < lifecycle_source.index(
        "callbacks.append(capsule_callback)"
    )
    capsule_source = inspect.getsource(continuation.EpochBoundaryContinuationManager.persist_epoch_boundary)
    assert capsule_source.index("os.replace(staging, target)") < capsule_source.index("LATEST.json")


def test_terminal_evaluation_inventory_is_exactly_accepted(monkeypatch, tmp_path):
    calls = []
    class FakeSelected:
        variables = []
    selected = FakeSelected()
    monkeypatch.setattr(tf.keras.models, "load_model", lambda path, compile: selected)
    metrics = {
        ("clean", "normal"): {"accuracy": .8, "macro_f1": .7, "loss": .5, "sample_count": 2},
        ("validation", "normal"): {"accuracy": .6, "macro_f1": .55, "loss": .8, "sample_count": 2},
        ("validation", "ones"): {"accuracy": .5, "macro_f1": .45, "loss": 1.0, "sample_count": 2},
    }
    def evaluate(model, dataset, *, support_override):
        calls.append((model, dataset, support_override))
        return metrics[(dataset, support_override)]
    monkeypatch.setattr(accepted, "evaluate", evaluate)
    monkeypatch.setattr(accepted, "classify_outcome", lambda **kwargs: "LOCKED_DECISION")
    checkpoint = type("Checkpoint", (), {"selected_epoch": 4})()
    result = runtime._terminal_evaluation(
        tmp_path, "clean", "validation", {}, [{"epoch": 1}], checkpoint
    )
    assert [(item[1], item[2]) for item in calls] == [
        ("clean", "normal"), ("validation", "normal"), ("validation", "ones")
    ]
    assert all(item[0] is selected for item in calls)
    assert result["terminal_evaluation_inventory"] == [
        "clean_train_normal_support", "validation_normal_support", "validation_all_ones_support"
    ]


@pytest.mark.parametrize("value", ["/data/test/train.csv", "/data/testing/priors", "/data/test_split/capsules", "/data/test-split/output"])
def test_all_cli_path_classes_preserve_lexical_test_isolation(value):
    with pytest.raises(data.WSHPGDataError):
        data.reject_test_path(value)


def test_fresh_cli_help_from_outside_repo_without_pythonpath(tmp_path):
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    script = Path(runtime.__file__).resolve()
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env=environment,
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--continuation-root" in completed.stdout


def test_no_pytorch_or_invalid_issue70_restore_and_status_is_implementation_only():
    package = ROOT / "research/candidates/tf_ws_hpg_v1_continuation"
    source = "\n".join(path.read_text(encoding="utf-8") for path in package.glob("*.py"))
    assert "import torch" not in source and "from torch" not in source
    assert "epoch79" not in source and "epoch_000079" not in source
    assert IMPLEMENTATION_STATUS == "WS_HPG_V1_EXACT_CONTINUATION_PREPARATION_ONLY"
    assert "--test" not in inspect.getsource(runtime.build_parser)


def test_accepted_ws_cf_ra_lap_sources_are_untouched_from_exact_parent():
    paths = [
        "research/candidates/tf_ws_hpg_v1_weak_support",
        "research/candidates/tf_ws_hpg_v1_training",
        "research/candidates/tf_cf_hpg",
        "research/candidates/tf_cf_hpg_v1_1_resolution",
        "research/candidates/tf_cf_hpg_v1_2_tokenizer",
        "research/candidates/tf_cf_hpg_v1_3_multiscale_readout",
        "research/candidates/tf_ra_hpg_v1_relation_aware",
        "research/candidates/tf_lap_gnn_ls005_runtime_equivalent",
        "standalone/lap_gnn_tensorflow_ofix7_mid_candidate",
    ]
    changed = subprocess.run(
        ["git", "diff", "--name-only", EXACT_PARENT, "--", *paths],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    assert changed == ""
