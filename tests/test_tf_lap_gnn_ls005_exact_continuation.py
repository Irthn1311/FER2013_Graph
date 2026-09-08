from __future__ import annotations

import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROOT = ROOT / "standalone" / "lap_gnn_tensorflow_ofix7_mid_candidate"
FROZEN_SRC = FROZEN_ROOT / "src"
if str(FROZEN_SRC) not in sys.path:
    sys.path.insert(0, str(FROZEN_SRC))

from lap_gnn_tf.training import trainer  # noqa: E402
from research.candidates.tf_lap_gnn_ls005_runtime_equivalent import (  # noqa: E402
    continuation,
    continuation_equivalence,
    train_validation_only as candidate,
    trainer_continuation_adapter as adapter,
)


ISSUE61_HEAD = "72624bf7bbf556a1d8909eb6f4cc9f00350255a9"
FROZEN_PAYLOAD = "286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e"


@pytest.fixture(scope="module")
def complete_capsules(tmp_path_factory):
    root = tmp_path_factory.mktemp("exact-continuation-capsules")
    result = root / "save.json"
    continuation_equivalence.run_worker("save", root, result)
    return root / "source-capsules"


def _manager(root: Path, *, identity=None, resume=None):
    return continuation.EpochBoundaryContinuationManager(
        root,
        continuation_equivalence.SYNTHETIC_IDENTITY if identity is None else identity,
        resume_root=resume,
    )


def _copied_capsules(source: Path, target: Path) -> Path:
    destination = target / "capsules"
    shutil.copytree(source, destination)
    return destination


def test_state_inventory_is_complete_and_explicit():
    assert continuation.REQUIRED_STATE_INVENTORY == (
        "trainable_model_variables",
        "non_trainable_model_variables",
        "optimizer_variables_and_slots",
        "optimizer_iteration",
        "current_learning_rate",
        "scheduler_state",
        "early_stopping_state",
        "checkpoint_selection_state",
        "completed_epoch_index",
        "training_generator_contract",
        "next_epoch_training_order",
        "python_rng_state",
        "numpy_legacy_rng_state",
        "keras_layer_rng_state_in_model_non_trainable_variables",
        "stateless_augmentation_seed_material",
    )


def test_frozen_trainer_transform_has_exactly_three_reversible_insertions():
    frozen = inspect.getsource(trainer.run_training)
    transformed = adapter.transformed_run_training_source()
    restored = transformed
    for insertion, anchor in (
        (adapter.RESTORE_INSERTION, adapter.RESTORE_ANCHOR),
        (adapter.LOOP_INSERTION, adapter.LOOP_ANCHOR),
        (adapter.PERSIST_INSERTION, adapter.PERSIST_ANCHOR),
    ):
        assert transformed.count(insertion) == 1
        restored = restored.replace(insertion, anchor, 1)
    assert restored == frozen
    assert "persist_epoch_boundary" in transformed
    assert transformed.index("persist_epoch_boundary") > transformed.index(
        "scheduler.step(val_metrics[\"loss\"])"
    )
    assert transformed.index("persist_epoch_boundary") > transformed.index(
        "_atomic_json(output_dir / \"latest_epoch_summary.json\", row)"
    )


def test_temporary_trainer_patch_restores_on_failure(tmp_path):
    original = trainer.run_training
    manager = _manager(tmp_path / "capsules")
    with pytest.raises(KeyboardInterrupt):
        with adapter.continuation_enabled_trainer(manager):
            assert trainer.run_training is not original
            raise KeyboardInterrupt
    assert trainer.run_training is original
    assert not hasattr(trainer, "_lap_ls005_continuation_manager")


def test_capsule_manifest_is_complete_self_contained_and_hashed(complete_capsules):
    capsule_dir, manifest = _manager(complete_capsules)._verify_latest(
        complete_capsules
    )
    assert manifest["status"] == continuation.CAPSULE_STATUS
    assert manifest["completed_epoch"] == 2
    assert manifest["state_inventory"] == list(
        continuation.REQUIRED_STATE_INVENTORY
    )
    assert manifest["issue60_artifacts_used"] is False
    assert manifest["member_count"] == len(manifest["members"])
    assert "runtime_state.index" in manifest["members"]
    assert "explicit_state.npz" in manifest["members"]
    assert "state.json" in manifest["members"]
    for name in continuation.REQUIRED_SELECTED_CHECKPOINT_FILES:
        assert f"selected_checkpoint/{name}" in manifest["members"]
    state = json.loads((capsule_dir / "state.json").read_text(encoding="utf-8"))
    assert state["completed_epoch"] == 2
    assert state["optimizer_iteration"] == 6
    assert state["training_generator_contract"]["next_epoch"] == 3
    assert state["tensorflow_rng_audit"] == {
        "augmentation_uses_epoch_sample_stateless_numpy_generators": True,
        "global_generator_used_by_registered_training": False,
        "keras_layer_seed_generators_captured_as_model_non_trainable_variables": True,
    }


def test_fresh_process_exact_continuation_is_zero_tolerance(tmp_path):
    proof = continuation_equivalence.prove_fresh_process_exact_continuation(
        tmp_path, sys.executable
    )
    assert proof["status"] == "PASS", proof
    assert proof["restore_process_was_distinct"] is True
    assert proof["exact_equality"] is True
    assert proof["floating_tolerance"] == 0.0
    assert proof["per_epoch_exact"] == {3: True, 4: True}
    assert "optimizer_slots_and_iteration" in proof["compared_state"]
    assert "next_epoch_augmentation_sequence" in proof["compared_state"]
    assert proof["issue60_artifacts_used"] is False


def test_missing_required_member_fails_closed(complete_capsules, tmp_path):
    copied = _copied_capsules(complete_capsules, tmp_path)
    (copied / "epoch_000002" / "explicit_state.npz").unlink()
    with pytest.raises(continuation.ExactContinuationError, match="inventory drift"):
        _manager(copied)._verify_latest(copied)


def test_corrupt_required_member_fails_closed(complete_capsules, tmp_path):
    copied = _copied_capsules(complete_capsules, tmp_path)
    state_path = copied / "epoch_000002" / "state.json"
    state_path.write_bytes(state_path.read_bytes() + b"\n")
    with pytest.raises(continuation.ExactContinuationError, match="hash mismatch"):
        _manager(copied)._verify_latest(copied)


def test_missing_state_inventory_fails_closed(complete_capsules, tmp_path):
    copied = _copied_capsules(complete_capsules, tmp_path)
    capsule = copied / "epoch_000002"
    state_path = capsule / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["state_inventory"].remove("optimizer_variables_and_slots")
    state_path.write_text(json.dumps(state), encoding="utf-8")
    manifest_path = capsule / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Even if an attacker updates the individual member hash, the aggregate
    # manifest hash and required inventory still make this fail closed.
    import hashlib

    manifest["members"]["state.json"] = hashlib.sha256(
        state_path.read_bytes()
    ).hexdigest()
    basis = {
        "epoch": manifest["completed_epoch"],
        "scientific_identity": manifest["scientific_identity"],
        "state_inventory": list(continuation.REQUIRED_STATE_INVENTORY),
        "members": manifest["members"],
    }
    manifest["capsule_sha256"] = continuation._canonical_sha256(basis)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    latest_path = copied / "LATEST.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    latest["manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    latest["capsule_sha256"] = manifest["capsule_sha256"]
    latest_path.write_text(json.dumps(latest), encoding="utf-8")
    with pytest.raises(
        continuation.ExactContinuationError, match="state inventory incomplete"
    ):
        _manager(copied)._verify_latest(copied)


def test_incomplete_newer_capsule_never_falls_back(complete_capsules, tmp_path):
    copied = _copied_capsules(complete_capsules, tmp_path)
    (copied / ".epoch_000003.interrupted.tmp").mkdir()
    with pytest.raises(
        continuation.ExactContinuationError, match="Incomplete continuation"
    ):
        _manager(copied)._verify_latest(copied)


def test_latest_pointer_must_reference_most_recent_complete_capsule(
    complete_capsules, tmp_path
):
    copied = _copied_capsules(complete_capsules, tmp_path)
    latest_path = copied / "LATEST.json"
    epoch_one = copied / "epoch_000001"
    manifest_path = epoch_one / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    import hashlib

    latest = {
        "status": continuation.CAPSULE_STATUS,
        "completed_epoch": 1,
        "capsule_directory": epoch_one.name,
        "capsule_sha256": manifest["capsule_sha256"],
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    latest_path.write_text(json.dumps(latest), encoding="utf-8")
    with pytest.raises(continuation.ExactContinuationError, match="most recent"):
        _manager(copied)._verify_latest(copied)


def test_missing_epoch_in_lineage_fails_closed(complete_capsules, tmp_path):
    copied = _copied_capsules(complete_capsules, tmp_path)
    shutil.rmtree(copied / "epoch_000001")
    with pytest.raises(continuation.ExactContinuationError, match="every completed"):
        _manager(copied)._verify_latest(copied)


def test_wrong_scientific_identity_fails_closed(complete_capsules):
    with pytest.raises(
        continuation.ExactContinuationError, match="scientific identity drift"
    ):
        _manager(complete_capsules, identity={"wrong": True})._verify_latest(
            complete_capsules
        )


def test_partial_epoch_cannot_be_persisted(tmp_path):
    manager = _manager(tmp_path / "capsules")
    manager._ensure_new_run_root()
    manager._prepared = True
    with pytest.raises(continuation.ExactContinuationError, match="fully completed"):
        manager.persist_epoch_boundary(
            completed_epoch=2,
            model=None,
            optimizer=None,
            scheduler=None,
            early=None,
            policy=None,
            train_data=None,
            config={},
            history=[{"epoch": 1}],
            output_dir=tmp_path,
        )


def test_epoch_zero_start_rejects_nonempty_capsule_root(tmp_path):
    root = tmp_path / "capsules"
    root.mkdir()
    (root / "unexpected.txt").write_text("not empty", encoding="utf-8")
    with pytest.raises(continuation.ExactContinuationError, match="absent or empty"):
        _manager(root)._ensure_new_run_root()


@pytest.mark.parametrize(
    "path",
    [
        "/kaggle/input/issue60/checkpoints",
        "/kaggle/input/issue-60-partial/capsules",
    ],
)
def test_issue60_resume_paths_are_permanently_rejected(path):
    with pytest.raises(candidate.RuntimeEquivalentError, match="permanently forbidden"):
        candidate.reject_issue60_artifact_path(path)


def test_scientific_identity_preserves_issue61_locks():
    identity = candidate.continuation_scientific_identity()
    assert identity["issue61_parent"] == candidate.IMPLEMENTATION_BASE
    assert identity["scientific_payload_sha256"] == FROZEN_PAYLOAD
    assert identity["label_smoothing"] == 0.05
    assert identity["seed"] == 42
    assert identity["checkpoint_policy"] == (
        "earliest_strict_max_validation_accuracy"
    )
    assert identity["runtime_resources"] is None
    assert identity["test_access"] is False
    assert identity["issue60_artifacts_used"] is False


def test_scientific_files_and_config_are_unchanged_from_issue61():
    assert subprocess.run(
        ["git", "cat-file", "-e", f"{ISSUE61_HEAD}^{{commit}}"],
        cwd=ROOT,
        check=False,
    ).returncode == 0
    unchanged_paths = [
        FROZEN_ROOT.relative_to(ROOT).as_posix(),
        "research/candidates/tf_lap_gnn_ls005_rescue",
        "research/candidates/tf_lap_gnn_ls005_runtime_equivalent/configs",
    ]
    changed = subprocess.run(
        ["git", "diff", "--name-only", ISSUE61_HEAD, "--", *unchanged_paths],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    assert changed == ""


def test_cli_requires_capsule_output_and_exposes_no_issue60_resume_option():
    parser = candidate.build_parser()
    destinations = {action.dest for action in parser._actions}
    assert "continuation_root" in destinations
    assert "resume_capsule_root" in destinations
    assert "issue60_checkpoint" not in destinations
    assert "test_csv" not in destinations


def test_no_pytorch_or_issue60_checkpoint_loading_in_continuation_sources():
    sources = [
        Path(continuation.__file__),
        Path(continuation_equivalence.__file__),
        Path(adapter.__file__),
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    assert "import torch" not in text
    assert "from torch" not in text
    assert "issue60-run-output" not in text
    assert "best_val_accuracy.keras" in text
