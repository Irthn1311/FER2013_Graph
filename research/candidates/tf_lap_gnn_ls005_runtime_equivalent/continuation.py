"""Fail-closed exact epoch-boundary continuation capsules.

The capsule is intentionally independent of Keras model serialization.  A
TensorFlow checkpoint restores trackable runtime state while an explicit NPZ
inventory proves every model and optimizer variable byte-for-byte.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import shutil
from typing import Any
import uuid

import numpy as np
import tensorflow as tf


CAPSULE_SCHEMA_VERSION = 1
CAPSULE_STATUS = "COMPLETE_EPOCH_BOUNDARY"
NOT_PROVEN_STATUS = "LAP_LS005_EXACT_RESUME_NOT_PROVEN"
REQUIRED_STATE_INVENTORY = (
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
REQUIRED_SELECTED_CHECKPOINT_FILES = (
    "best_val_accuracy.keras",
    "best_val_accuracy.weights.h5",
    "best_val_accuracy.metadata.json",
)
CAPSULE_CONTRACT_DEFINITION = {
    "schema_version": CAPSULE_SCHEMA_VERSION,
    "complete_status": CAPSULE_STATUS,
    "required_state_inventory": list(REQUIRED_STATE_INVENTORY),
    "required_selected_checkpoint_files": list(REQUIRED_SELECTED_CHECKPOINT_FILES),
    "member_integrity": "sha256",
    "aggregate_integrity": "canonical_json_sha256",
    "partial_epoch_resume": "forbidden",
    "older_capsule_fallback_on_incomplete_newer": "forbidden",
}
CAPSULE_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        CAPSULE_CONTRACT_DEFINITION,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
).hexdigest()


class ExactContinuationError(RuntimeError):
    """Fail-closed capsule creation or restore error."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExactContinuationError(f"Unreadable {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ExactContinuationError(f"{label} must be a JSON object: {path}")
    return value


def _variable_array(variable: Any) -> np.ndarray:
    return np.asarray(variable.numpy())


def _variable_inventory(variables: list[Any], prefix: str) -> tuple[dict, dict]:
    arrays: dict[str, np.ndarray] = {}
    entries: list[dict[str, Any]] = []
    for index, variable in enumerate(variables):
        key = f"{prefix}_{index:04d}"
        array = _variable_array(variable)
        arrays[key] = array
        entries.append(
            {
                "index": index,
                "key": key,
                "name": str(
                    getattr(variable, "path", getattr(variable, "name", key))
                ),
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            }
        )
    return arrays, {"count": len(entries), "variables": entries}


def _assert_variable_inventory(
    variables: list[Any], inventory: dict[str, Any], arrays: Any, label: str
) -> None:
    entries = inventory.get("variables")
    if not isinstance(entries, list) or inventory.get("count") != len(variables):
        raise ExactContinuationError(f"{label} variable count drift")
    for index, (variable, entry) in enumerate(zip(variables, entries)):
        if entry.get("index") != index or entry.get("key") not in arrays:
            raise ExactContinuationError(f"{label} variable inventory malformed")
        actual = _variable_array(variable)
        expected = np.asarray(arrays[entry["key"]])
        if (
            str(actual.dtype) != entry.get("dtype")
            or list(actual.shape) != entry.get("shape")
            or not np.array_equal(actual, expected)
        ):
            raise ExactContinuationError(
                f"{label} variable {index} differs after exact restore"
            )
        if hashlib.sha256(actual.tobytes(order="C")).hexdigest() != entry.get(
            "sha256"
        ):
            raise ExactContinuationError(f"{label} variable hash mismatch")


def _assign_variable_inventory(
    variables: list[Any], inventory: dict[str, Any], arrays: Any, label: str
) -> None:
    entries = inventory.get("variables")
    if not isinstance(entries, list) or inventory.get("count") != len(variables):
        raise ExactContinuationError(f"{label} variable count drift")
    for index, (variable, entry) in enumerate(zip(variables, entries)):
        key = entry.get("key")
        if entry.get("index") != index or key not in arrays:
            raise ExactContinuationError(f"{label} variable inventory malformed")
        expected = np.asarray(arrays[key])
        actual = _variable_array(variable)
        if str(expected.dtype) != entry.get("dtype") or expected.shape != actual.shape:
            raise ExactContinuationError(f"{label} variable shape/dtype drift")
        variable.assign(expected)


def _jsonable_tuple(value: Any) -> Any:
    if isinstance(value, tuple):
        return {"__tuple__": [_jsonable_tuple(item) for item in value]}
    if isinstance(value, list):
        return [_jsonable_tuple(item) for item in value]
    return value


def _restore_tuple(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"__tuple__"}:
        return tuple(_restore_tuple(item) for item in value["__tuple__"])
    if isinstance(value, list):
        return [_restore_tuple(item) for item in value]
    return value


def _numpy_rng_state() -> tuple[dict[str, Any], np.ndarray]:
    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return (
        {
            "algorithm": algorithm,
            "position": int(position),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached_gaussian),
        },
        np.asarray(keys, dtype=np.uint32),
    )


def _augmentation_seed_material(
    train_data: Any, config: dict[str, Any], next_epoch: int
) -> tuple[np.ndarray, str]:
    order = np.asarray(train_data._order(next_epoch), dtype=np.int64)
    corruption = config["graph"].get("prior_corruption") or {}
    seed = int(corruption.get("seed", 137) or 137)
    material = (
        seed + int(next_epoch) * 1_000_003 + order.astype(np.uint64) * 97_531
    ) % (2**32 - 1)
    material = np.asarray(material, dtype=np.uint64)
    return material, hashlib.sha256(material.tobytes(order="C")).hexdigest()


def _generator_contract(
    train_data: Any, config: dict[str, Any], completed_epoch: int
) -> dict[str, Any]:
    next_epoch = int(completed_epoch) + 1
    order = np.asarray(train_data._order(next_epoch), dtype=np.int64)
    material, material_sha = _augmentation_seed_material(
        train_data, config, next_epoch
    )
    return {
        "split": str(train_data.split),
        "sample_count": len(train_data.dataset),
        "batch_size": int(train_data.batch_size),
        "seed": int(train_data.seed),
        "shuffle": bool(train_data.shuffle),
        "cache_size": int(train_data.cache_size),
        "graph_workers": int(train_data.graph_workers),
        "dataset_epoch": int(train_data.dataset.epoch),
        "next_epoch": next_epoch,
        "next_order_count": len(order),
        "next_order_sha256": hashlib.sha256(
            order.tobytes(order="C")
        ).hexdigest(),
        "augmentation_rng_semantics": (
            "stateless default_rng(seed + epoch*1000003 + sample_index*97531)"
        ),
        "augmentation_seed_material_count": len(material),
        "augmentation_seed_material_sha256": material_sha,
        "prior_corruption": config["graph"].get("prior_corruption"),
    }


def _checkpoint_selection_state(policy: Any) -> dict[str, Any]:
    return {
        "best_macro": float(policy.best_macro),
        "best_accuracy": float(policy.best_accuracy),
        "best_macro_epoch": int(policy.best_macro_epoch),
        "best_accuracy_epoch": int(policy.best_accuracy_epoch),
        "selection_policy": "earliest_strict_max_validation_accuracy",
    }


class EpochBoundaryContinuationManager:
    """Create and restore immutable, self-contained epoch capsules."""

    def __init__(
        self,
        write_root: str | Path,
        scientific_identity: dict[str, Any],
        *,
        resume_root: str | Path | None = None,
    ) -> None:
        self.write_root = Path(write_root).expanduser().resolve()
        self.resume_root = (
            None
            if resume_root is None
            else Path(resume_root).expanduser().resolve()
        )
        self.scientific_identity = json.loads(json.dumps(scientific_identity))
        self._prepared = False

    def _ensure_new_run_root(self) -> None:
        if self.write_root.exists() and any(self.write_root.iterdir()):
            raise ExactContinuationError(
                "Epoch-0 continuation root must be absent or empty"
            )
        self.write_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _reject_incomplete_directories(root: Path) -> None:
        incomplete = sorted(
            path.name
            for path in root.iterdir()
            if path.is_dir() and (path.name.startswith(".epoch_") or ".tmp" in path.name)
        )
        if incomplete:
            raise ExactContinuationError(
                f"Incomplete continuation capsule directories exist: {incomplete}"
            )

    def _verify_capsule(self, capsule_dir: Path) -> dict[str, Any]:
        manifest_path = capsule_dir / "manifest.json"
        manifest = _json_object(manifest_path, "continuation manifest")
        if manifest.get("schema_version") != CAPSULE_SCHEMA_VERSION:
            raise ExactContinuationError("Continuation schema drift")
        if manifest.get("status") != CAPSULE_STATUS:
            raise ExactContinuationError("Continuation capsule is not complete")
        if manifest.get("capsule_contract_sha256") != CAPSULE_CONTRACT_SHA256:
            raise ExactContinuationError("Continuation contract drift")
        if manifest.get("scientific_identity") != self.scientific_identity:
            raise ExactContinuationError("Continuation scientific identity drift")
        if manifest.get("issue60_artifacts_used") is not False:
            raise ExactContinuationError("Issue #60 artifact exclusion is not proven")
        members = manifest.get("members")
        if not isinstance(members, dict) or not members:
            raise ExactContinuationError("Continuation member inventory missing")
        actual_members = {
            path.relative_to(capsule_dir).as_posix()
            for path in capsule_dir.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        }
        if actual_members != set(members):
            raise ExactContinuationError("Continuation member inventory drift")
        for relative, expected_sha in members.items():
            if _sha256(capsule_dir / relative) != expected_sha:
                raise ExactContinuationError(
                    f"Continuation member hash mismatch: {relative}"
                )
        state = _json_object(capsule_dir / "state.json", "continuation state")
        if state.get("state_inventory") != list(REQUIRED_STATE_INVENTORY):
            raise ExactContinuationError("Continuation state inventory incomplete")
        if state.get("issue60_artifacts_used") is not False:
            raise ExactContinuationError("Issue #60 artifact exclusion is not proven")
        capsule_basis = {
            "epoch": manifest.get("completed_epoch"),
            "scientific_identity": self.scientific_identity,
            "state_inventory": list(REQUIRED_STATE_INVENTORY),
            "members": members,
        }
        if _canonical_sha256(capsule_basis) != manifest.get("capsule_sha256"):
            raise ExactContinuationError("Continuation aggregate hash mismatch")
        return manifest

    def _verify_latest(self, root: Path) -> tuple[Path, dict[str, Any]]:
        if not root.is_dir():
            raise ExactContinuationError(f"Continuation root is absent: {root}")
        self._reject_incomplete_directories(root)
        latest_path = root / "LATEST.json"
        latest = _json_object(latest_path, "continuation latest pointer")
        if latest.get("status") != CAPSULE_STATUS:
            raise ExactContinuationError("Latest continuation pointer is incomplete")
        capsule_name = latest.get("capsule_directory")
        if not isinstance(capsule_name, str) or Path(capsule_name).name != capsule_name:
            raise ExactContinuationError("Latest continuation path is invalid")
        capsule_dir = root / capsule_name
        manifest = self._verify_capsule(capsule_dir)
        if _sha256(capsule_dir / "manifest.json") != latest.get("manifest_sha256"):
            raise ExactContinuationError("Latest manifest hash mismatch")
        if latest.get("capsule_sha256") != manifest.get("capsule_sha256"):
            raise ExactContinuationError("Latest aggregate hash mismatch")
        epoch = int(manifest["completed_epoch"])
        if latest.get("completed_epoch") != epoch:
            raise ExactContinuationError("Latest completed epoch mismatch")
        epoch_dirs = sorted(
            path for path in root.glob("epoch_*") if path.is_dir()
        )
        if not epoch_dirs or capsule_dir != epoch_dirs[-1]:
            raise ExactContinuationError(
                "Latest pointer is not the most recent completed capsule"
            )
        try:
            lineage_epochs = [int(path.name.removeprefix("epoch_")) for path in epoch_dirs]
        except ValueError as exc:
            raise ExactContinuationError("Malformed continuation lineage directory") from exc
        if lineage_epochs != list(range(1, epoch + 1)):
            raise ExactContinuationError(
                "Continuation lineage must contain every completed epoch"
            )
        return capsule_dir, manifest

    def _copy_resume_lineage(self) -> tuple[Path, dict[str, Any]]:
        assert self.resume_root is not None
        source_capsule, source_manifest = self._verify_latest(self.resume_root)
        if self.write_root == self.resume_root:
            return source_capsule, source_manifest
        self._ensure_new_run_root()
        for source in sorted(self.resume_root.glob("epoch_*")):
            if source.is_dir():
                self._verify_capsule(source)
                shutil.copytree(source, self.write_root / source.name)
        shutil.copy2(self.resume_root / "LATEST.json", self.write_root / "LATEST.json")
        return self._verify_latest(self.write_root)

    def restore_or_initialize(
        self,
        *,
        model: tf.keras.Model,
        optimizer: Any,
        scheduler: Any,
        early: Any,
        policy: Any,
        train_data: Any,
        config: dict[str, Any],
        output_dir: Path,
    ) -> tuple[int, list[dict[str, Any]]]:
        if self._prepared:
            raise ExactContinuationError("Continuation manager was prepared twice")
        self._prepared = True
        if self.resume_root is None:
            self._ensure_new_run_root()
            return 1, []
        capsule_dir, manifest = self._copy_resume_lineage()
        state = _json_object(capsule_dir / "state.json", "continuation state")
        if state.get("state_inventory") != list(REQUIRED_STATE_INVENTORY):
            raise ExactContinuationError("Continuation state inventory incomplete")
        completed_epoch = int(state.get("completed_epoch", -1))
        if completed_epoch != int(manifest.get("completed_epoch", -2)):
            raise ExactContinuationError("Continuation epoch identity mismatch")
        history = state.get("history")
        if (
            not isinstance(history, list)
            or len(history) != completed_epoch
            or history[-1].get("epoch") != completed_epoch
        ):
            raise ExactContinuationError("Continuation history is incomplete")

        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        # ``write``/``read`` deliberately omit Checkpoint.save_counter; using
        # ``restore`` would create an unmatched counter and weaken exactness.
        status = checkpoint.read(str(capsule_dir / "runtime_state"))
        try:
            status.assert_consumed()
        except (AssertionError, ValueError) as exc:
            raise ExactContinuationError(
                "TensorFlow model/optimizer checkpoint restore is incomplete"
            ) from exc

        with np.load(capsule_dir / "explicit_state.npz", allow_pickle=False) as arrays:
            inventories = state.get("variable_inventories") or {}
            # Keras SeedGenerator state can be a non-trainable variable that is
            # not reachable from a generic tf.train.Checkpoint object. The
            # explicit inventory is therefore an active restore source, not
            # merely a post-hoc checksum.
            _assign_variable_inventory(
                list(model.trainable_variables),
                inventories.get("model_trainable") or {},
                arrays,
                "trainable model",
            )
            _assign_variable_inventory(
                list(model.non_trainable_variables),
                inventories.get("model_non_trainable") or {},
                arrays,
                "non-trainable model",
            )
            _assign_variable_inventory(
                list(optimizer.variables),
                inventories.get("optimizer") or {},
                arrays,
                "optimizer",
            )
            _assert_variable_inventory(
                list(model.trainable_variables),
                inventories.get("model_trainable") or {},
                arrays,
                "trainable model",
            )
            _assert_variable_inventory(
                list(model.non_trainable_variables),
                inventories.get("model_non_trainable") or {},
                arrays,
                "non-trainable model",
            )
            _assert_variable_inventory(
                list(optimizer.variables),
                inventories.get("optimizer") or {},
                arrays,
                "optimizer",
            )
            expected_order = np.asarray(arrays["next_epoch_order"], dtype=np.int64)
            expected_aug = np.asarray(
                arrays["augmentation_seed_material"], dtype=np.uint64
            )
            numpy_keys = np.asarray(arrays["numpy_legacy_keys"], dtype=np.uint32)

        if int(optimizer.iterations.numpy()) != state.get("optimizer_iteration"):
            raise ExactContinuationError("Optimizer iteration restore mismatch")
        if float(optimizer.learning_rate.numpy()) != state.get("learning_rate"):
            raise ExactContinuationError("Learning-rate restore mismatch")
        scheduler.set_state(state["scheduler_state"])
        early.set_state(state["early_stopping_state"])
        checkpoint_state = state["checkpoint_selection_state"]
        policy.best_macro = float(checkpoint_state["best_macro"])
        policy.best_accuracy = float(checkpoint_state["best_accuracy"])
        policy.best_macro_epoch = int(checkpoint_state["best_macro_epoch"])
        policy.best_accuracy_epoch = int(checkpoint_state["best_accuracy_epoch"])

        generator = state["training_generator_contract"]
        actual_generator = _generator_contract(train_data, config, completed_epoch)
        actual_generator["dataset_epoch"] = generator.get("dataset_epoch")
        if actual_generator != generator:
            raise ExactContinuationError("Training generator contract drift")
        next_epoch = completed_epoch + 1
        actual_order = np.asarray(train_data._order(next_epoch), dtype=np.int64)
        actual_aug, _ = _augmentation_seed_material(train_data, config, next_epoch)
        if not np.array_equal(actual_order, expected_order):
            raise ExactContinuationError("Next-epoch order restore mismatch")
        if not np.array_equal(actual_aug, expected_aug):
            raise ExactContinuationError("Augmentation sequence restore mismatch")
        train_data.dataset.set_epoch(completed_epoch)

        random.setstate(_restore_tuple(state["python_rng_state"]))
        numpy_state = state["numpy_legacy_rng_state"]
        np.random.set_state(
            (
                numpy_state["algorithm"],
                numpy_keys,
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
        selected_source = capsule_dir / "selected_checkpoint"
        selected_target = output_dir / "checkpoints"
        selected_target.mkdir(parents=True, exist_ok=True)
        for name in REQUIRED_SELECTED_CHECKPOINT_FILES:
            shutil.copy2(selected_source / name, selected_target / name)
        return completed_epoch + 1, history

    def persist_epoch_boundary(
        self,
        *,
        completed_epoch: int,
        model: tf.keras.Model,
        optimizer: Any,
        scheduler: Any,
        early: Any,
        policy: Any,
        train_data: Any,
        config: dict[str, Any],
        history: list[dict[str, Any]],
        output_dir: Path,
    ) -> dict[str, Any]:
        if not self._prepared:
            raise ExactContinuationError("Continuation manager was not initialized")
        epoch = int(completed_epoch)
        if epoch < 1 or len(history) != epoch or history[-1].get("epoch") != epoch:
            raise ExactContinuationError("Only a fully completed epoch may be persisted")
        target = self.write_root / f"epoch_{epoch:06d}"
        if target.exists():
            raise ExactContinuationError(f"Continuation capsule already exists: {target}")
        staging = self.write_root / f".epoch_{epoch:06d}.{uuid.uuid4().hex}.tmp"
        staging.mkdir(parents=False)
        try:
            checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
            checkpoint.write(str(staging / "runtime_state"))
            arrays: dict[str, np.ndarray] = {}
            trainable_arrays, trainable_inventory = _variable_inventory(
                list(model.trainable_variables), "model_trainable"
            )
            non_trainable_arrays, non_trainable_inventory = _variable_inventory(
                list(model.non_trainable_variables), "model_non_trainable"
            )
            optimizer_arrays, optimizer_inventory = _variable_inventory(
                list(optimizer.variables), "optimizer"
            )
            arrays.update(trainable_arrays)
            arrays.update(non_trainable_arrays)
            arrays.update(optimizer_arrays)
            next_epoch = epoch + 1
            arrays["next_epoch_order"] = np.asarray(
                train_data._order(next_epoch), dtype=np.int64
            )
            augmentation_material, _ = _augmentation_seed_material(
                train_data, config, next_epoch
            )
            arrays["augmentation_seed_material"] = augmentation_material
            numpy_state, numpy_keys = _numpy_rng_state()
            arrays["numpy_legacy_keys"] = numpy_keys
            np.savez(staging / "explicit_state.npz", **arrays)

            generator_contract = _generator_contract(train_data, config, epoch)
            state = {
                "schema_version": CAPSULE_SCHEMA_VERSION,
                "status": CAPSULE_STATUS,
                "state_inventory": list(REQUIRED_STATE_INVENTORY),
                "completed_epoch": epoch,
                "history": history,
                "variable_inventories": {
                    "model_trainable": trainable_inventory,
                    "model_non_trainable": non_trainable_inventory,
                    "optimizer": optimizer_inventory,
                },
                "optimizer_iteration": int(optimizer.iterations.numpy()),
                "learning_rate": float(optimizer.learning_rate.numpy()),
                "scheduler_state": scheduler.get_state(),
                "early_stopping_state": early.get_state(),
                "checkpoint_selection_state": _checkpoint_selection_state(policy),
                "training_generator_contract": generator_contract,
                "python_rng_state": _jsonable_tuple(random.getstate()),
                "numpy_legacy_rng_state": numpy_state,
                "tensorflow_rng_audit": {
                    "global_generator_used_by_registered_training": False,
                    "keras_layer_seed_generators_captured_as_model_non_trainable_variables": True,
                    "augmentation_uses_epoch_sample_stateless_numpy_generators": True,
                },
                "issue60_artifacts_used": False,
            }
            _atomic_json(staging / "state.json", state)
            selected_source = output_dir / "checkpoints"
            selected_target = staging / "selected_checkpoint"
            selected_target.mkdir()
            for name in REQUIRED_SELECTED_CHECKPOINT_FILES:
                source = selected_source / name
                if not source.is_file():
                    raise ExactContinuationError(
                        f"Selected-checkpoint continuation member missing: {source}"
                    )
                shutil.copy2(source, selected_target / name)

            members = {
                path.relative_to(staging).as_posix(): _sha256(path)
                for path in sorted(staging.rglob("*"))
                if path.is_file()
            }
            capsule_basis = {
                "epoch": epoch,
                "scientific_identity": self.scientific_identity,
                "state_inventory": list(REQUIRED_STATE_INVENTORY),
                "members": members,
            }
            manifest = {
                "schema_version": CAPSULE_SCHEMA_VERSION,
                "status": CAPSULE_STATUS,
                "completed_epoch": epoch,
                "capsule_contract_sha256": CAPSULE_CONTRACT_SHA256,
                "scientific_identity": self.scientific_identity,
                "state_inventory": list(REQUIRED_STATE_INVENTORY),
                "members": members,
                "member_count": len(members),
                "capsule_sha256": _canonical_sha256(capsule_basis),
                "issue60_artifacts_used": False,
            }
            _atomic_json(staging / "manifest.json", manifest)
            os.replace(staging, target)
            latest = {
                "schema_version": CAPSULE_SCHEMA_VERSION,
                "status": CAPSULE_STATUS,
                "completed_epoch": epoch,
                "capsule_directory": target.name,
                "capsule_sha256": manifest["capsule_sha256"],
                "manifest_sha256": _sha256(target / "manifest.json"),
                "issue60_artifacts_used": False,
            }
            _atomic_json(self.write_root / "LATEST.json", latest)
            self._verify_latest(self.write_root)
            return latest
        except BaseException:
            # Deliberately retain an incomplete directory as a fail-closed marker.
            # A future restore rejects it instead of falling back silently.
            raise
