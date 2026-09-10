"""Fail-closed, atomic epoch-boundary continuation capsules for WS-HPG v1.0."""

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

from .data_order import (
    AcceptedShufflePlan,
    augmentation_parameter_matrix,
    next_epoch_stream,
)


CAPSULE_SCHEMA_VERSION = 2
CAPSULE_STATUS = "COMPLETE_EPOCH_BOUNDARY"
IMMUTABLE_PLAN_NAME = "accepted_shuffle_plan.npz"
IMMUTABLE_PLAN_MANIFEST_NAME = "accepted_shuffle_plan.manifest.json"
IMMUTABLE_AUGMENTATION_NAME = "accepted_augmentation_parameters.npz"
REQUIRED_STATE_INVENTORY = (
    "all_118_trainable_model_variables",
    "all_20_non_trainable_model_variables_including_dropout_seed_generators",
    "all_138_keras_variables",
    "adamw_variables_and_slots",
    "optimizer_iteration",
    "warmup_cosine_position_and_current_learning_rate",
    "early_stopping_best_wait_stopped_epoch_best_epoch",
    "earliest_strict_max_checkpoint_best_selected_epoch_and_artifact",
    "completed_epoch_count",
    "full_epoch_history",
    "accepted_shuffle_plan_identity",
    "next_epoch_original_sample_order",
    "next_epoch_post_shuffle_enumeration_indices",
    "next_epoch_augmentation_parameter_sequence",
    "python_rng_state",
    "numpy_rng_state",
    "tensorflow_global_rng_state",
)
REQUIRED_SELECTED_CHECKPOINT_FILES = (
    "best_val_accuracy.keras",
    "metadata.json",
)
CAPSULE_CONTRACT_DEFINITION = {
    "schema_version": CAPSULE_SCHEMA_VERSION,
    "complete_status": CAPSULE_STATUS,
    "required_state_inventory": list(REQUIRED_STATE_INVENTORY),
    "required_selected_checkpoint_files": list(REQUIRED_SELECTED_CHECKPOINT_FILES),
    "member_integrity": "sha256",
    "aggregate_integrity": "canonical_json_sha256",
    "atomic_publish": "staging_then_directory_replace_then_locked_latest",
    "partial_epoch_resume": "forbidden",
    "stale_fallback": "forbidden",
    "lineage": "contiguous_parent_hash_chain",
    "immutable_shuffle_plan": "single_root_asset_referenced_by_every_capsule",
    "immutable_augmentation_parameters": "single_root_asset_referenced_by_every_capsule",
    "invalid_issue70_state_used": False,
}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


CAPSULE_CONTRACT_SHA256 = canonical_sha256(CAPSULE_CONTRACT_DEFINITION)


class ExactContinuationError(RuntimeError):
    """A continuation capsule cannot be proven complete and exact."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExactContinuationError(f"Unreadable {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ExactContinuationError(f"{label} must be a JSON object: {path}")
    return value


def _array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _variable_inventory(variables: list[Any], prefix: str) -> tuple[dict, dict]:
    arrays: dict[str, np.ndarray] = {}
    entries = []
    for index, variable in enumerate(variables):
        key = f"{prefix}_{index:04d}"
        array = np.asarray(variable.numpy())
        arrays[key] = array
        entries.append(
            {
                "index": index,
                "key": key,
                "name": str(getattr(variable, "path", getattr(variable, "name", key))),
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "sha256": _array_hash(array),
            }
        )
    return arrays, {"count": len(entries), "variables": entries}


def _assign_and_verify_inventory(
    variables: list[Any], inventory: dict[str, Any], arrays: Any, label: str
) -> None:
    entries = inventory.get("variables")
    if not isinstance(entries, list) or inventory.get("count") != len(variables):
        raise ExactContinuationError(f"{label} variable count drift")
    for index, (variable, entry) in enumerate(zip(variables, entries)):
        key = entry.get("key")
        if entry.get("index") != index or key not in arrays:
            raise ExactContinuationError(f"{label} inventory malformed")
        expected = np.asarray(arrays[key])
        actual = np.asarray(variable.numpy())
        if (
            str(expected.dtype) != entry.get("dtype")
            or list(expected.shape) != entry.get("shape")
            or actual.shape != expected.shape
        ):
            raise ExactContinuationError(f"{label} variable shape/dtype drift")
        variable.assign(expected)
        restored = np.asarray(variable.numpy())
        if not np.array_equal(restored, expected) or _array_hash(restored) != entry.get("sha256"):
            raise ExactContinuationError(f"{label} variable {index} restore mismatch")


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
    algorithm, keys, position, has_gauss, cached = np.random.get_state()
    return (
        {
            "algorithm": algorithm,
            "position": int(position),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached),
        },
        np.asarray(keys, dtype=np.uint32),
    )


def _tensorflow_rng_state() -> tuple[dict[str, Any], np.ndarray]:
    generator = tf.random.get_global_generator()
    state = np.asarray(generator.state.numpy())
    return (
        {
            "algorithm": int(generator.algorithm),
            "used_by_stateless_registered_augmentation": False,
            "keras_dropout_state_captured_in_model_non_trainable_inventory": True,
        },
        state,
    )


def early_stopping_state(callback: tf.keras.callbacks.EarlyStopping) -> dict[str, Any]:
    return {
        "best": float(callback.best),
        "wait": int(callback.wait),
        "stopped_epoch": int(callback.stopped_epoch),
        "best_epoch": int(getattr(callback, "best_epoch", 0)),
        "monitor": callback.monitor,
        "patience": int(callback.patience),
        "min_delta": float(callback.min_delta),
        "restore_best_weights": bool(callback.restore_best_weights),
    }


def restore_early_stopping(
    callback: tf.keras.callbacks.EarlyStopping, state: dict[str, Any]
) -> None:
    expected = {
        "monitor": "val_loss",
        "patience": 15,
        "min_delta": 0.0,
        "restore_best_weights": False,
    }
    if any(state.get(key) != value for key, value in expected.items()):
        raise ExactContinuationError("EarlyStopping contract drift")
    callback.best = float(state["best"])
    callback.wait = int(state["wait"])
    callback.stopped_epoch = int(state["stopped_epoch"])
    if hasattr(callback, "best_epoch"):
        callback.best_epoch = int(state["best_epoch"])


def checkpoint_state(callback: Any) -> dict[str, Any]:
    return {
        "best": float(callback.best),
        "selected_epoch_zero_based": (
            None if callback.selected_epoch is None else int(callback.selected_epoch)
        ),
        "selected_weights_sha256": callback.selected_weights_sha256,
        "selection_policy": "earliest_strict_max_val_accuracy",
    }


class EpochBoundaryContinuationManager:
    """Create, verify and restore immutable completed-epoch capsules."""

    def __init__(
        self,
        write_root: str | Path,
        scientific_identity: dict[str, Any],
        order_plan: AcceptedShufflePlan,
        *,
        resume_root: str | Path | None = None,
    ) -> None:
        self.write_root = Path(write_root).expanduser().resolve()
        self.resume_root = None if resume_root is None else Path(resume_root).expanduser().resolve()
        self.scientific_identity = json.loads(json.dumps(scientific_identity))
        self.scientific_identity_sha256 = canonical_sha256(self.scientific_identity)
        self.order_plan = order_plan
        self._augmentation_parameters: np.ndarray | None = None
        self._prepared = False

    def _ensure_new_root(self) -> None:
        if self.write_root.exists() and any(self.write_root.iterdir()):
            raise ExactContinuationError("Epoch-0 continuation root must be absent or empty")
        self.write_root.mkdir(parents=True, exist_ok=True)

    def _publish_immutable_plan(self) -> dict[str, Any]:
        plan_path = self.write_root / IMMUTABLE_PLAN_NAME
        augmentation_path = self.write_root / IMMUTABLE_AUGMENTATION_NAME
        manifest_path = self.write_root / IMMUTABLE_PLAN_MANIFEST_NAME
        if plan_path.exists() or augmentation_path.exists() or manifest_path.exists():
            raise ExactContinuationError("Immutable shuffle plan already exists")
        temporary = plan_path.with_name(f".{plan_path.name}.{uuid.uuid4().hex}.tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, orders=np.asarray(self.order_plan.orders, dtype=np.int64))
        os.replace(temporary, plan_path)
        self._augmentation_parameters = augmentation_parameter_matrix(
            self.order_plan.orders.shape[1]
        )
        augmentation_temporary = augmentation_path.with_name(
            f".{augmentation_path.name}.{uuid.uuid4().hex}.tmp"
        )
        with augmentation_temporary.open("wb") as handle:
            np.savez(handle, parameters=self._augmentation_parameters)
        os.replace(augmentation_temporary, augmentation_path)
        augmentation_sha256 = next_epoch_stream(
            self.order_plan, 0,
            augmentation_parameters=self._augmentation_parameters,
        )["augmentation_parameters_sha256"]
        manifest = {
            "schema_version": CAPSULE_SCHEMA_VERSION,
            "status": "COMPLETE_IMMUTABLE_SHUFFLE_PLAN",
            "capsule_contract_sha256": CAPSULE_CONTRACT_SHA256,
            "plan_file": IMMUTABLE_PLAN_NAME,
            "plan_file_sha256": file_sha256(plan_path),
            "plan_values_sha256": self.order_plan.sha256,
            "augmentation_file": IMMUTABLE_AUGMENTATION_NAME,
            "augmentation_file_sha256": file_sha256(augmentation_path),
            "augmentation_values_sha256": augmentation_sha256,
            "augmentation_shape": list(self._augmentation_parameters.shape),
            "augmentation_dtype": "float32",
            "shape": list(self.order_plan.orders.shape),
            "dtype": "int64",
            "seed": int(self.order_plan.seed),
        }
        _atomic_json(manifest_path, manifest)
        return self.verify_immutable_plan(self.write_root)

    def verify_immutable_plan(self, root: str | Path | None = None) -> dict[str, Any]:
        target_root = self.write_root if root is None else Path(root).expanduser().resolve()
        manifest_path = target_root / IMMUTABLE_PLAN_MANIFEST_NAME
        manifest = _read_json(manifest_path, "immutable shuffle-plan manifest")
        if (
            manifest.get("schema_version") != CAPSULE_SCHEMA_VERSION
            or manifest.get("status") != "COMPLETE_IMMUTABLE_SHUFFLE_PLAN"
            or manifest.get("capsule_contract_sha256") != CAPSULE_CONTRACT_SHA256
            or manifest.get("plan_file") != IMMUTABLE_PLAN_NAME
        ):
            raise ExactContinuationError("Immutable shuffle-plan contract drift")
        plan_path = target_root / IMMUTABLE_PLAN_NAME
        if not plan_path.is_file():
            raise ExactContinuationError("Immutable shuffle-plan member missing")
        if file_sha256(plan_path) != manifest.get("plan_file_sha256"):
            raise ExactContinuationError("Immutable shuffle-plan member hash mismatch")
        try:
            with np.load(plan_path, allow_pickle=False) as arrays:
                persisted = np.asarray(arrays["orders"], dtype=np.int64)
        except (OSError, KeyError, ValueError) as exc:
            raise ExactContinuationError("Unreadable immutable shuffle plan") from exc
        if (
            not np.array_equal(persisted, self.order_plan.orders)
            or self.order_plan.sha256 != manifest.get("plan_values_sha256")
            or list(persisted.shape) != manifest.get("shape")
            or manifest.get("dtype") != "int64"
            or manifest.get("seed") != int(self.order_plan.seed)
        ):
            raise ExactContinuationError("Immutable shuffle-plan values drift")
        augmentation_path = target_root / IMMUTABLE_AUGMENTATION_NAME
        if (
            manifest.get("augmentation_file") != IMMUTABLE_AUGMENTATION_NAME
            or not augmentation_path.is_file()
        ):
            raise ExactContinuationError("Immutable augmentation member missing")
        if file_sha256(augmentation_path) != manifest.get("augmentation_file_sha256"):
            raise ExactContinuationError("Immutable augmentation member hash mismatch")
        try:
            with np.load(augmentation_path, allow_pickle=False) as arrays:
                parameters = np.asarray(arrays["parameters"], dtype=np.float32)
        except (OSError, KeyError, ValueError) as exc:
            raise ExactContinuationError("Unreadable immutable augmentation parameters") from exc
        expected_shape = [self.order_plan.orders.shape[1], 9]
        parameter_sha256 = next_epoch_stream(
            self.order_plan, 0, augmentation_parameters=parameters
        )["augmentation_parameters_sha256"]
        if (
            list(parameters.shape) != expected_shape
            or list(parameters.shape) != manifest.get("augmentation_shape")
            or manifest.get("augmentation_dtype") != "float32"
            or parameter_sha256 != manifest.get("augmentation_values_sha256")
        ):
            raise ExactContinuationError("Immutable augmentation values drift")
        self._augmentation_parameters = parameters
        return manifest

    @staticmethod
    def _reject_staging(root: Path) -> None:
        incomplete = sorted(
            path.name for path in root.iterdir()
            if path.is_dir() and (path.name.startswith(".epoch_") or ".tmp" in path.name)
        )
        if incomplete:
            raise ExactContinuationError(f"Incomplete newer staging capsule exists: {incomplete}")

    def verify_capsule(
        self, capsule_dir: Path, *, verify_members: bool = True
    ) -> dict[str, Any]:
        manifest = _read_json(capsule_dir / "manifest.json", "capsule manifest")
        if manifest.get("schema_version") != CAPSULE_SCHEMA_VERSION:
            raise ExactContinuationError("Capsule schema drift")
        if manifest.get("status") != CAPSULE_STATUS:
            raise ExactContinuationError("Partial epoch capsule is forbidden")
        if manifest.get("capsule_contract_sha256") != CAPSULE_CONTRACT_SHA256:
            raise ExactContinuationError("Capsule contract drift")
        if manifest.get("scientific_identity") != self.scientific_identity:
            raise ExactContinuationError("Scientific/config/source identity drift")
        if manifest.get("invalid_issue70_state_used") is not False:
            raise ExactContinuationError("Invalid Issue #70 state exclusion not proven")
        if manifest.get("immutable_shuffle_plan_sha256") != self.order_plan.sha256:
            raise ExactContinuationError("Capsule immutable shuffle-plan reference drift")
        members = manifest.get("members")
        if not isinstance(members, dict) or not members:
            raise ExactContinuationError("Capsule member inventory missing")
        if verify_members:
            actual = {
                path.relative_to(capsule_dir).as_posix()
                for path in capsule_dir.rglob("*")
                if path.is_file() and path.name != "manifest.json"
            }
            if set(members) != actual:
                raise ExactContinuationError("Capsule member inventory drift")
            for relative, expected in members.items():
                if file_sha256(capsule_dir / relative) != expected:
                    raise ExactContinuationError(f"Capsule member hash mismatch: {relative}")
            state = _read_json(capsule_dir / "state.json", "capsule state")
            if state.get("state_inventory") != list(REQUIRED_STATE_INVENTORY):
                raise ExactContinuationError("Required continuation state inventory missing")
            if state.get("accepted_shuffle_plan_sha256") != self.order_plan.sha256:
                raise ExactContinuationError("Accepted shuffle plan identity drift")
        basis = {
            "completed_epoch": manifest.get("completed_epoch"),
            "parent_capsule_sha256": manifest.get("parent_capsule_sha256"),
            "scientific_identity_sha256": self.scientific_identity_sha256,
            "immutable_shuffle_plan_sha256": self.order_plan.sha256,
            "state_inventory": list(REQUIRED_STATE_INVENTORY),
            "members": members,
        }
        if canonical_sha256(basis) != manifest.get("capsule_sha256"):
            raise ExactContinuationError("Capsule aggregate hash mismatch")
        return manifest

    def verify_latest(
        self,
        root: str | Path | None = None,
        *,
        full_lineage_members: bool = True,
    ) -> tuple[Path, dict[str, Any]]:
        target_root = self.write_root if root is None else Path(root).expanduser().resolve()
        if not target_root.is_dir():
            raise ExactContinuationError(f"Continuation root is absent: {target_root}")
        self._reject_staging(target_root)
        plan_manifest = self.verify_immutable_plan(target_root)
        latest = _read_json(target_root / "LATEST.json", "LATEST pointer")
        if (
            latest.get("status") != CAPSULE_STATUS
            or latest.get("capsule_contract_sha256") != CAPSULE_CONTRACT_SHA256
        ):
            raise ExactContinuationError("LATEST status/contract drift")
        if latest.get("scientific_identity_sha256") != self.scientific_identity_sha256:
            raise ExactContinuationError("Scientific/config/source identity drift")
        if (
            latest.get("immutable_shuffle_plan_sha256") != self.order_plan.sha256
            or latest.get("immutable_shuffle_plan_manifest_sha256")
            != file_sha256(target_root / IMMUTABLE_PLAN_MANIFEST_NAME)
            or plan_manifest.get("plan_values_sha256") != self.order_plan.sha256
        ):
            raise ExactContinuationError("LATEST immutable shuffle-plan identity drift")
        name = latest.get("capsule_directory")
        if not isinstance(name, str) or Path(name).name != name:
            raise ExactContinuationError("LATEST capsule path is invalid")
        capsule_dir = target_root / name
        manifest = self.verify_capsule(capsule_dir)
        if latest.get("manifest_sha256") != file_sha256(capsule_dir / "manifest.json"):
            raise ExactContinuationError("LATEST manifest hash mismatch")
        if latest.get("capsule_sha256") != manifest.get("capsule_sha256"):
            raise ExactContinuationError("LATEST aggregate hash mismatch")
        directories = sorted(path for path in target_root.glob("epoch_*") if path.is_dir())
        if not directories or directories[-1] != capsule_dir:
            raise ExactContinuationError("LATEST is stale")
        try:
            epochs = [int(path.name.removeprefix("epoch_")) for path in directories]
        except ValueError as exc:
            raise ExactContinuationError("Malformed epoch capsule directory") from exc
        if epochs != list(range(1, int(manifest["completed_epoch"]) + 1)):
            raise ExactContinuationError("Continuation lineage gap")
        parent = None
        for directory in directories:
            item = (
                manifest
                if directory == capsule_dir
                else self.verify_capsule(
                    directory, verify_members=full_lineage_members
                )
            )
            if item.get("parent_capsule_sha256") != parent:
                raise ExactContinuationError("Continuation parent lineage mismatch")
            parent = item["capsule_sha256"]
        return capsule_dir, manifest

    def _copy_resume_lineage(self) -> tuple[Path, dict[str, Any]]:
        assert self.resume_root is not None
        self.verify_latest(self.resume_root)
        if self.write_root == self.resume_root:
            return self.verify_latest()
        self._ensure_new_root()
        shutil.copy2(
            self.resume_root / IMMUTABLE_PLAN_NAME,
            self.write_root / IMMUTABLE_PLAN_NAME,
        )
        shutil.copy2(
            self.resume_root / IMMUTABLE_PLAN_MANIFEST_NAME,
            self.write_root / IMMUTABLE_PLAN_MANIFEST_NAME,
        )
        shutil.copy2(
            self.resume_root / IMMUTABLE_AUGMENTATION_NAME,
            self.write_root / IMMUTABLE_AUGMENTATION_NAME,
        )
        for source in sorted(self.resume_root.glob("epoch_*")):
            if source.is_dir():
                shutil.copytree(source, self.write_root / source.name)
        shutil.copy2(self.resume_root / "LATEST.json", self.write_root / "LATEST.json")
        return self.verify_latest()

    def restore_or_initialize(
        self,
        *,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        early_stop: tf.keras.callbacks.EarlyStopping,
        checkpoint_callback: Any,
        output_root: Path,
    ) -> tuple[int, list[dict[str, Any]]]:
        if self._prepared:
            raise ExactContinuationError("Continuation manager prepared twice")
        self._prepared = True
        if self.resume_root is None:
            self._ensure_new_root()
            self._publish_immutable_plan()
            return 1, []
        capsule_dir, manifest = self._copy_resume_lineage()
        state = _read_json(capsule_dir / "state.json", "capsule state")
        completed = int(state.get("completed_epoch", -1))
        if completed != int(manifest.get("completed_epoch", -2)):
            raise ExactContinuationError("Completed epoch identity mismatch")
        history = state.get("history")
        if not isinstance(history, list) or len(history) != completed:
            raise ExactContinuationError("Full epoch history is incomplete")
        if history and int(history[-1].get("epoch", -1)) != completed:
            raise ExactContinuationError("History endpoint does not match capsule")

        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        status = checkpoint.read(str(capsule_dir / "runtime_state"))
        status.expect_partial()
        with np.load(capsule_dir / "explicit_state.npz", allow_pickle=False) as arrays:
            inventories = state.get("variable_inventories") or {}
            _assign_and_verify_inventory(
                list(model.trainable_variables), inventories.get("model_trainable") or {}, arrays, "trainable model"
            )
            _assign_and_verify_inventory(
                list(model.non_trainable_variables), inventories.get("model_non_trainable") or {}, arrays, "non-trainable model"
            )
            _assign_and_verify_inventory(
                list(optimizer.variables), inventories.get("optimizer") or {}, arrays, "optimizer"
            )
            expected_order = np.asarray(arrays["next_epoch_original_sample_order"], dtype=np.int64)
            expected_indices = np.asarray(arrays["next_epoch_enumeration_indices"], dtype=np.int64)
            numpy_keys = np.asarray(arrays["numpy_rng_keys"], dtype=np.uint32)
            tensorflow_state = np.asarray(arrays["tensorflow_global_rng_state"])

        if int(optimizer.iterations.numpy()) != int(state["optimizer_iteration"]):
            raise ExactContinuationError("Optimizer iteration restore mismatch")
        current_lr = float(optimizer.learning_rate.numpy())
        if current_lr != float(state["current_learning_rate"]):
            raise ExactContinuationError("WarmupCosine position/current LR mismatch")
        restore_early_stopping(early_stop, state["early_stopping_state"])
        selected = state["checkpoint_selection_state"]
        checkpoint_callback.best = float(selected["best"])
        checkpoint_callback.selected_epoch = selected["selected_epoch_zero_based"]
        checkpoint_callback.selected_weights_sha256 = selected["selected_weights_sha256"]

        stream = next_epoch_stream(
            self.order_plan,
            completed,
            augmentation_parameters=self._augmentation_parameters,
        )
        if not np.array_equal(stream["original_sample_order"], expected_order):
            raise ExactContinuationError("Next-epoch original sample order mismatch")
        if not np.array_equal(stream["enumeration_indices"], expected_indices):
            raise ExactContinuationError("Post-shuffle enumeration index mismatch")
        if (
            state.get("next_epoch_stream", {}).get("augmentation_parameters_sha256")
            != stream["augmentation_parameters_sha256"]
        ):
            raise ExactContinuationError("Next-epoch augmentation sequence mismatch")

        random.setstate(_restore_tuple(state["python_rng_state"]))
        numpy_state = state["numpy_rng_state"]
        np.random.set_state((
            numpy_state["algorithm"], numpy_keys, int(numpy_state["position"]),
            int(numpy_state["has_gauss"]), float(numpy_state["cached_gaussian"]),
        ))
        generator = tf.random.get_global_generator()
        if int(generator.algorithm) != int(state["tensorflow_rng_state"]["algorithm"]):
            raise ExactContinuationError("TensorFlow RNG algorithm drift")
        generator.state.assign(tensorflow_state)

        checkpoint_target = output_root / "checkpoints"
        checkpoint_target.mkdir(parents=True, exist_ok=True)
        for name in REQUIRED_SELECTED_CHECKPOINT_FILES:
            shutil.copy2(capsule_dir / "selected_checkpoint" / name, checkpoint_target / name)
        return completed + 1, history

    def persist_epoch_boundary(
        self,
        *,
        completed_epoch: int,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        early_stop: tf.keras.callbacks.EarlyStopping,
        checkpoint_callback: Any,
        history: list[dict[str, Any]],
        output_root: Path,
    ) -> dict[str, Any]:
        if not self._prepared:
            raise ExactContinuationError("Continuation manager was not initialized")
        epoch = int(completed_epoch)
        if epoch < 1 or len(history) != epoch or int(history[-1].get("epoch", -1)) != epoch:
            raise ExactContinuationError("Only a fully completed epoch may be persisted")
        target = self.write_root / f"epoch_{epoch:06d}"
        if target.exists():
            raise ExactContinuationError(f"Capsule already exists: {target}")
        previous = None
        if epoch > 1:
            _, previous_manifest = self.verify_latest(full_lineage_members=False)
            if int(previous_manifest["completed_epoch"]) != epoch - 1:
                raise ExactContinuationError("Cannot publish a lineage gap")
            previous = previous_manifest["capsule_sha256"]
        staging = self.write_root / f".epoch_{epoch:06d}.{uuid.uuid4().hex}.tmp"
        staging.mkdir(parents=False)
        try:
            tf.train.Checkpoint(model=model, optimizer=optimizer).write(str(staging / "runtime_state"))
            arrays: dict[str, np.ndarray] = {}
            trainable_arrays, trainable_inventory = _variable_inventory(list(model.trainable_variables), "model_trainable")
            non_trainable_arrays, non_trainable_inventory = _variable_inventory(list(model.non_trainable_variables), "model_non_trainable")
            optimizer_arrays, optimizer_inventory = _variable_inventory(list(optimizer.variables), "optimizer")
            arrays.update(trainable_arrays)
            arrays.update(non_trainable_arrays)
            arrays.update(optimizer_arrays)
            stream = next_epoch_stream(
                self.order_plan,
                epoch,
                augmentation_parameters=self._augmentation_parameters,
            )
            arrays["next_epoch_original_sample_order"] = stream["original_sample_order"]
            arrays["next_epoch_enumeration_indices"] = stream["enumeration_indices"]
            numpy_state, numpy_keys = _numpy_rng_state()
            tensorflow_metadata, tensorflow_state = _tensorflow_rng_state()
            arrays["numpy_rng_keys"] = numpy_keys
            arrays["tensorflow_global_rng_state"] = tensorflow_state
            np.savez(staging / "explicit_state.npz", **arrays)
            schedule = optimizer._learning_rate
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
                "model_variable_counts": {
                    "trainable": len(model.trainable_variables),
                    "non_trainable": len(model.non_trainable_variables),
                    "keras_total": len(model.variables),
                },
                "optimizer_iteration": int(optimizer.iterations.numpy()),
                "current_learning_rate": float(optimizer.learning_rate.numpy()),
                "warmup_cosine_state": {
                    "class_name": type(schedule).__name__,
                    "config": schedule.get_config(),
                    "position": int(optimizer.iterations.numpy()),
                },
                "early_stopping_state": early_stopping_state(early_stop),
                "checkpoint_selection_state": checkpoint_state(checkpoint_callback),
                "accepted_shuffle_plan_sha256": self.order_plan.sha256,
                "next_epoch_stream": {
                    key: stream[key] for key in (
                        "next_epoch", "order_sha256", "enumeration_sha256", "augmentation_parameters_sha256"
                    )
                },
                "python_rng_state": _jsonable_tuple(random.getstate()),
                "numpy_rng_state": numpy_state,
                "tensorflow_rng_state": tensorflow_metadata,
                "invalid_issue70_state_used": False,
                "test_access": False,
            }
            _atomic_json(staging / "state.json", state)
            selected_target = staging / "selected_checkpoint"
            selected_target.mkdir()
            selected_source = output_root / "checkpoints"
            for name in REQUIRED_SELECTED_CHECKPOINT_FILES:
                source = selected_source / name
                if not source.is_file():
                    raise ExactContinuationError(f"Selected checkpoint member missing: {source}")
                shutil.copy2(source, selected_target / name)
            members = {
                path.relative_to(staging).as_posix(): file_sha256(path)
                for path in sorted(staging.rglob("*")) if path.is_file()
            }
            basis = {
                "completed_epoch": epoch,
                "parent_capsule_sha256": previous,
                "scientific_identity_sha256": self.scientific_identity_sha256,
                "immutable_shuffle_plan_sha256": self.order_plan.sha256,
                "state_inventory": list(REQUIRED_STATE_INVENTORY),
                "members": members,
            }
            manifest = {
                "schema_version": CAPSULE_SCHEMA_VERSION,
                "status": CAPSULE_STATUS,
                "completed_epoch": epoch,
                "parent_capsule_sha256": previous,
                "capsule_contract_sha256": CAPSULE_CONTRACT_SHA256,
                "scientific_identity": self.scientific_identity,
                "scientific_identity_sha256": self.scientific_identity_sha256,
                "immutable_shuffle_plan_sha256": self.order_plan.sha256,
                "state_inventory": list(REQUIRED_STATE_INVENTORY),
                "members": members,
                "member_count": len(members),
                "capsule_sha256": canonical_sha256(basis),
                "invalid_issue70_state_used": False,
                "test_access": False,
            }
            _atomic_json(staging / "manifest.json", manifest)
            os.replace(staging, target)
            latest = {
                "schema_version": CAPSULE_SCHEMA_VERSION,
                "status": CAPSULE_STATUS,
                "completed_epoch": epoch,
                "capsule_directory": target.name,
                "capsule_contract_sha256": CAPSULE_CONTRACT_SHA256,
                "scientific_identity_sha256": self.scientific_identity_sha256,
                "immutable_shuffle_plan_sha256": self.order_plan.sha256,
                "immutable_shuffle_plan_manifest_sha256": file_sha256(
                    self.write_root / IMMUTABLE_PLAN_MANIFEST_NAME
                ),
                "capsule_sha256": manifest["capsule_sha256"],
                "manifest_sha256": file_sha256(target / "manifest.json"),
                "invalid_issue70_state_used": False,
            }
            _atomic_json(self.write_root / "LATEST.json", latest)
            self.verify_latest(full_lineage_members=False)
            return latest
        except Exception:
            # Retain the staging directory as an unmistakable incomplete marker.
            raise
