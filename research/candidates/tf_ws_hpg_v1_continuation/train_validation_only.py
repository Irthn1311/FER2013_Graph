"""Technical-only WS-HPG lifecycle with exact completed-epoch continuation.

This module does not execute on import.  It preserves the accepted Issue #70
scientific lifecycle and adds only atomic continuation plus a preregistered
epoch-60 technical pause.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from research.candidates.tf_ws_hpg_v1_continuation import (  # noqa: E402
    IMPLEMENTATION_STATUS,
    PLANNED_PAUSE_STATUS,
)
from research.candidates.tf_ws_hpg_v1_continuation.continuation import (  # noqa: E402
    CAPSULE_CONTRACT_SHA256,
    EpochBoundaryContinuationManager,
    IMMUTABLE_PLAN_NAME,
    canonical_sha256,
    early_stopping_state,
    file_sha256,
    restore_early_stopping,
)
from research.candidates.tf_ws_hpg_v1_continuation.data_order import (  # noqa: E402
    AcceptedShufflePlan,
    build_segment_training_dataset,
)
from research.candidates.tf_ws_hpg_v1_training import (  # noqa: E402
    train_validation_only as accepted,
)
from research.candidates.tf_ws_hpg_v1_training.data import (  # noqa: E402
    TRAIN_SAMPLES,
    VALIDATION_SAMPLES,
    build_dataset,
    load_fer_csv,
    load_support_split,
    reject_test_path,
)
from research.candidates.tf_ws_hpg_v1_weak_support.model import (  # noqa: E402
    build_ws_hpg_v1_weak_support,
)


EXACT_PARENT = "ab7c7a49e923764b6192d9e774352fddf8f1df8b"
PLANNED_PAUSE_EPOCH = 60
MAX_EPOCHS = 100
EXPECTED_MODEL_SHA256 = "177a782cd8d5c2178303c44d120dcdd22b0a2108a0b720c5091a19f3d7cbffe3"
EXPECTED_SUPPORT_SHA256 = "b6ed2ddd20a4e82824208929709ff2d6bcb1c5557144d778ed0157fd4768aeee"
EXPECTED_TRAINING_SOURCE_SHA256 = {
    "__init__.py": "b211b99534a7c3e9ff57ea5ac0b2a7f059307cb56d49f9f1e1214189e6402f43",
    "augmentation.py": "cd89a727a2fb0037dad6a51da87227101f4d51f9e6a8945f5ccca223d66967cd",
    "data.py": "c9a0310a2dcda5eab779366b7c2a9357961c273ea2279525ccfb6453f79823b7",
    "train_validation_only.py": "5b84f29703bc53fbcf5941ec3abbd3af434c6b289156a56845541b35b9fadbe9",
}
EXPECTED_IDENTITY = {
    "parameters": 707_213,
    "trainable_variables": 118,
    "keras_variables": 138,
}
TRAINING_ROOT = REPOSITORY_ROOT / "research/candidates/tf_ws_hpg_v1_training"
WS_ROOT = REPOSITORY_ROOT / "research/candidates/tf_ws_hpg_v1_weak_support"
CONTINUATION_ROOT = Path(__file__).resolve().parent


class WSContinuationError(RuntimeError):
    """Fail-closed WS continuation or lifecycle error."""


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _variables_sha256(variables) -> str:
    digest = hashlib.sha256()
    for variable in variables:
        value = np.asarray(variable.numpy())
        digest.update(str(getattr(variable, "path", variable.name)).encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def verify_locked_sources() -> dict[str, Any]:
    actual_training = {
        name: file_sha256(TRAINING_ROOT / name)
        for name in EXPECTED_TRAINING_SOURCE_SHA256
    }
    actual = {
        "model.py": file_sha256(WS_ROOT / "model.py"),
        "support.py": file_sha256(WS_ROOT / "support.py"),
        "accepted_training": actual_training,
    }
    expected = {
        "model.py": EXPECTED_MODEL_SHA256,
        "support.py": EXPECTED_SUPPORT_SHA256,
        "accepted_training": EXPECTED_TRAINING_SOURCE_SHA256,
    }
    if actual != expected:
        raise WSContinuationError(f"Accepted WS source identity drift: {actual}")
    if accepted.TRAINING_CONFIG != {
        "seed": 42,
        "optimizer": "AdamW",
        "learning_rate": 3e-4,
        "weight_decay": 5e-4,
        "global_clipnorm": 1.0,
        "batch_size": 64,
        "max_epochs": 100,
        "warmup_epochs": 5,
        "cosine_final_learning_rate": 1e-6,
        "training_label_smoothing": 0.05,
        "validation_every_epochs": 1,
        "checkpoint": "earliest_strict_max_val_accuracy",
        "early_stopping_monitor": "val_loss",
        "early_stopping_patience": 15,
        "early_stopping_min_delta": 0.0,
        "mixed_precision": False,
        "xla": False,
        "mirrored_strategy": False,
        "support_dropout": False,
    }:
        raise WSContinuationError("Accepted Issue #70 training configuration drift")
    return actual


def validate_model_identity(model: tf.keras.Model) -> dict[str, int]:
    observed = {
        "parameters": int(model.count_params()),
        "trainable_variables": len(model.trainable_variables),
        "keras_variables": len(model.variables),
    }
    if observed != EXPECTED_IDENTITY or len(model.non_trainable_variables) != 20:
        raise WSContinuationError(f"Accepted WS architecture identity drift: {observed}")
    return observed


def scientific_identity(order_plan: AcceptedShufflePlan, sample_count: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "exact_parent": EXACT_PARENT,
        "model_sha256": EXPECTED_MODEL_SHA256,
        "support_sha256": EXPECTED_SUPPORT_SHA256,
        "accepted_training_source_sha256": EXPECTED_TRAINING_SOURCE_SHA256,
        "continuation_runtime_source_sha256": {
            name: file_sha256(CONTINUATION_ROOT / name)
            for name in ("__init__.py", "continuation.py", "data_order.py", "train_validation_only.py")
        },
        "architecture_identity": EXPECTED_IDENTITY,
        "training_config": accepted.TRAINING_CONFIG,
        "accepted_shuffle": {
            "implementation": "tf.data.Dataset.shuffle",
            "seed": 42,
            "reshuffle_each_iteration": True,
            "enumerate": "after_shuffle",
            "keras_3_15_training_shuffle_iterations": "one_based_odd_1_3_5_etc",
            "sample_count": int(sample_count),
            "epochs_materialized": int(len(order_plan.orders)),
            "plan_sha256": order_plan.sha256,
        },
        "continuation_only": True,
        "planned_pause_epoch": PLANNED_PAUSE_EPOCH,
        "invalid_issue70_state_used": False,
        "test_access": False,
    }


def _load_persisted_order_plan(
    resume_root: str | Path, sample_count: int, *, max_epochs: int = MAX_EPOCHS
) -> AcceptedShufflePlan:
    """Load the single immutable plan; the manager then verifies both hashes."""

    root = Path(resume_root).expanduser().resolve()
    try:
        with np.load(root / IMMUTABLE_PLAN_NAME, allow_pickle=False) as arrays:
            plan = AcceptedShufflePlan.from_orders(arrays["orders"])
        plan.verify(sample_count=sample_count, max_epochs=int(max_epochs) + 1)
        return plan
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise WSContinuationError("Cannot load the locked immutable accepted shuffle plan") from exc


class _RestoreEarlyStoppingOnFitBegin(tf.keras.callbacks.Callback):
    """Undo Keras' fit-boundary reset after a verified capsule restore."""

    def __init__(self, callback, state):
        super().__init__()
        self.callback = callback
        self.state = state

    def on_train_begin(self, logs=None):
        restore_early_stopping(self.callback, self.state)


class _EpochBoundaryCapsuleCallback(tf.keras.callbacks.Callback):
    """Persist only after accepted checkpoint/early-stop epoch bookkeeping."""

    def __init__(self, *, manager, optimizer, early_stop, checkpoint, history, output_root):
        super().__init__()
        self.manager = manager
        self.optimizer = optimizer
        self.early_stop = early_stop
        self.checkpoint = checkpoint
        self.history = history
        self.output_root = output_root
        self.planned_pause = False

    def on_epoch_end(self, epoch, logs=None):
        values = dict(logs or {})
        row = {"epoch": int(epoch) + 1}
        required = {"loss", "accuracy", "val_loss", "val_accuracy"}
        for key, value in values.items():
            numeric = float(value)
            if not math.isfinite(numeric):
                raise WSContinuationError(f"Non-finite epoch metric: {key}")
            row[key] = numeric
        if not required <= set(row):
            raise WSContinuationError(f"Epoch history missing fields: {required - set(row)}")
        self.history.append(row)
        self.manager.persist_epoch_boundary(
            completed_epoch=int(epoch) + 1,
            model=self.model,
            optimizer=self.optimizer,
            early_stop=self.early_stop,
            checkpoint_callback=self.checkpoint,
            history=self.history,
            output_root=self.output_root,
        )
        # Natural accepted early stopping has priority over the technical pause.
        if not self.model.stop_training and int(epoch) + 1 == PLANNED_PAUSE_EPOCH:
            self.planned_pause = True
            self.model.stop_training = True


class ContinuableEarliestStrictMaximumCheckpoint(accepted.EarliestStrictMaximumCheckpoint):
    """Accepted checkpoint rule plus explicit continuation metadata."""

    def __init__(self, output_root):
        super().__init__(output_root)
        self.selected_weights_sha256 = None

    def on_epoch_end(self, epoch, logs=None):
        previous = self.selected_epoch
        super().on_epoch_end(epoch, logs)
        if self.selected_epoch != previous:
            self.selected_weights_sha256 = _variables_sha256(self.model.variables)
        metadata = {
            "selection_policy": "earliest_strict_max_val_accuracy",
            "best": float(self.best),
            "selected_epoch_zero_based": self.selected_epoch,
            "selected_weights_sha256": self.selected_weights_sha256,
        }
        _atomic_json(self.output_root / "checkpoints" / "metadata.json", metadata)


def _build_runtime(output_root: Path, steps_per_epoch: int):
    tf.keras.backend.clear_session()
    random.seed(42)
    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)
    # The accepted model's Dropout layers own Keras SeedGenerators and the
    # augmentation is stateless.  The TensorFlow global generator is therefore
    # not consumed by the registered path, but initialize it explicitly so its
    # audited continuation state is reproducible rather than OS-seeded.
    tf.random.set_global_generator(tf.random.Generator.from_seed(42))
    model = build_ws_hpg_v1_weak_support()
    validate_model_identity(model)
    optimizer = accepted.build_optimizer(steps_per_epoch)
    optimizer.build(model.trainable_variables)
    model.compile(
        optimizer=optimizer,
        loss=accepted.training_loss,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    checkpoint = ContinuableEarliestStrictMaximumCheckpoint(output_root)
    checkpoint.set_model(model)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        min_delta=0.0,
        restore_best_weights=False,
    )
    early_stop.set_model(model)
    early_stop.on_train_begin()
    return model, optimizer, checkpoint, early_stop


def _history_row(epoch: int, values: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    row: dict[str, Any] = {"epoch": int(epoch)}
    for key, series in values.items():
        if len(series) != 1 or not math.isfinite(float(series[0])):
            raise WSContinuationError(f"Non-finite/incomplete epoch metric: {key}")
        row[key] = float(series[0])
    required = {"loss", "accuracy", "val_loss", "val_accuracy"}
    if not required <= set(row):
        raise WSContinuationError(f"Epoch history missing fields: {required - set(row)}")
    return row


def _terminal_evaluation(
    output_root: Path,
    clean_train_dataset,
    validation_dataset,
    support_diagnostics: Mapping[str, Any],
    history: list[dict[str, Any]],
    checkpoint: ContinuableEarliestStrictMaximumCheckpoint,
) -> dict[str, Any]:
    selected_path = output_root / "checkpoints" / "best_val_accuracy.keras"
    selected = tf.keras.models.load_model(selected_path, compile=False)
    before = _variables_sha256(selected.variables)
    clean = accepted.evaluate(selected, clean_train_dataset, support_override="normal")
    normal = accepted.evaluate(selected, validation_dataset, support_override="normal")
    no_prior = accepted.evaluate(selected, validation_dataset, support_override="ones")
    after = _variables_sha256(selected.variables)
    if before != after:
        raise WSContinuationError("Selected checkpoint changed during terminal evaluation")
    dependency_accuracy = 100.0 * (normal["accuracy"] - no_prior["accuracy"])
    dependency_macro = 100.0 * (normal["macro_f1"] - no_prior["macro_f1"])
    result = {
        "status": "COMPLETE",
        "scientific_result_valid": True,
        "implementation_status": IMPLEMENTATION_STATUS,
        "selected_epoch_zero_based": checkpoint.selected_epoch,
        "epochs_completed": len(history),
        "clean_train": clean,
        "validation_normal_support": normal,
        "validation_all_ones_support": no_prior,
        "accuracy_gap_pp": 100.0 * (clean["accuracy"] - normal["accuracy"]),
        "macro_gap_pp": 100.0 * (clean["macro_f1"] - normal["macro_f1"]),
        "support_dependency_accuracy_pp": dependency_accuracy,
        "support_dependency_macro_pp": dependency_macro,
        "selected_weights_sha256_before_after": [before, after],
        "decision": accepted.classify_outcome(
            validation_accuracy=normal["accuracy"],
            validation_macro_f1=normal["macro_f1"],
            clean_train_macro_f1=clean["macro_f1"],
            support_dependency_accuracy_pp=dependency_accuracy,
            support_dependency_macro_pp=dependency_macro,
        ),
        "support_diagnostics": dict(support_diagnostics),
        "terminal_evaluation_inventory": [
            "clean_train_normal_support",
            "validation_normal_support",
            "validation_all_ones_support",
        ],
        "training": True,
        "test_access": False,
    }
    _atomic_json(output_root / "validation_only_result.json", result)
    return result


def _planned_pause_result(
    manager: EpochBoundaryContinuationManager,
    completed_epoch: int,
) -> dict[str, Any]:
    capsule_dir, manifest = manager.verify_latest()
    if int(completed_epoch) != PLANNED_PAUSE_EPOCH:
        raise WSContinuationError("Planned pause is registered only for epoch 60")
    if int(manifest.get("completed_epoch", -1)) != PLANNED_PAUSE_EPOCH:
        raise WSContinuationError("Planned pause requires a verified complete epoch-60 capsule")
    return {
        "schema_version": 1,
        "status": PLANNED_PAUSE_STATUS,
        "scientific_result_valid": False,
        "scientific_interpretation": None,
        "completed_epoch": PLANNED_PAUSE_EPOCH,
        "capsule_verified": True,
        "capsule_directory": str(capsule_dir),
        "capsule_contract_sha256": CAPSULE_CONTRACT_SHA256,
        "latest": manifest,
        "final_selected_checkpoint_evaluation_performed": False,
        "partial_following_epoch_used": False,
        "invalid_issue70_state_used": False,
        "training": True,
        "test_access": False,
    }


def run_continuable_lifecycle(
    *,
    train_images,
    train_support,
    train_labels,
    validation_dataset,
    clean_train_dataset,
    output_root: str | Path,
    continuation_root: str | Path,
    support_diagnostics: Mapping[str, Any] | None = None,
    resume_capsule_root: str | Path | None = None,
    planned_pause_epoch: int = PLANNED_PAUSE_EPOCH,
    max_epochs: int = MAX_EPOCHS,
    batch_size: int = 64,
    verbose: int = 2,
) -> dict[str, Any]:
    """Run only when separately authorized; default is a fixed epoch-60 pause."""

    if int(max_epochs) != MAX_EPOCHS or int(batch_size) != 64:
        raise WSContinuationError("Registered max_epochs/batch_size drift")
    if int(planned_pause_epoch) != PLANNED_PAUSE_EPOCH:
        raise WSContinuationError("Registered planned pause must remain epoch 60")
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    sample_count = int(len(train_labels))
    order_plan = (
        AcceptedShufflePlan.materialize(sample_count, MAX_EPOCHS + 1)
        if resume_capsule_root is None
        else _load_persisted_order_plan(resume_capsule_root, sample_count)
    )
    model, optimizer, checkpoint, early_stop = _build_runtime(
        output_root, math.ceil(sample_count / batch_size)
    )
    manager = EpochBoundaryContinuationManager(
        continuation_root,
        scientific_identity(order_plan, sample_count),
        order_plan,
        resume_root=resume_capsule_root,
    )
    start_epoch, history = manager.restore_or_initialize(
        model=model,
        optimizer=optimizer,
        early_stop=early_stop,
        checkpoint_callback=checkpoint,
        output_root=output_root,
    )
    if resume_capsule_root is None and start_epoch != 1:
        raise WSContinuationError("A fresh run must start at epoch 1/epoch-0 state")
    if resume_capsule_root is not None and start_epoch < 2:
        raise WSContinuationError("Resume requires at least one verified fully completed epoch")

    train_segment = build_segment_training_dataset(
        train_images,
        train_support,
        train_labels,
        plan=order_plan,
        start_epoch=start_epoch,
        end_epoch=MAX_EPOCHS,
        batch_size=batch_size,
    )
    callbacks: list[tf.keras.callbacks.Callback] = [checkpoint, early_stop]
    if resume_capsule_root is not None:
        callbacks.append(
            _RestoreEarlyStoppingOnFitBegin(
                early_stop, early_stopping_state(early_stop)
            )
        )
    capsule_callback = _EpochBoundaryCapsuleCallback(
        manager=manager,
        optimizer=optimizer,
        early_stop=early_stop,
        checkpoint=checkpoint,
        history=history,
        output_root=output_root,
    )
    callbacks.append(capsule_callback)
    model.fit(
        train_segment,
        validation_data=validation_dataset,
        validation_freq=1,
        initial_epoch=start_epoch - 1,
        epochs=MAX_EPOCHS,
        steps_per_epoch=math.ceil(sample_count / batch_size),
        callbacks=callbacks,
        verbose=verbose,
    )
    if capsule_callback.planned_pause:
        pause = _planned_pause_result(manager, PLANNED_PAUSE_EPOCH)
        _atomic_json(output_root / "planned_pause.json", pause)
        return pause

    return _terminal_evaluation(
        output_root,
        clean_train_dataset,
        validation_dataset,
        support_diagnostics or {},
        history,
        checkpoint,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WS-HPG exact epoch-boundary continuation; validation only"
    )
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--continuation-root", type=Path, required=True)
    parser.add_argument("--resume-capsule-root", type=Path)
    parser.add_argument("--planned-pause-epoch", type=int, default=PLANNED_PAUSE_EPOCH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    verify_locked_sources()
    train_csv = reject_test_path(args.train_csv)
    val_csv = reject_test_path(args.val_csv)
    prior_root = reject_test_path(args.prior_root)
    reject_test_path(args.output_root)
    reject_test_path(args.continuation_root)
    if args.resume_capsule_root is not None:
        reject_test_path(args.resume_capsule_root)
    train_images, train_labels = load_fer_csv(train_csv, TRAIN_SAMPLES)
    val_images, val_labels = load_fer_csv(val_csv, VALIDATION_SAMPLES)
    train_support, train_detected, train_coverage = load_support_split(
        prior_root, "train", train_images, train_labels
    )
    val_support, val_detected, val_coverage = load_support_split(
        prior_root, "val", val_images, val_labels
    )
    validation = build_dataset(val_images, val_support, val_labels, training=False)
    clean_train = build_dataset(train_images, train_support, train_labels, training=False)
    diagnostics = {
        "train": {
            "coverage_mean": float(np.mean(train_coverage)),
            "coverage_std": float(np.std(train_coverage)),
            "detector_failure_count": int(np.count_nonzero(~train_detected)),
            "detector_failure_rate": float(np.mean(~train_detected)),
        },
        "validation": {
            "coverage_mean": float(np.mean(val_coverage)),
            "coverage_std": float(np.std(val_coverage)),
            "detector_failure_count": int(np.count_nonzero(~val_detected)),
            "detector_failure_rate": float(np.mean(~val_detected)),
        },
    }
    result = run_continuable_lifecycle(
        train_images=train_images,
        train_support=train_support,
        train_labels=train_labels,
        validation_dataset=validation,
        clean_train_dataset=clean_train,
        output_root=args.output_root,
        continuation_root=args.continuation_root,
        support_diagnostics=diagnostics,
        resume_capsule_root=args.resume_capsule_root,
        planned_pause_epoch=args.planned_pause_epoch,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
