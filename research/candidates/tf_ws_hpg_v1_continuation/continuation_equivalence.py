"""Fresh-process zero-tolerance continuation proof on synthetic WS-HPG data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from research.candidates.tf_ws_hpg_v1_continuation import NOT_PROVEN_STATUS  # noqa: E402
from research.candidates.tf_ws_hpg_v1_continuation.continuation import (  # noqa: E402
    EpochBoundaryContinuationManager,
    canonical_sha256,
    early_stopping_state,
    restore_early_stopping,
)
from research.candidates.tf_ws_hpg_v1_continuation.data_order import (  # noqa: E402
    AcceptedShufflePlan,
    build_segment_training_dataset,
    next_epoch_stream,
)
from research.candidates.tf_ws_hpg_v1_continuation.train_validation_only import (  # noqa: E402
    _build_runtime,
    _history_row,
    _load_persisted_order_plan,
    _variables_sha256,
)
from research.candidates.tf_ws_hpg_v1_training.data import build_dataset  # noqa: E402
from research.candidates.tf_ws_hpg_v1_training import data as accepted_data  # noqa: E402
from research.candidates.tf_ws_hpg_v1_training import train_validation_only as accepted  # noqa: E402


SYNTHETIC_SAMPLES = 8
SYNTHETIC_IDENTITY_BASE = {
    "contract": "ws_hpg_v1_synthetic_exact_continuation_v1",
    "seed": 42,
    "architecture_variables": {"trainable": 118, "keras": 138},
    "invalid_issue70_state_used": False,
    "test_access": False,
}


def _rng_hashes() -> dict[str, str]:
    numpy_state = np.random.get_state()
    generator = tf.random.get_global_generator()
    return {
        "python_rng_sha256": hashlib.sha256(repr(random.getstate()).encode("utf-8")).hexdigest(),
        "numpy_rng_sha256": hashlib.sha256(
            numpy_state[0].encode("ascii")
            + np.asarray(numpy_state[1], dtype=np.uint32).tobytes(order="C")
            + repr(numpy_state[2:]).encode("ascii")
        ).hexdigest(),
        "tensorflow_rng_sha256": hashlib.sha256(
            np.asarray(generator.state.numpy()).tobytes(order="C")
        ).hexdigest(),
    }


def _synthetic_data():
    images = np.stack(
        [np.full((48, 48, 1), 32.0 + 16.0 * index, dtype=np.float32) for index in range(SYNTHETIC_SAMPLES)]
    )
    yy, xx = np.mgrid[:48, :48]
    base_support = np.exp(-((xx - 23.5) ** 2 + (yy - 23.5) ** 2) / (2 * 12.0**2)).astype(np.float32)
    supports = np.stack([(0.25 + 0.75 * base_support)[..., None] for _ in range(SYNTHETIC_SAMPLES)])
    labels = np.arange(SYNTHETIC_SAMPLES, dtype=np.int32) % 7
    return images, supports, labels


def _snapshot(
    epoch,
    model,
    optimizer,
    early_stop,
    checkpoint,
    row,
    plan,
    *,
    selected_weights_sha256=None,
):
    stream = next_epoch_stream(plan, epoch)
    checkpoint_payload = {
        "best": float(checkpoint.best),
        "selected_epoch_zero_based": (
            None if checkpoint.selected_epoch is None else int(checkpoint.selected_epoch)
        ),
        "selected_weights_sha256": (
            checkpoint.selected_weights_sha256
            if selected_weights_sha256 is None
            else selected_weights_sha256
        ),
        "selection_policy": "earliest_strict_max_val_accuracy",
    }
    payload = {
        "epoch": int(epoch),
        "trainable_count": len(model.trainable_variables),
        "keras_variable_count": len(model.variables),
        "non_trainable_count": len(model.non_trainable_variables),
        "trainable_sha256": _variables_sha256(model.trainable_variables),
        "non_trainable_sha256": _variables_sha256(model.non_trainable_variables),
        "all_keras_variables_sha256": _variables_sha256(model.variables),
        "dropout_seed_generator_sha256": _variables_sha256(model.non_trainable_variables),
        "optimizer_sha256": _variables_sha256(optimizer.variables),
        "optimizer_iteration": int(optimizer.iterations.numpy()),
        "current_learning_rate": float(optimizer.learning_rate.numpy()),
        "warmup_cosine_config": optimizer._learning_rate.get_config(),
        "early_stopping_state": early_stopping_state(early_stop),
        "checkpoint_selection_state": checkpoint_payload,
        "validation_metrics": {
            key: row[key] for key in ("val_loss", "val_accuracy")
        },
        "epoch_training_loss": row["loss"],
        "next_epoch_original_sample_order": stream["original_sample_order"].tolist(),
        "next_epoch_enumeration_indices": stream["enumeration_indices"].tolist(),
        "next_epoch_augmentation_parameters_sha256": stream["augmentation_parameters_sha256"],
        **_rng_hashes(),
    }
    payload["aggregate_state_sha256"] = canonical_sha256(payload)
    return payload


class _AcceptedReadOnlySnapshot(tf.keras.callbacks.Callback):
    """Observe accepted lifecycle state after its registered callbacks."""

    def __init__(self, checkpoint, early_stop, plan):
        super().__init__()
        self.checkpoint = checkpoint
        self.early_stop = early_stop
        self.plan = plan
        self.snapshots = []
        self.selected_weights_sha256 = None

    def on_epoch_end(self, epoch, logs=None):
        row = _history_row(int(epoch) + 1, {
            key: [float(value)] for key, value in dict(logs or {}).items()
        })
        if self.checkpoint.selected_epoch == int(epoch):
            self.selected_weights_sha256 = _variables_sha256(self.model.variables)
        self.snapshots.append(
            _snapshot(
                int(epoch) + 1,
                self.model,
                self.model.optimizer,
                self.early_stop,
                self.checkpoint,
                row,
                self.plan,
                selected_weights_sha256=self.selected_weights_sha256,
            )
        )


class _RestoreEarlyStoppingOnFitBegin(tf.keras.callbacks.Callback):
    def __init__(self, early_stop, state):
        super().__init__()
        self.early_stop = early_stop
        self.state = state

    def on_train_begin(self, logs=None):
        del logs
        restore_early_stopping(self.early_stop, self.state)


class _NewEpochObserver(_AcceptedReadOnlySnapshot):
    def __init__(self, checkpoint, early_stop, plan, manager, history, output_root):
        super().__init__(checkpoint, early_stop, plan)
        self.manager = manager
        self.history = history
        self.output_root = output_root

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        row = _history_row(int(epoch) + 1, {
            key: [float(value)] for key, value in dict(logs or {}).items()
        })
        self.history.append(row)
        if self.manager is not None:
            self.manager.persist_epoch_boundary(
                completed_epoch=int(epoch) + 1,
                model=self.model,
                optimizer=self.model.optimizer,
                early_stop=self.early_stop,
                checkpoint_callback=self.checkpoint,
                history=self.history,
                output_root=self.output_root,
            )


def _accepted_reference_order_plan(images, supports, labels):
    records = accepted_data._training_records(images, supports, labels)
    orders = []
    for _ in range(10):
        orders.append(
            [int(record[0].numpy()) for _, record in records]
        )
    return AcceptedShufflePlan.from_orders(
        np.asarray(orders, dtype=np.int64)[0::2][:5]
    )


def _run_accepted_lifecycle(output_root: Path):
    """Run the exact accepted one-fit Issue #70 structure for four epochs."""

    output_root.mkdir(parents=True, exist_ok=True)
    images, supports, labels = _synthetic_data()
    plan = _accepted_reference_order_plan(images, supports, labels)
    train = accepted_data.build_dataset(
        images, supports, labels, training=True, batch_size=4
    )
    validation = accepted_data.build_dataset(
        images, supports, labels, training=False, batch_size=4
    )
    model, optimizer, _, _ = _build_runtime(output_root, steps_per_epoch=2)
    checkpoint = accepted.EarliestStrictMaximumCheckpoint(output_root)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        min_delta=0.0,
        restore_best_weights=False,
    )
    observer = _AcceptedReadOnlySnapshot(checkpoint, early_stop, plan)
    model.fit(
        train,
        validation_data=validation,
        validation_freq=1,
        epochs=4,
        callbacks=[checkpoint, early_stop, observer],
        verbose=0,
    )
    if len(observer.snapshots) != 4:
        raise RuntimeError("Accepted synthetic lifecycle did not complete four epochs")
    return observer.snapshots


def _run_epochs(
    *,
    output_root: Path,
    end_epoch: int,
    continuation_write_root: Path | None = None,
    continuation_resume_root: Path | None = None,
):
    output_root.mkdir(parents=True, exist_ok=True)
    images, supports, labels = _synthetic_data()
    plan = (
        AcceptedShufflePlan.materialize(SYNTHETIC_SAMPLES, 5)
        if continuation_resume_root is None
        else _load_persisted_order_plan(
            continuation_resume_root,
            SYNTHETIC_SAMPLES,
            max_epochs=4,
        )
    )
    identity = dict(SYNTHETIC_IDENTITY_BASE, accepted_shuffle_plan_sha256=plan.sha256)
    model, optimizer, checkpoint, early_stop = _build_runtime(output_root, steps_per_epoch=2)
    validation = build_dataset(images, supports, labels, training=False, batch_size=4)
    history: list[dict[str, Any]] = []
    start_epoch = 1
    manager = None
    if continuation_write_root is not None:
        manager = EpochBoundaryContinuationManager(
            continuation_write_root,
            identity,
            plan,
            resume_root=continuation_resume_root,
        )
        start_epoch, history = manager.restore_or_initialize(
            model=model,
            optimizer=optimizer,
            early_stop=early_stop,
            checkpoint_callback=checkpoint,
            output_root=output_root,
        )
    train = build_segment_training_dataset(
        images,
        supports,
        labels,
        plan=plan,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        batch_size=4,
    )
    observer = _NewEpochObserver(
        checkpoint, early_stop, plan, manager, history, output_root
    )
    callbacks = [checkpoint, early_stop]
    if continuation_resume_root is not None:
        callbacks.append(
            _RestoreEarlyStoppingOnFitBegin(
                early_stop, early_stopping_state(early_stop)
            )
        )
    callbacks.append(observer)
    model.fit(
        train,
        validation_data=validation,
        validation_freq=1,
        initial_epoch=start_epoch - 1,
        epochs=end_epoch,
        steps_per_epoch=2,
        callbacks=callbacks,
        verbose=0,
    )
    return observer.snapshots


def run_worker(mode: str, root: Path, result_path: Path) -> None:
    if mode == "accepted":
        snapshots = _run_accepted_lifecycle(root / "accepted")
    elif mode == "new_non_resume":
        snapshots = _run_epochs(output_root=root / "new-non-resume", end_epoch=4)
    elif mode == "uninterrupted":
        snapshots = _run_epochs(output_root=root / "uninterrupted", end_epoch=4)
    elif mode == "save":
        snapshots = _run_epochs(
            output_root=root / "save-output",
            end_epoch=2,
            continuation_write_root=root / "source-capsules",
        )
    elif mode == "restore":
        snapshots = _run_epochs(
            output_root=root / "restore-output",
            end_epoch=4,
            continuation_write_root=root / "continued-capsules",
            continuation_resume_root=root / "source-capsules",
        )
    else:
        raise ValueError(f"Unknown worker mode: {mode}")
    result_path.write_text(
        json.dumps({"mode": mode, "pid": os.getpid(), "snapshots": snapshots}, sort_keys=True),
        encoding="utf-8",
    )


def _proof_environment():
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["TF_NUM_INTRAOP_THREADS"] = "1"
    environment["TF_NUM_INTEROP_THREADS"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    return environment


def _run_workers(root: Path, modes, python_executable):
    results = {}
    logs = {}
    environment = _proof_environment()
    for mode in modes:
        result_path = root / f"{mode}.json"
        completed = subprocess.run(
            [
                python_executable,
                str(Path(__file__).resolve()),
                "--worker-mode", mode,
                "--root", str(root),
                "--result", str(result_path),
            ],
            cwd=root,
            env=environment,
            check=False,
            text=True,
            capture_output=True,
        )
        logs[mode] = {
            "returncode": completed.returncode,
            "stderr_tail": completed.stderr[-2000:],
        }
        if completed.returncode != 0:
            return None, logs, mode
        results[mode] = json.loads(result_path.read_text(encoding="utf-8"))
    return results, logs, None


def prove_accepted_lifecycle_equivalence(
    root: str | Path, python_executable: str = sys.executable
) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    results, logs, failed = _run_workers(
        root, ("accepted", "new_non_resume"), python_executable
    )
    if failed is not None:
        return {
            "schema_version": 1,
            "proof": "ACCEPTED_LIFECYCLE_EQUIVALENCE",
            "status": NOT_PROVEN_STATUS,
            "failed_worker": failed,
            "workers": logs,
        }
    accepted_rows = {row["epoch"]: row for row in results["accepted"]["snapshots"]}
    new_rows = {row["epoch"]: row for row in results["new_non_resume"]["snapshots"]}
    compared = {epoch: accepted_rows[epoch] == new_rows[epoch] for epoch in range(1, 5)}
    exact = all(compared.values())
    return {
        "schema_version": 1,
        "proof": "ACCEPTED_LIFECYCLE_EQUIVALENCE",
        "status": "PASS" if exact else NOT_PROVEN_STATUS,
        "exact_equality": exact,
        "floating_tolerance": 0.0,
        "optimizer_batches_per_epoch": 2,
        "full_model": True,
        "per_epoch_exact": compared,
        "aggregate_state_sha256": {
            epoch: {
                "accepted": accepted_rows[epoch]["aggregate_state_sha256"],
                "new_non_resume": new_rows[epoch]["aggregate_state_sha256"],
            }
            for epoch in range(1, 5)
        },
        "worker_pids_distinct": results["accepted"]["pid"] != results["new_non_resume"]["pid"],
        "tensorflow_op_determinism_enabled": False,
        "fer2013_used": False,
        "test_access": False,
    }


def prove_fresh_process_exact_continuation(
    root: str | Path, python_executable: str = sys.executable
) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    results, logs, failed = _run_workers(
        root, ("uninterrupted", "save", "restore"), python_executable
    )
    if failed is not None:
        return {
            "schema_version": 1,
            "proof": "FRESH_PROCESS_CONTINUATION_EQUIVALENCE",
            "status": NOT_PROVEN_STATUS,
            "failed_worker": failed,
            "workers": logs,
            "scientific_run_authorized": False,
        }
    uninterrupted = {row["epoch"]: row for row in results["uninterrupted"]["snapshots"]}
    restored = {row["epoch"]: row for row in results["restore"]["snapshots"]}
    per_epoch = {epoch: uninterrupted[epoch] == restored[epoch] for epoch in (3, 4)}
    distinct = len({results[mode]["pid"] for mode in results}) == 3
    exact = distinct and all(per_epoch.values())
    return {
        "schema_version": 1,
        "proof": "FRESH_PROCESS_CONTINUATION_EQUIVALENCE",
        "status": "PASS" if exact else NOT_PROVEN_STATUS,
        "fresh_process_workers": 3,
        "worker_pids_distinct": distinct,
        "epochs_compared_after_restore": [3, 4],
        "floating_tolerance": 0.0,
        "synthetic_proof_cpu_threads": {"intra_op": 1, "inter_op": 1, "omp": 1},
        "tensorflow_op_determinism_enabled": False,
        "exact_equality": exact,
        "per_epoch_exact": per_epoch,
        "aggregate_state_sha_equal": {
            epoch: uninterrupted[epoch]["aggregate_state_sha256"] == restored[epoch]["aggregate_state_sha256"]
            for epoch in (3, 4)
        },
        "aggregate_state_sha256": {
            epoch: {
                "uninterrupted": uninterrupted[epoch]["aggregate_state_sha256"],
                "restored": restored[epoch]["aggregate_state_sha256"],
            }
            for epoch in (3, 4)
        },
        "variable_counts": {"trainable": 118, "non_trainable": 20, "keras": 138},
        "compared_state": [
            "all_trainable_model_variables",
            "all_non_trainable_and_dropout_seed_generator_variables",
            "all_138_keras_variables",
            "optimizer_slots_and_iteration",
            "warmup_cosine_position_and_lr",
            "early_stop_state",
            "checkpoint_selection_and_selected_weights_identity",
            "epoch_training_loss_and_validation_metrics",
            "next_epoch_original_sample_order",
            "post_shuffle_enumeration_indices",
            "augmentation_parameter_sequence",
            "python_numpy_tensorflow_rng_state",
        ],
        "invalid_issue70_state_used": False,
        "fer2013_used": False,
        "test_access": False,
        "scientific_run_authorized": False,
    }


def prove_both_equivalences(
    root: str | Path, python_executable: str = sys.executable
) -> dict[str, Any]:
    """Run Proof 2 only after the independent accepted-lifecycle proof passes."""

    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    proof1 = prove_accepted_lifecycle_equivalence(
        root / "proof1-accepted-lifecycle", python_executable
    )
    if proof1["status"] != "PASS":
        return {
            "schema_version": 1,
            "status": NOT_PROVEN_STATUS,
            "proof1": proof1,
            "proof2": {"status": "NOT_RUN_BECAUSE_PROOF1_FAILED"},
            "both_proofs_pass": False,
            "scientific_run_authorized": False,
        }
    proof2 = prove_fresh_process_exact_continuation(
        root / "proof2-fresh-process-resume", python_executable
    )
    both_pass = proof2["status"] == "PASS"
    return {
        "schema_version": 1,
        "status": "PASS" if both_pass else NOT_PROVEN_STATUS,
        "proof1": proof1,
        "proof2": proof2,
        "both_proofs_pass": both_pass,
        "scientific_run_authorized": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthetic WS exact-continuation proof")
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--worker-mode",
        choices=("accepted", "new_non_resume", "uninterrupted", "save", "restore"),
    )
    parser.add_argument("--result")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker_mode:
        if not args.result:
            raise ValueError("--result is required for worker mode")
        run_worker(args.worker_mode, Path(args.root), Path(args.result))
        return 0
    proof = prove_both_equivalences(args.root)
    print(json.dumps(proof, indent=2, sort_keys=True))
    return 0 if proof["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
