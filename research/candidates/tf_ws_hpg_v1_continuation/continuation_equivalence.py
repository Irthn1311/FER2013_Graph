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
    checkpoint_state,
    early_stopping_state,
)
from research.candidates.tf_ws_hpg_v1_continuation.data_order import (  # noqa: E402
    AcceptedShufflePlan,
    build_epoch_training_dataset,
    next_epoch_stream,
)
from research.candidates.tf_ws_hpg_v1_continuation.train_validation_only import (  # noqa: E402
    _build_runtime,
    _history_row,
    _load_persisted_order_plan,
    _variables_sha256,
)
from research.candidates.tf_ws_hpg_v1_training.data import build_dataset  # noqa: E402


SYNTHETIC_SAMPLES = 4
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
    labels = np.asarray([0, 1, 2, 3], dtype=np.int32)
    return images, supports, labels


def _snapshot(epoch, model, optimizer, early_stop, checkpoint, row, plan):
    stream = next_epoch_stream(plan, epoch)
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
        "checkpoint_selection_state": checkpoint_state(checkpoint),
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
    model, optimizer, checkpoint, early_stop = _build_runtime(output_root, steps_per_epoch=1)
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
    snapshots = []
    for epoch in range(start_epoch, end_epoch + 1):
        train = build_epoch_training_dataset(
            images, supports, labels, plan=plan, epoch_one_based=epoch, batch_size=4
        )
        one_epoch = model.fit(
            train,
            validation_data=validation,
            validation_freq=1,
            initial_epoch=epoch - 1,
            epochs=epoch,
            callbacks=[],
            verbose=0,
        )
        row = _history_row(epoch, one_epoch.history)
        checkpoint.on_epoch_end(epoch - 1, row)
        early_stop.on_epoch_end(epoch - 1, row)
        history.append(row)
        snapshots.append(_snapshot(epoch, model, optimizer, early_stop, checkpoint, row, plan))
        if manager is not None:
            manager.persist_epoch_boundary(
                completed_epoch=epoch,
                model=model,
                optimizer=optimizer,
                early_stop=early_stop,
                checkpoint_callback=checkpoint,
                history=history,
                output_root=output_root,
            )
    return snapshots


def run_worker(mode: str, root: Path, result_path: Path) -> None:
    if mode == "uninterrupted":
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


def prove_fresh_process_exact_continuation(
    root: str | Path, python_executable: str = sys.executable
) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    # The golden proof isolates continuation state from CPU thread scheduling.
    # This does not call enable_op_determinism and is not used by the future T4
    # scientific CLI; production resource/training semantics remain untouched.
    environment["TF_NUM_INTRAOP_THREADS"] = "1"
    environment["TF_NUM_INTEROP_THREADS"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    results = {}
    logs = {}
    for mode in ("uninterrupted", "save", "restore"):
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
        logs[mode] = {"returncode": completed.returncode, "stderr_tail": completed.stderr[-2000:]}
        if completed.returncode != 0:
            return {
                "schema_version": 1,
                "status": NOT_PROVEN_STATUS,
                "failed_worker": mode,
                "workers": logs,
                "scientific_run_authorized": False,
            }
        results[mode] = json.loads(result_path.read_text(encoding="utf-8"))
    uninterrupted = {row["epoch"]: row for row in results["uninterrupted"]["snapshots"]}
    restored = {row["epoch"]: row for row in results["restore"]["snapshots"]}
    per_epoch = {epoch: uninterrupted[epoch] == restored[epoch] for epoch in (3, 4)}
    distinct = len({results[mode]["pid"] for mode in results}) == 3
    exact = distinct and all(per_epoch.values())
    return {
        "schema_version": 1,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthetic WS exact-continuation proof")
    parser.add_argument("--root", required=True)
    parser.add_argument("--worker-mode", choices=("uninterrupted", "save", "restore"))
    parser.add_argument("--result")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker_mode:
        if not args.result:
            raise ValueError("--result is required for worker mode")
        run_worker(args.worker_mode, Path(args.root), Path(args.result))
        return 0
    proof = prove_fresh_process_exact_continuation(args.root)
    print(json.dumps(proof, indent=2, sort_keys=True))
    return 0 if proof["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
