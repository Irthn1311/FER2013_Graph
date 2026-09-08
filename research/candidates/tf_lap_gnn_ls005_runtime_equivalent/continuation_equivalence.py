"""Fresh-process exact continuation equivalence for the LS0.05 lifecycle."""

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
FROZEN_PACKAGE_SRC = (
    REPOSITORY_ROOT / "standalone" / "lap_gnn_tensorflow_ofix7_mid_candidate" / "src"
)
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(FROZEN_PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(FROZEN_PACKAGE_SRC))

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from lap_gnn_tf.training.early_stopping import ValidationLossEarlyStopping  # noqa: E402
from lap_gnn_tf.training.optimizer import TorchCompatibleAdamW  # noqa: E402
from lap_gnn_tf.training.plateau import TorchCompatibleReduceLROnPlateau  # noqa: E402
from research.candidates.tf_lap_gnn_ls005_rescue.loss_adapter import (  # noqa: E402
    smoothed_sparse_cross_entropy,
)
from research.candidates.tf_lap_gnn_ls005_runtime_equivalent.continuation import (  # noqa: E402
    EpochBoundaryContinuationManager,
)
from research.candidates.tf_lap_gnn_ls005_runtime_equivalent.equivalence import (  # noqa: E402
    PRIOR_CORRUPTION_SEED,
    SEED,
    TinyLifecycleModel,
    _arrays_sha256,
    _augmentation_value,
    _dataset,
    _epoch_order,
    _metrics,
    _next_epoch_data_state,
)


SYNTHETIC_IDENTITY = {
    "contract": "ls005_epoch_boundary_continuation_v1",
    "seed": SEED,
    "label_smoothing": 0.05,
    "issue60_artifacts_used": False,
}
SYNTHETIC_CONFIG = {
    "graph": {"prior_corruption": {"enabled": True, "seed": PRIOR_CORRUPTION_SEED}}
}


class TinyDataset:
    def __init__(self, size: int) -> None:
        self.size = int(size)
        self.epoch = 0

    def __len__(self) -> int:
        return self.size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


class TinyGenerator:
    def __init__(self) -> None:
        self.dataset = TinyDataset(12)
        self.split = "train"
        self.batch_size = 4
        self.seed = SEED
        self.shuffle = True
        self.cache_size = 0
        self.graph_workers = 1

    def _order(self, epoch: int) -> np.ndarray:
        return _epoch_order(epoch)


class TinyCheckpointPolicy:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_macro = -float("inf")
        self.best_accuracy = -float("inf")
        self.best_macro_epoch = -1
        self.best_accuracy_epoch = -1

    def update(self, model: tf.keras.Model, epoch: int, metrics: dict[str, float]) -> None:
        macro = float(metrics["macro_f1"])
        accuracy = float(metrics["accuracy"])
        if macro > self.best_macro:
            self.best_macro = macro
            self.best_macro_epoch = int(epoch)
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_accuracy_epoch = int(epoch)
            payload = _arrays_sha256(model.variables).encode("ascii")
            (self.checkpoint_dir / "best_val_accuracy.keras").write_bytes(payload)
            (self.checkpoint_dir / "best_val_accuracy.weights.h5").write_bytes(payload)
            (self.checkpoint_dir / "best_val_accuracy.metadata.json").write_text(
                json.dumps({"epoch": epoch, "validation_metrics": metrics}),
                encoding="utf-8",
            )

    def state(self) -> dict[str, Any]:
        return {
            "best_macro": self.best_macro,
            "best_accuracy": self.best_accuracy,
            "best_macro_epoch": self.best_macro_epoch,
            "best_accuracy_epoch": self.best_accuracy_epoch,
        }


def _rng_hashes() -> dict[str, str]:
    python_payload = repr(random.getstate()).encode("utf-8")
    numpy_state = np.random.get_state()
    numpy_payload = (
        numpy_state[0].encode("ascii")
        + np.asarray(numpy_state[1], dtype=np.uint32).tobytes(order="C")
        + repr(numpy_state[2:]).encode("ascii")
    )
    return {
        "python_rng_sha256": hashlib.sha256(python_payload).hexdigest(),
        "numpy_rng_sha256": hashlib.sha256(numpy_payload).hexdigest(),
    }


def _setup_runtime(output_dir: Path):
    tf.keras.backend.clear_session()
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    tf.keras.utils.set_random_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    model = TinyLifecycleModel()
    model(tf.zeros((1, 4), dtype=tf.float32), training=False)
    inner_optimizer = TorchCompatibleAdamW(
        learning_rate=3e-4,
        weight_decay=1e-3,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        global_clipnorm=5.0,
        name="adamw",
    )
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(inner_optimizer)
    optimizer.build(model.trainable_variables)
    scheduler = TorchCompatibleReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        threshold=1e-4,
        min_lr=3e-5,
    )
    early = ValidationLossEarlyStopping(min_epochs=30, patience=15)
    policy = TinyCheckpointPolicy(output_dir)
    generator = TinyGenerator()

    @tf.function(reduce_retracing=True)
    def train_step(features: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            logits = model(features, training=True)
            loss = smoothed_sparse_cross_entropy(labels, logits)
            scaled_loss = optimizer.scale_loss(loss)
        gradients = tape.gradient(scaled_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    return model, optimizer, scheduler, early, policy, generator, train_step


def _snapshot(
    epoch: int,
    model: tf.keras.Model,
    optimizer: Any,
    scheduler: Any,
    early: Any,
    policy: TinyCheckpointPolicy,
    validation: dict[str, float],
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "trainable_sha256": _arrays_sha256(model.trainable_variables),
        "non_trainable_sha256": _arrays_sha256(model.non_trainable_variables),
        "optimizer_sha256": _arrays_sha256(optimizer.variables),
        "optimizer_iterations": int(optimizer.iterations.numpy()),
        "learning_rate": float(optimizer.learning_rate.numpy()),
        "scheduler_state": scheduler.get_state(),
        "early_stopping_state": early.get_state(),
        "checkpoint_selection_state": policy.state(),
        "validation": validation,
        "next_epoch_data_state": _next_epoch_data_state(epoch),
        **_rng_hashes(),
    }


def _run_epochs(
    *,
    output_dir: Path,
    end_epoch: int,
    continuation_write_root: Path | None = None,
    continuation_resume_root: Path | None = None,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (
        model,
        optimizer,
        scheduler,
        early,
        policy,
        generator,
        train_step,
    ) = _setup_runtime(output_dir)
    history: list[dict[str, Any]] = []
    start_epoch = 1
    manager = None
    if continuation_write_root is not None:
        manager = EpochBoundaryContinuationManager(
            continuation_write_root,
            SYNTHETIC_IDENTITY,
            resume_root=continuation_resume_root,
        )
        start_epoch, history = manager.restore_or_initialize(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            early=early,
            policy=policy,
            train_data=generator,
            config=SYNTHETIC_CONFIG,
            output_dir=output_dir,
        )
    train_x, train_y, val_x, val_y = _dataset()
    snapshots: list[dict[str, Any]] = []
    for epoch in range(start_epoch, end_epoch + 1):
        generator.dataset.set_epoch(epoch)
        order = generator._order(epoch)
        for start in range(0, len(order), 4):
            indices = order[start : start + 4]
            augmented = np.stack(
                [
                    train_x[index] + _augmentation_value(epoch, int(index))
                    for index in indices
                ]
            )
            train_step(tf.constant(augmented), tf.constant(train_y[indices]))
        # Exercise and therefore prove restoration of both process RNG streams.
        random.random()
        np.random.random()
        validation = _metrics(model, val_x, val_y)
        validation["macro_f1"] = validation["accuracy"]
        early.update(epoch, validation["loss"])
        policy.update(model, epoch, validation)
        scheduler.step(validation["loss"])
        history.append({"epoch": epoch, **validation})
        snapshot = _snapshot(
            epoch, model, optimizer, scheduler, early, policy, validation
        )
        snapshots.append(snapshot)
        if manager is not None:
            manager.persist_epoch_boundary(
                completed_epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                early=early,
                policy=policy,
                train_data=generator,
                config=SYNTHETIC_CONFIG,
                history=history,
                output_dir=output_dir,
            )
    return snapshots


def run_worker(mode: str, root: Path, result_path: Path) -> None:
    if mode == "uninterrupted":
        snapshots = _run_epochs(output_dir=root / "uninterrupted", end_epoch=4)
    elif mode == "save":
        snapshots = _run_epochs(
            output_dir=root / "save-output",
            end_epoch=2,
            continuation_write_root=root / "source-capsules",
        )
    elif mode == "restore":
        snapshots = _run_epochs(
            output_dir=root / "restore-output",
            end_epoch=4,
            continuation_write_root=root / "continued-capsules",
            continuation_resume_root=root / "source-capsules",
        )
    else:
        raise ValueError(f"Unknown worker mode: {mode}")
    result_path.write_text(
        json.dumps({"mode": mode, "snapshots": snapshots}, sort_keys=True),
        encoding="utf-8",
    )


def prove_fresh_process_exact_continuation(
    root: str | Path, python_executable: str = sys.executable
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    results = {}
    for mode in ("uninterrupted", "save", "restore"):
        result_path = root_path / f"{mode}.json"
        completed = subprocess.run(
            [
                python_executable,
                str(script),
                "--worker-mode",
                mode,
                "--root",
                str(root_path),
                "--result",
                str(result_path),
            ],
            cwd=root_path,
            env=environment,
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            return {
                "status": "LAP_LS005_EXACT_RESUME_NOT_PROVEN",
                "failed_worker": mode,
                "returncode": completed.returncode,
                "stderr": completed.stderr[-4000:],
            }
        results[mode] = json.loads(result_path.read_text(encoding="utf-8"))
    uninterrupted = {
        row["epoch"]: row for row in results["uninterrupted"]["snapshots"]
    }
    restored = {row["epoch"]: row for row in results["restore"]["snapshots"]}
    compared = {epoch: uninterrupted[epoch] == restored[epoch] for epoch in (3, 4)}
    exact = all(compared.values())
    return {
        "schema_version": 1,
        "status": "PASS" if exact else "LAP_LS005_EXACT_RESUME_NOT_PROVEN",
        "fresh_process_workers": 3,
        "restore_process_was_distinct": True,
        "epochs_compared_after_restore": [3, 4],
        "exact_equality": exact,
        "floating_tolerance": 0.0,
        "per_epoch_exact": compared,
        "compared_state": [
            "trainable_model_variables",
            "non_trainable_model_variables",
            "optimizer_slots_and_iteration",
            "learning_rate",
            "scheduler_state",
            "early_stop_state",
            "checkpoint_selection_state",
            "validation_metrics",
            "next_epoch_training_order",
            "next_epoch_augmentation_sequence",
            "python_rng_state",
            "numpy_rng_state",
        ],
        "restored_state_sha256": hashlib.sha256(
            json.dumps(restored, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
        "issue60_artifacts_used": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-mode", choices=("uninterrupted", "save", "restore"))
    parser.add_argument("--root", required=True)
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
    return 0 if proof.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
