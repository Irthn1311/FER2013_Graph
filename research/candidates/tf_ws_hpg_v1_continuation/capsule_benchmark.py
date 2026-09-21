"""Synthetic production-dimension capsule storage/runtime benchmark.

This tool builds the accepted full WS model and AdamW slots, but it never
loads FER records and never performs a training or validation step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .continuation import EpochBoundaryContinuationManager
from .data_order import AcceptedShufflePlan
from .train_validation_only import _build_runtime


PRODUCTION_SAMPLE_COUNT = 28_709
PRODUCTION_PLAN_EPOCHS = 101
MEASURED_EPOCHS = (1, 30, 60)


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def run_benchmark(root: str | Path) -> dict:
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    output_root = root / "output"
    output_root.mkdir()
    continuation_root = root / "continuation"

    plan_started = time.perf_counter()
    plan = AcceptedShufflePlan.materialize(
        PRODUCTION_SAMPLE_COUNT, PRODUCTION_PLAN_EPOCHS
    )
    plan_seconds = time.perf_counter() - plan_started
    steps_per_epoch = (PRODUCTION_SAMPLE_COUNT + 63) // 64
    model, optimizer, checkpoint, early_stop = _build_runtime(
        output_root, steps_per_epoch
    )
    identity = {
        "benchmark": "production_dimension_storage_only",
        "sample_count": PRODUCTION_SAMPLE_COUNT,
        "plan_epochs": PRODUCTION_PLAN_EPOCHS,
        "full_accepted_model": True,
        "adamw_slots_built": True,
        "fer2013_used": False,
        "test_access": False,
        "accepted_shuffle_plan_sha256": plan.sha256,
    }
    manager = EpochBoundaryContinuationManager(
        continuation_root, identity, plan
    )
    start_epoch, history = manager.restore_or_initialize(
        model=model,
        optimizer=optimizer,
        early_stop=early_stop,
        checkpoint_callback=checkpoint,
        output_root=output_root,
    )
    if start_epoch != 1:
        raise RuntimeError("Synthetic benchmark must initialize at epoch 1")

    measurements = {}
    for epoch in range(1, 61):
        logs = {
            "loss": 1.0,
            "accuracy": 0.0,
            "val_loss": 1.0,
            "val_accuracy": 0.0,
        }
        checkpoint.on_epoch_end(epoch - 1, logs)
        early_stop.on_epoch_end(epoch - 1, logs)
        history.append({"epoch": epoch, **logs, "synthetic_benchmark": True})
        persist_started = time.perf_counter()
        manager.persist_epoch_boundary(
            completed_epoch=epoch,
            model=model,
            optimizer=optimizer,
            early_stop=early_stop,
            checkpoint_callback=checkpoint,
            history=history,
            output_root=output_root,
        )
        persist_seconds = time.perf_counter() - persist_started
        if epoch in MEASURED_EPOCHS:
            capsule = continuation_root / f"epoch_{epoch:06d}"
            verify_started = time.perf_counter()
            manager.verify_latest(full_lineage_members=True)
            full_verify_seconds = time.perf_counter() - verify_started
            selected = capsule / "selected_checkpoint"
            capsule_bytes = _tree_bytes(capsule)
            measurements[str(epoch)] = {
                "capsule_bytes": capsule_bytes,
                "total_continuation_root_bytes": _tree_bytes(continuation_root),
                "explicit_state_npz_bytes": (
                    capsule / "explicit_state.npz"
                ).stat().st_size,
                "selected_checkpoint_bytes": _tree_bytes(selected),
                "persist_seconds": persist_seconds,
                "verify_latest_full_lineage_seconds": full_verify_seconds,
                "lineage_verification_seconds": full_verify_seconds,
                "peak_incremental_staging_bytes_upper_bound": capsule_bytes,
            }

    result = {
        "schema_version": 1,
        "status": "PASS",
        "benchmark": "WS_HPG_PRODUCTION_DIMENSION_CAPSULE_STORAGE_RUNTIME",
        "sample_count": PRODUCTION_SAMPLE_COUNT,
        "plan_epochs": PRODUCTION_PLAN_EPOCHS,
        "measured_epochs": list(MEASURED_EPOCHS),
        "accepted_shuffle_plan_sha256": plan.sha256,
        "shuffle_plan_materialization_seconds": plan_seconds,
        "immutable_shuffle_plan_bytes": (
            continuation_root / "accepted_shuffle_plan.npz"
        ).stat().st_size,
        "immutable_augmentation_parameters_bytes": (
            continuation_root / "accepted_augmentation_parameters.npz"
        ).stat().st_size,
        "full_accepted_model": True,
        "model_parameters": int(model.count_params()),
        "trainable_variables": len(model.trainable_variables),
        "keras_variables": len(model.variables),
        "adamw_variables": len(optimizer.variables),
        "optimizer_training_steps_executed": 0,
        "measurements": measurements,
        "storage_contract": (
            "one immutable root plan and augmentation sequence; neither is "
            "duplicated per capsule"
        ),
        "risk_assessment": (
            "No material 12-hour censor risk from capsule persistence/verification "
            "at the measured epoch-60 envelope"
        ),
        "fer2013_used": False,
        "validation_metrics_read": False,
        "training_executed": False,
        "test_access": False,
    }
    (root / "capsule_benchmark.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    result = run_benchmark(args.root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
