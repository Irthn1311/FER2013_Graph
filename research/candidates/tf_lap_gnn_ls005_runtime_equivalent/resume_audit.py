"""Read-only audit of whether Issue #60 partial artifacts permit exact resume."""

from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path
from typing import Any
import zipfile

import h5py


RESUME_DISPOSITION = "ISSUE60_CHECKPOINT_NOT_AUTHORIZED_FOR_EXACT_RESUME"


class ResumeAuditError(RuntimeError):
    """Raised when the bounded partial-artifact audit cannot be performed safely."""


def _reject_test_path(path: str | Path) -> None:
    parts = tuple(
        part.casefold()
        for part in os.fspath(path).replace("\\", "/").split("/")
        if part
    )
    if {"test", "testing", "test_split", "test-split"}.intersection(parts):
        raise ResumeAuditError("Resume audit must not inspect a test-specific path")


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResumeAuditError(f"Unreadable {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ResumeAuditError(f"{label} must be an object: {path}")
    return value


def _keras_optimizer_inventory(path: Path) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
            members = archive.namelist()
            weights_payload = archive.read("model.weights.h5")
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise ResumeAuditError(f"Unreadable partial Keras checkpoint: {path}") from exc
    optimizer_datasets: list[str] = []
    with h5py.File(BytesIO(weights_payload), "r") as handle:
        if "optimizer" in handle:
            handle["optimizer"].visititems(
                lambda name, value: optimizer_datasets.append(name)
                if isinstance(value, h5py.Dataset)
                else None
            )
    return {
        "keras_integrity": bad_member is None,
        "keras_member_count": len(members),
        "optimizer_dataset_count": len(optimizer_datasets),
        "optimizer_datasets_present": bool(optimizer_datasets),
    }


def audit_issue60_partial_resume(artifact_root: str | Path) -> dict[str, Any]:
    """Audit serialized state without loading or executing the partial checkpoint."""

    root = Path(artifact_root).resolve()
    _reject_test_path(root)
    checkpoint = root / "checkpoints" / "best_val_accuracy.keras"
    weights = root / "checkpoints" / "best_val_accuracy.weights.h5"
    metadata_path = root / "checkpoints" / "best_val_accuracy.metadata.json"
    history_path = root / "history.json"
    for path in (checkpoint, weights, metadata_path, history_path):
        if not path.is_file():
            raise ResumeAuditError(f"Required partial artifact missing: {path}")
    metadata = _json_object(metadata_path, "checkpoint metadata")
    history_payload = _json_object(history_path, "partial history")
    history = history_payload.get("epochs")
    if not isinstance(history, list) or not history:
        raise ResumeAuditError("Partial history has no completed epochs")
    optimizer = metadata.get("optimizer_state")
    scheduler = metadata.get("scheduler_state")
    early = metadata.get("early_stopping_state")
    checkpoint_inventory = _keras_optimizer_inventory(checkpoint)
    checkpoint_epoch = metadata.get("epoch")
    last_completed_epoch = history[-1].get("epoch")
    reasons: list[str] = []
    if not checkpoint_inventory["optimizer_datasets_present"] or not isinstance(optimizer, dict):
        reasons.append("optimizer_variables_or_iteration_not_proven")
    if not isinstance(scheduler, dict):
        reasons.append("scheduler_state_absent")
    else:
        reasons.append(
            "checkpoint_scheduler_state_was_serialized_before_the_epoch_scheduler_step"
        )
    if not isinstance(early, dict):
        reasons.append("early_stopping_state_absent")
    if checkpoint_epoch != last_completed_epoch:
        reasons.append("checkpoint_epoch_is_not_latest_completed_epoch")
    reasons.extend(
        [
            "data_generator_runtime_state_not_serialized",
            "global_tensorflow_rng_state_not_serialized",
            "augmentation_rng_state_not_serialized_as_continuation_state",
            "partial_epoch_model_optimizer_state_not_serialized",
        ]
    )
    return {
        "schema_version": 1,
        "status": RESUME_DISPOSITION,
        "checkpoint_loaded": False,
        "checkpoint_executed": False,
        "model_weights_present": checkpoint.is_file() and weights.is_file(),
        "optimizer_state": {
            "metadata": optimizer,
            **checkpoint_inventory,
        },
        "scheduler_state": scheduler,
        "early_stopping_state": early,
        "checkpoint_epoch": checkpoint_epoch,
        "last_completed_epoch": last_completed_epoch,
        "completed_history_rows": len(history),
        "data_generator_state_present": False,
        "augmentation_rng_state_present": False,
        "global_tensorflow_rng_state_present": False,
        "exact_resume_proven": False,
        "reasons": reasons,
        "scientific_interpretation": None,
        "test_access": False,
    }
