"""Checkpoint-conditioned validation-only continuation for TF Step 12C.

This external harness resumes the reviewed Step-12 candidate from its exact
epoch-30 Keras checkpoint after the original Kaggle process was hard-censored.
It intentionally has no final-test lifecycle.  The resulting trajectory is a
checkpoint-conditioned restart, not a bitwise-uninterrupted seed42 run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Callable, Sequence
import zipfile

import numpy as np
import tensorflow as tf
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
FROZEN_PACKAGE_ROOT = (
    REPOSITORY_ROOT / "standalone" / "lap_gnn_tensorflow_ofix7_mid_candidate"
)
FROZEN_PACKAGE_SRC = FROZEN_PACKAGE_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(FROZEN_PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(FROZEN_PACKAGE_SRC))

from lap_gnn_tf.compat import keras_version, save_model_with_optimizer  # noqa: E402
from lap_gnn_tf.config import (  # noqa: E402
    canonical_config_hash,
    load_config,
    validate_locked_config,
)
from lap_gnn_tf.data.graph_generator import GraphBatchGenerator  # noqa: E402
from lap_gnn_tf.provenance import write_provenance  # noqa: E402
from lap_gnn_tf.resources import ResourceControls, RuntimeTelemetry  # noqa: E402
from lap_gnn_tf.seed import seed_everything  # noqa: E402
from lap_gnn_tf.signatures import scientific_payload_checksum  # noqa: E402
from lap_gnn_tf.training.artifacts import write_training_curves  # noqa: E402
from lap_gnn_tf.training.checkpointing import CheckpointPolicy  # noqa: E402
from lap_gnn_tf.training.early_stopping import (  # noqa: E402
    ValidationLossEarlyStopping,
)
from lap_gnn_tf.training.evaluator import (  # noqa: E402
    build_compiled_evaluation_step,
    evaluate_batches,
)
from lap_gnn_tf.training.execution import (  # noqa: E402
    MAX_REGISTERED_TRAIN_STEP_TRACES,
    validate_execution_config,
)
from lap_gnn_tf.training.plateau import (  # noqa: E402
    TorchCompatibleReduceLROnPlateau,
)
from lap_gnn_tf.training.trainer import (  # noqa: E402
    _atomic_history_csv,
    _print_epoch_summary,
)
from research.candidates.tf_learned_local_residual_slots.candidate_execution import (  # noqa: E402
    build_candidate_restricted_graph_train_step,
)
from research.candidates.tf_learned_local_residual_slots.model import (  # noqa: E402
    LearnedLocalResidualSlotLapGNN,
)


PROTOCOL_ID = "tf-step12c-checkpoint-conditioned-continuation-v1"
IMPLEMENTATION_BASE = "cc54ec045f2af0dad6aca4bf4b8b1710677ab1a4"
STOCHASTIC_CONTINUATION = "checkpoint_conditioned_restart"

EXPECTED_CANDIDATE_MODEL_SHA256 = (
    "0a7cfa315baf0d6666b7ab86139b328ab6d138985cfddd71180410675602fcca"
)
EXPECTED_CANDIDATE_EXECUTION_SHA256 = (
    "48c0e5f8ad4676e17fb4127b3a30ad053beedca8e04e05cfb6fb24f2bb9236f9"
)
EXPECTED_CANDIDATE_EXECUTION_CONTRACT_SHA256 = (
    "331570bacd3ec97474c85f25e7e3cb461ef42b0aa3f442caf3dd1f52314bcbc7"
)
EXPECTED_CANDIDATE_VALIDATION_HARNESS_SHA256 = (
    "1b0707c41f30a9a5b9b9dba3995030ac50fccc90cf439d1ac26a31a32a878f2f"
)
EXPECTED_SCIENTIFIC_PAYLOAD_SHA256 = (
    "286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e"
)
EXPECTED_BASELINE_EXECUTION_CONTRACT_SHA256 = (
    "14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22"
)
EXPECTED_SEED42_CONFIG_SHA256 = (
    "aa3bf2d3932bbad6c5f8cdcc347f4a9866e2c027d6135a60b5002a8f6a3b6908"
)

EXPECTED_SOURCE_ARCHIVE_SHA256 = (
    "2ada6cfd1ce1c07f6d7ae36264a1f14840a0936e9448a72e6bb464ae6ab71357"
)
SOURCE_MEMBER_SHA256 = {
    "run/checkpoints/best_val_accuracy.keras": (
        "818450d56cb480cf08637bee01061e8028a3d58c0f13346716618f0ee186d932"
    ),
    "run/checkpoints/best_val_accuracy.weights.h5": (
        "981b1864a5b997b092b128a0c863a9f8dee41105425fce25e63c94e1c165ed78"
    ),
    "run/checkpoints/best_val_accuracy.metadata.json": (
        "adf4fc95e36f85610a280056e1e518cffe71108b8160b852860058ae6708f9ce"
    ),
    "run/history.json": (
        "0a2edffbc595f09660e01ccacc5338656aef06892949aad4a9e209aac280789c"
    ),
    "run/resolved_config.json": (
        "3c028dd2f32ebed3a252544e170220b150b5e29920cea865924dddce6aef5a32"
    ),
}

SOURCE_CHECKPOINT_MEMBER = "run/checkpoints/best_val_accuracy.keras"
SOURCE_WEIGHTS_MEMBER = "run/checkpoints/best_val_accuracy.weights.h5"
SOURCE_METADATA_MEMBER = "run/checkpoints/best_val_accuracy.metadata.json"
SOURCE_HISTORY_MEMBER = "run/history.json"
SOURCE_CONFIG_MEMBER = "run/resolved_config.json"

RESUME_EPOCH = 30
FIRST_CONTINUATION_EPOCH = 31
EXPECTED_MODEL_CLASS = "LearnedLocalResidualSlotLapGNN"
EXPECTED_PARAMETER_COUNT = 1_061_576
EXPECTED_TRAINABLE_VARIABLE_COUNT = 128
EXPECTED_Q_INDEX = 127
EXPECTED_Q_SHAPE = (4, 96)
EXPECTED_Q_DTYPE = "float32"
EXPECTED_Q_SHA256 = (
    "166f6e09191f94c52c17af81c2d9ba357c765b2077aab5fc809563a9de6d6270"
)
EXPECTED_OPTIMIZER_CLASS = "LossScaleOptimizer"
EXPECTED_OPTIMIZER_ITERATIONS = 53_822
EXPECTED_OPTIMIZER_VARIABLE_COUNT = 262
EXPECTED_OPTIMIZER_LR = 0.0001500000071246177
LR_ABS_TOLERANCE = 1e-12

EARLY_STATE_POST_E30 = {
    "best": 1.1009891497350373,
    "epochs_without_improvement": 1,
}
EARLY_MIN_EPOCHS = 30
EARLY_PATIENCE = 15
SCHEDULER_STATE_POST_E30 = {
    "best": 1.1009891497350373,
    "num_bad_epochs": 1,
    "cooldown_counter": 0,
    "last_epoch": 30,
}
CHECKPOINT_POLICY_POST_E30 = {
    "best_macro": 0.5634445160028113,
    "best_macro_epoch": 30,
    "best_accuracy": 0.603789356366676,
    "best_accuracy_epoch": 30,
}

PRETRAIN_VALIDATION_REFERENCE = {
    "sample_count": 3589,
    "accuracy": 0.603789356366676,
    "macro_f1": 0.5634445160028113,
    "loss": 1.1265364869505958,
}
PRETRAIN_VALIDATION_TOLERANCE = {
    "accuracy": 0.001,
    "macro_f1": 0.001,
    "loss": 0.005,
}
REGISTERED_RESOURCE_VALUES = {
    "intra_op_threads": 0,
    "inter_op_threads": 0,
    "graph_workers": 2,
    "tf_data_prefetch": 2,
    "tf_data_parallel_calls": 1,
    "graph_cache_size": 64,
    "memory_growth": True,
    "mixed_precision": True,
    "xla": False,
    "batch_size": 16,
    "eval_batch_size": 32,
    "device": "gpu",
}

COMPLETION_MARKER_NAME = "CHECKPOINT_CONTINUATION_VALIDATION_ONLY_COMPLETE.json"
FAILURE_MARKER_NAME = "CHECKPOINT_CONTINUATION_TECHNICAL_FAILURE.json"
OVERLAP_SOURCE_NAME = "FIRST_RUN_OVERLAP_DIAGNOSTICS.json"
LATEST_STATE_NAME = "latest_state.keras"
LATEST_STATE_METADATA_NAME = "latest_state.metadata.json"
PROGRESS_INTERVAL_BATCHES = 100
FORBIDDEN_ARTIFACT_NAMES = (
    "TRAINING_COMPLETE.json",
    "run_summary.json",
    "predictions.csv",
    "per_class_metrics.csv",
    "confusion_matrix.csv",
    "confusion_matrix.png",
)

CANDIDATE_ROOT = Path(__file__).resolve().parent
SOURCE_LOCK_PATHS = {
    "candidate_model": CANDIDATE_ROOT / "model.py",
    "candidate_execution_adapter": CANDIDATE_ROOT / "candidate_execution.py",
    "candidate_execution_contract": CANDIDATE_ROOT
    / "candidate_execution_contract.json",
    "candidate_validation_harness": CANDIDATE_ROOT / "train_validation_only.py",
}
EXPECTED_SOURCE_LOCKS = {
    "candidate_model": EXPECTED_CANDIDATE_MODEL_SHA256,
    "candidate_execution_adapter": EXPECTED_CANDIDATE_EXECUTION_SHA256,
    "candidate_execution_contract": EXPECTED_CANDIDATE_EXECUTION_CONTRACT_SHA256,
    "candidate_validation_harness": EXPECTED_CANDIDATE_VALIDATION_HARNESS_SHA256,
}


class CheckpointContinuationError(RuntimeError):
    """Raised whenever the registered continuation cannot be proven safe."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_object(path: str | Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointContinuationError(f"Unreadable {label}: {path}") from exc
    if not isinstance(value, dict):
        raise CheckpointContinuationError(f"{label} must be a JSON object: {path}")
    return value


def _atomic_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="",
    )
    os.replace(temporary, path)


def _atomic_copy(source: str | Path, destination: str | Path) -> None:
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with source.open("rb") as input_stream, temporary.open("wb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream)
        output_stream.flush()
        os.fsync(output_stream.fileno())
    if sha256_file(temporary) != sha256_file(source):
        temporary.unlink(missing_ok=True)
        raise CheckpointContinuationError(f"Atomic copy verification failed: {source}")
    os.replace(temporary, destination)


def verify_source_locks() -> dict[str, str]:
    actual: dict[str, str] = {}
    for name, path in SOURCE_LOCK_PATHS.items():
        if not path.is_file():
            raise CheckpointContinuationError(f"Required locked source missing: {path}")
        actual[name] = sha256_file(path)
    drift = {
        key: {"expected": EXPECTED_SOURCE_LOCKS[key], "actual": value}
        for key, value in actual.items()
        if value != EXPECTED_SOURCE_LOCKS[key]
    }
    if drift:
        raise CheckpointContinuationError(f"Reviewed candidate source drift: {drift}")
    if scientific_payload_checksum(FROZEN_PACKAGE_ROOT) != EXPECTED_SCIENTIFIC_PAYLOAD_SHA256:
        raise CheckpointContinuationError("Frozen scientific payload drift")
    actual["checkpoint_continuation_harness"] = sha256_file(Path(__file__).resolve())
    return actual


def verify_seed42_config(config_path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file() or sha256_file(path) != EXPECTED_SEED42_CONFIG_SHA256:
        raise CheckpointContinuationError("Seed42 input config SHA drift")
    config = load_config(path)
    validate_locked_config(config)
    locked = config.get("locked", {})
    training = config.get("training", {})
    required = {
        "seed": (config.get("seed"), 42),
        "training seed": (training.get("seed"), 42),
        "payload": (
            locked.get("package_checksum"),
            EXPECTED_SCIENTIFIC_PAYLOAD_SHA256,
        ),
        "execution contract": (
            locked.get("execution_contract_sha256"),
            EXPECTED_BASELINE_EXECUTION_CONTRACT_SHA256,
        ),
        "optimizer mode": (
            training.get("optimizer_execution_mode"),
            "restricted_tf_function",
        ),
        "Grappler profile": (training.get("grappler_profile"), "G1-A"),
        "mixed precision": (training.get("amp"), True),
        "max epochs": (training.get("max_epochs"), 90),
    }
    drift = {
        key: {"actual": actual, "expected": expected}
        for key, (actual, expected) in required.items()
        if actual != expected
    }
    if drift:
        raise CheckpointContinuationError(f"Registered seed42 config drift: {drift}")
    return path, config


def verify_resource_controls(controls: ResourceControls) -> None:
    drift = {
        key: {"actual": getattr(controls, key), "expected": expected}
        for key, expected in REGISTERED_RESOURCE_VALUES.items()
        if getattr(controls, key) != expected
    }
    if drift:
        raise CheckpointContinuationError(f"Registered continuation resource drift: {drift}")


def verify_and_extract_source_archive(
    archive_path: str | Path, extraction_root: str | Path
) -> dict[str, Path]:
    """Verify the exact external archive and extract only registered members."""

    archive_path = Path(archive_path).expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"Censored rolling archive missing: {archive_path}")
    if sha256_file(archive_path) != EXPECTED_SOURCE_ARCHIVE_SHA256:
        raise CheckpointContinuationError("Censored source archive SHA drift")
    extraction_root = Path(extraction_root)
    extracted: dict[str, Path] = {}
    with zipfile.ZipFile(archive_path, "r") as archive:
        inventory = set(archive.namelist())
        missing = sorted(set(SOURCE_MEMBER_SHA256) - inventory)
        if missing:
            raise CheckpointContinuationError(
                f"Censored source archive members missing: {missing}"
            )
        for member, expected_sha in SOURCE_MEMBER_SHA256.items():
            payload = archive.read(member)
            if _sha256_bytes(payload) != expected_sha:
                raise CheckpointContinuationError(
                    f"Censored source member SHA drift: {member}"
                )
            relative = Path(*member.split("/"))
            destination = extraction_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.tmp")
            temporary.write_bytes(payload)
            os.replace(temporary, destination)
            if sha256_file(destination) != expected_sha:
                raise CheckpointContinuationError(
                    f"Extracted source member verification failed: {member}"
                )
            extracted[member] = destination
    return extracted


def split_reviewed_history(
    history_payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    epochs = history_payload.get("epochs")
    if not isinstance(epochs, list) or len(epochs) < 32:
        raise CheckpointContinuationError(
            "Reviewed censored history must contain at least epochs 1..32"
        )
    if any(not isinstance(row, dict) for row in epochs):
        raise CheckpointContinuationError("Reviewed history rows must be objects")
    observed = [row.get("epoch") for row in epochs]
    if observed != list(range(1, len(epochs) + 1)):
        raise CheckpointContinuationError("Reviewed history epoch order is not contiguous")
    prefix = [dict(row) for row in epochs[:RESUME_EPOCH]]
    if [row["epoch"] for row in prefix] != list(range(1, RESUME_EPOCH + 1)):
        raise CheckpointContinuationError("Immutable scientific prefix is not epochs 1..30")
    overlap = {31: dict(epochs[30]), 32: dict(epochs[31])}
    return prefix, overlap


def load_reviewed_source(
    extracted: dict[str, Path],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any]]:
    history = _json_object(extracted[SOURCE_HISTORY_MEMBER], "source history")
    resolved_config = _json_object(extracted[SOURCE_CONFIG_MEMBER], "source config")
    prefix, overlap = split_reviewed_history(history)
    locked = resolved_config.get("locked", {})
    if locked.get("package_checksum") != EXPECTED_SCIENTIFIC_PAYLOAD_SHA256:
        raise CheckpointContinuationError("Source resolved config payload drift")
    if (
        locked.get("execution_contract_sha256")
        != EXPECTED_BASELINE_EXECUTION_CONTRACT_SHA256
    ):
        raise CheckpointContinuationError("Source resolved execution contract drift")
    if resolved_config.get("seed") != 42:
        raise CheckpointContinuationError("Source resolved seed drift")
    return prefix, overlap, resolved_config


def _q_digest(model: Any) -> str:
    slot_layer = getattr(model, "learned_local_residual_slots", None)
    q = getattr(slot_layer, "Q", None)
    if q is None:
        raise CheckpointContinuationError("Candidate Q is missing")
    values = np.asarray(q.numpy(), dtype=np.float32).reshape(-1)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _optimizer_lr(optimizer: Any) -> float:
    return float(tf.keras.backend.get_value(optimizer.learning_rate))


def optimizer_identity(optimizer: Any) -> dict[str, Any]:
    return {
        "class": type(optimizer).__name__,
        "iterations": int(optimizer.iterations.numpy()),
        "variable_count": len(optimizer.variables),
        "learning_rate": _optimizer_lr(optimizer),
    }


def model_identity(model: Any) -> dict[str, Any]:
    q = model.learned_local_residual_slots.Q
    return {
        "class": type(model).__name__,
        "parameter_count": int(model.count_params()),
        "trainable_variable_count": len(model.trainable_variables),
        "q_index": EXPECTED_Q_INDEX,
        "q_is_index_127": model.trainable_variables[EXPECTED_Q_INDEX] is q,
        "q_shape": list(q.shape),
        "q_dtype": str(q.dtype),
        "q_flat_float32_sha256": _q_digest(model),
    }


def verify_model_optimizer_identity(
    model: Any,
    *,
    expected_q_sha256: str | None = EXPECTED_Q_SHA256,
    expected_iterations: int | None = EXPECTED_OPTIMIZER_ITERATIONS,
    expected_optimizer_variable_count: int | None = EXPECTED_OPTIMIZER_VARIABLE_COUNT,
    expected_lr: float | None = EXPECTED_OPTIMIZER_LR,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = model_identity(model)
    expected_model = {
        "class": EXPECTED_MODEL_CLASS,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "trainable_variable_count": EXPECTED_TRAINABLE_VARIABLE_COUNT,
        "q_index": EXPECTED_Q_INDEX,
        "q_is_index_127": True,
        "q_shape": list(EXPECTED_Q_SHAPE),
        "q_dtype": EXPECTED_Q_DTYPE,
    }
    drift = {
        key: {"actual": identity.get(key), "expected": expected}
        for key, expected in expected_model.items()
        if identity.get(key) != expected
    }
    if expected_q_sha256 is not None and identity["q_flat_float32_sha256"] != expected_q_sha256:
        drift["q_flat_float32_sha256"] = {
            "actual": identity["q_flat_float32_sha256"],
            "expected": expected_q_sha256,
        }
    optimizer = getattr(model, "optimizer", None)
    if optimizer is None:
        raise CheckpointContinuationError(
            "Checkpoint did not restore its optimizer; fresh optimizer creation is forbidden"
        )
    optimizer_state = optimizer_identity(optimizer)
    if optimizer_state["class"] != EXPECTED_OPTIMIZER_CLASS:
        drift["optimizer_class"] = {
            "actual": optimizer_state["class"],
            "expected": EXPECTED_OPTIMIZER_CLASS,
        }
    if expected_iterations is not None and optimizer_state["iterations"] != expected_iterations:
        drift["optimizer_iterations"] = {
            "actual": optimizer_state["iterations"],
            "expected": expected_iterations,
        }
    if (
        expected_optimizer_variable_count is not None
        and optimizer_state["variable_count"] != expected_optimizer_variable_count
    ):
        drift["optimizer_variable_count"] = {
            "actual": optimizer_state["variable_count"],
            "expected": expected_optimizer_variable_count,
        }
    if expected_lr is not None and not np.isclose(
        optimizer_state["learning_rate"], expected_lr, rtol=0.0, atol=LR_ABS_TOLERANCE
    ):
        drift["optimizer_learning_rate"] = {
            "actual": optimizer_state["learning_rate"],
            "expected": expected_lr,
        }
    if drift:
        raise CheckpointContinuationError(f"Resume checkpoint identity drift: {drift}")
    return identity, optimizer_state


def load_resume_checkpoint(checkpoint_path: str | Path) -> Any:
    """Load the exact checkpoint with its serialized optimizer state."""

    checkpoint_path = Path(checkpoint_path)
    if sha256_file(checkpoint_path) != SOURCE_MEMBER_SHA256[SOURCE_CHECKPOINT_MEMBER]:
        raise CheckpointContinuationError("Resume checkpoint SHA drift")
    model = tf.keras.models.load_model(checkpoint_path, compile=True)
    verify_model_optimizer_identity(model)
    return model


def reconstruct_control_state(
    optimizer: Any, config: dict[str, Any], output_root: str | Path
) -> tuple[ValidationLossEarlyStopping, TorchCompatibleReduceLROnPlateau, CheckpointPolicy]:
    early_cfg = config["training"]["early_stopping"]
    if (
        int(early_cfg["min_epochs_before_stop"]) != EARLY_MIN_EPOCHS
        or int(early_cfg["patience"]) != EARLY_PATIENCE
    ):
        raise CheckpointContinuationError("Early-stopping configuration drift")
    early = ValidationLossEarlyStopping(
        min_epochs=EARLY_MIN_EPOCHS, patience=EARLY_PATIENCE
    )
    early.set_state(dict(EARLY_STATE_POST_E30))

    scheduler_cfg = config["training"]["scheduler"]
    scheduler = TorchCompatibleReduceLROnPlateau(
        optimizer,
        mode=scheduler_cfg["mode"],
        factor=scheduler_cfg["factor"],
        patience=scheduler_cfg["patience"],
        threshold=scheduler_cfg["threshold"],
        min_lr=scheduler_cfg["min_lr"],
    )
    # This is the actual post-step(e30) state.  Never replay step(e30).
    scheduler.set_state(dict(SCHEDULER_STATE_POST_E30))

    policy = CheckpointPolicy(output_root)
    for key, value in CHECKPOINT_POLICY_POST_E30.items():
        setattr(policy, key, value)
    if early.get_state() != EARLY_STATE_POST_E30:
        raise CheckpointContinuationError("Early-stopping state reconstruction failed")
    if scheduler.get_state() != SCHEDULER_STATE_POST_E30:
        raise CheckpointContinuationError("Scheduler state reconstruction failed")
    if checkpoint_policy_state(policy) != CHECKPOINT_POLICY_POST_E30:
        raise CheckpointContinuationError("Checkpoint-policy state reconstruction failed")
    if not np.isclose(scheduler.current_lr, EXPECTED_OPTIMIZER_LR, rtol=0.0, atol=LR_ABS_TOLERANCE):
        raise CheckpointContinuationError("Scheduler current LR reconstruction failed")
    return early, scheduler, policy


def checkpoint_policy_state(policy: CheckpointPolicy) -> dict[str, Any]:
    return {
        "best_macro": float(policy.best_macro),
        "best_macro_epoch": int(policy.best_macro_epoch),
        "best_accuracy": float(policy.best_accuracy),
        "best_accuracy_epoch": int(policy.best_accuracy_epoch),
    }


def validate_pretrain_validation_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    sample_count = int(sum(metrics.get("support_per_class", [])))
    deltas = {
        key: abs(float(metrics[key]) - float(PRETRAIN_VALIDATION_REFERENCE[key]))
        for key in PRETRAIN_VALIDATION_TOLERANCE
    }
    passed = sample_count == PRETRAIN_VALIDATION_REFERENCE["sample_count"] and all(
        deltas[key] <= PRETRAIN_VALIDATION_TOLERANCE[key]
        for key in PRETRAIN_VALIDATION_TOLERANCE
    )
    evidence = {
        "status": "PASS" if passed else "FAIL",
        "sample_count": sample_count,
        "references": PRETRAIN_VALIDATION_REFERENCE,
        "tolerances": PRETRAIN_VALIDATION_TOLERANCE,
        "observed": {
            key: float(metrics[key]) for key in ("accuracy", "macro_f1", "loss")
        },
        "absolute_differences": deltas,
        "optimizer_updates_before_gate": 0,
    }
    if not passed:
        raise CheckpointContinuationError(
            f"Pre-train epoch30 validation gate failed closed: {evidence}"
        )
    return evidence


def overlap_diagnostic(
    resumed_row: dict[str, Any], first_run_row: dict[str, Any]
) -> dict[str, Any]:
    epoch = int(resumed_row["epoch"])
    if epoch not in (31, 32) or int(first_run_row.get("epoch", -1)) != epoch:
        raise CheckpointContinuationError("Overlap diagnostic epoch mismatch")
    fields = {
        "val_accuracy": "delta_val_accuracy",
        "val_macro_f1": "delta_val_macro_f1",
        "val_loss": "delta_val_loss",
        "train_macro_f1": "delta_train_macro_f1",
        "lr": "delta_lr",
        "early_stopping_wait": "delta_early_stopping_wait",
    }
    deltas: dict[str, float | None] = {}
    for source_key, delta_key in fields.items():
        resumed = resumed_row.get(source_key)
        original = first_run_row.get(source_key)
        deltas[delta_key] = (
            None if resumed is None or original is None else float(resumed) - float(original)
        )
    return {
        "schema_version": 1,
        "classification": "FIRST_RUN_OVERLAP_DIAGNOSTICS",
        "epoch": epoch,
        "descriptive_only": True,
        "affects_training": False,
        "affects_stopping": False,
        "affects_scheduler": False,
        "affects_checkpoint_selection": False,
        "affects_primary_endpoint": False,
        "triggers_retry": False,
        "first_run_row": first_run_row,
        "resumed_row": resumed_row,
        **deltas,
    }


def initialize_output(
    output_root: str | Path,
    extracted: dict[str, Path],
    prefix: list[dict[str, Any]],
    overlap: dict[int, dict[str, Any]],
    source_hashes: dict[str, str],
    input_config_path: Path,
) -> Path:
    output_root = Path(output_root).expanduser().resolve()
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise FileExistsError(f"Fresh continuation output must be absent or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    copy_map = {
        SOURCE_CHECKPOINT_MEMBER: checkpoint_dir / "best_val_accuracy.keras",
        SOURCE_WEIGHTS_MEMBER: checkpoint_dir / "best_val_accuracy.weights.h5",
        SOURCE_METADATA_MEMBER: checkpoint_dir / "best_val_accuracy.metadata.json",
    }
    for member, destination in copy_map.items():
        _atomic_copy(extracted[member], destination)
        if sha256_file(destination) != SOURCE_MEMBER_SHA256[member]:
            raise CheckpointContinuationError(f"Initial epoch30 checkpoint copy drift: {member}")
    _atomic_json(output_root / "history.json", {"epochs": prefix})
    _atomic_history_csv(output_root / "train_log.csv", prefix)
    _atomic_json(
        output_root / OVERLAP_SOURCE_NAME,
        {
            "schema_version": 1,
            "classification": "FIRST_RUN_OVERLAP_DIAGNOSTICS",
            "descriptive_only": True,
            "excluded_from_combined_scientific_history": True,
            "rows": {str(epoch): row for epoch, row in overlap.items()},
        },
    )
    _atomic_copy(extracted[SOURCE_CONFIG_MEMBER], output_root / "source_resolved_config.json")
    _atomic_json(
        output_root / "continuation_pre_run_manifest.json",
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "implementation_base": IMPLEMENTATION_BASE,
            "stochastic_continuation": STOCHASTIC_CONTINUATION,
            "source_archive_sha256": EXPECTED_SOURCE_ARCHIVE_SHA256,
            "source_member_sha256": SOURCE_MEMBER_SHA256,
            "source_checkpoint_epoch": RESUME_EPOCH,
            "first_continuation_epoch": FIRST_CONTINUATION_EPOCH,
            "input_config_path": str(input_config_path),
            "input_config_sha256": EXPECTED_SEED42_CONFIG_SHA256,
            "source_code_sha256": source_hashes,
            "scientific_prefix_epochs": [1, 30],
            "first_run_overlap_epochs": [31, 32],
            "first_run_overlap_is_diagnostic_only": True,
            "original_step12_scientific_result_valid": False,
            "original_step12_scientific_interpretation": None,
            "scientific_result_valid": False,
            "scientific_interpretation": None,
            "training_started": False,
            "test_access": False,
        },
    )
    return output_root


def _latest_state_metadata(
    *,
    model: Any,
    model_sha256: str,
    completed_epoch: int,
    scheduler: TorchCompatibleReduceLROnPlateau,
    early: ValidationLossEarlyStopping,
    policy: CheckpointPolicy,
    history_path: Path,
    output_root: Path,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    identity = model_identity(model)
    optimizer_state = optimizer_identity(model.optimizer)
    return {
        "schema_version": 1,
        "continuation_protocol_id": PROTOCOL_ID,
        "stochastic_continuation": STOCHASTIC_CONTINUATION,
        "source_archive_sha256": EXPECTED_SOURCE_ARCHIVE_SHA256,
        "source_checkpoint_sha256": SOURCE_MEMBER_SHA256[SOURCE_CHECKPOINT_MEMBER],
        "completed_epoch": int(completed_epoch),
        "next_epoch": int(completed_epoch) + 1,
        "model_class": identity["class"],
        "model_parameter_count": identity["parameter_count"],
        "model_trainable_variable_count": identity["trainable_variable_count"],
        "q_index": identity["q_index"],
        "q_shape": identity["q_shape"],
        "q_dtype": identity["q_dtype"],
        "q_flat_float32_sha256": identity["q_flat_float32_sha256"],
        "latest_state_keras_sha256": model_sha256,
        "optimizer_class": optimizer_state["class"],
        "optimizer_iterations": optimizer_state["iterations"],
        "optimizer_variable_count": optimizer_state["variable_count"],
        "optimizer_learning_rate": optimizer_state["learning_rate"],
        "scheduler_post_step_state": scheduler.get_state(),
        "early_stopping_state": early.get_state(),
        "checkpoint_policy_state": checkpoint_policy_state(policy),
        "combined_history_sha256": sha256_file(history_path),
        "best_val_accuracy_checkpoint_sha256": sha256_file(
            output_root / "checkpoints" / "best_val_accuracy.keras"
        ),
        "source_code_sha256": source_hashes,
        "scientific_result_valid": False,
        "scientific_interpretation": None,
        "test_access": False,
        "partial_epoch": False,
    }


def publish_latest_completed_state(
    *,
    model: Any,
    completed_epoch: int,
    scheduler: TorchCompatibleReduceLROnPlateau,
    early: ValidationLossEarlyStopping,
    policy: CheckpointPolicy,
    history_path: str | Path,
    output_root: str | Path,
    source_hashes: dict[str, str],
    save_model: Callable[[Any, Path], None] = save_model_with_optimizer,
    load_model: Callable[..., Any] = tf.keras.models.load_model,
) -> dict[str, Any]:
    """Verify a temporary full state before replacing the last complete epoch."""

    output_root = Path(output_root)
    history_path = Path(history_path)
    published_model = output_root / LATEST_STATE_NAME
    published_metadata = output_root / LATEST_STATE_METADATA_NAME
    temporary_model = output_root / f".{LATEST_STATE_NAME}.tmp.keras"
    temporary_metadata = output_root / f".{LATEST_STATE_METADATA_NAME}.tmp"
    temporary_model.unlink(missing_ok=True)
    temporary_metadata.unlink(missing_ok=True)
    expected_optimizer = optimizer_identity(model.optimizer)
    expected_q = _q_digest(model)
    try:
        save_model(model, temporary_model)
        restored = load_model(temporary_model, compile=True)
        restored_model, restored_optimizer = verify_model_optimizer_identity(
            restored,
            expected_q_sha256=expected_q,
            expected_iterations=expected_optimizer["iterations"],
            expected_optimizer_variable_count=expected_optimizer["variable_count"],
            expected_lr=expected_optimizer["learning_rate"],
        )
        if restored_model != model_identity(model) or restored_optimizer != expected_optimizer:
            raise CheckpointContinuationError("Latest-state roundtrip identity drift")
        model_sha = sha256_file(temporary_model)
        metadata = _latest_state_metadata(
            model=model,
            model_sha256=model_sha,
            completed_epoch=completed_epoch,
            scheduler=scheduler,
            early=early,
            policy=policy,
            history_path=history_path,
            output_root=output_root,
            source_hashes=source_hashes,
        )
        temporary_metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
            newline="",
        )
        parsed = _json_object(temporary_metadata, "temporary latest-state metadata")
        if parsed != metadata or parsed["latest_state_keras_sha256"] != model_sha:
            raise CheckpointContinuationError("Latest-state metadata verification failed")
        # Both files have been fully written and verified.  A partial next epoch
        # never reaches these replacements, so the preceding complete state stays.
        os.replace(temporary_model, published_model)
        os.replace(temporary_metadata, published_metadata)
        if sha256_file(published_model) != metadata["latest_state_keras_sha256"]:
            raise CheckpointContinuationError("Published latest-state SHA drift")
        return metadata
    finally:
        temporary_model.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)


def _persist_history(output_root: Path, history: list[dict[str, Any]], row: dict[str, Any]) -> None:
    _atomic_json(output_root / "history.json", {"epochs": history})
    _atomic_history_csv(output_root / "train_log.csv", history)
    _atomic_json(output_root / "latest_epoch_summary.json", row)
    write_training_curves(output_root, history)


def run_continuation_epoch_loop(
    *,
    model: Any,
    optimizer: Any,
    execute_train_step: Callable[[Any], Any],
    train_data: Any,
    val_data: Any,
    train_eval_data: Any,
    eval_step: Callable[[Any], Any],
    controls: ResourceControls,
    config: dict[str, Any],
    execution_state: dict[str, Any],
    telemetry: RuntimeTelemetry,
    early: ValidationLossEarlyStopping,
    scheduler: TorchCompatibleReduceLROnPlateau,
    policy: CheckpointPolicy,
    history: list[dict[str, Any]],
    overlap_rows: dict[int, dict[str, Any]],
    output_root: Path,
    source_hashes: dict[str, str],
    max_epoch: int,
    latest_state_publisher: Callable[..., dict[str, Any]] = publish_latest_completed_state,
) -> list[dict[str, Any]]:
    """Mechanically preserve the frozen epoch body from epoch 31 onward."""

    if [row.get("epoch") for row in history] != list(range(1, 31)):
        raise CheckpointContinuationError("Continuation must start from immutable epochs 1..30")
    if max_epoch < FIRST_CONTINUATION_EPOCH:
        raise CheckpointContinuationError("Continuation max epoch must include epoch31")
    optimizer_mode = execution_state["optimizer_execution_mode"]
    for epoch in range(FIRST_CONTINUATION_EPOCH, int(max_epoch) + 1):
        epoch_started = time.perf_counter()
        total_loss = 0.0
        batches = 0
        total_train_batches = len(train_data)
        for batch in train_data.as_dataset(
            epoch, prefetch=controls.tf_data_prefetch
        ):
            step_started = time.perf_counter()
            loss = execute_train_step(batch)
            total_loss += float(loss.numpy())
            batches += 1
            telemetry.train_step_sec.append(time.perf_counter() - step_started)
            trace_count = (
                int(execute_train_step.experimental_get_tracing_count())
                if hasattr(execute_train_step, "experimental_get_tracing_count")
                else 0
            )
            if (
                optimizer_mode == "restricted_tf_function"
                and trace_count > MAX_REGISTERED_TRAIN_STEP_TRACES
            ):
                raise CheckpointContinuationError(
                    f"Registered candidate train step retraced: {trace_count}"
                )
            if batches in (1, total_train_batches) or batches % PROGRESS_INTERVAL_BATCHES == 0:
                _atomic_json(
                    output_root / "runtime_progress.json",
                    {
                        "event": "tensorflow_checkpoint_continuation_train_progress",
                        "epoch": epoch,
                        "batch": batches,
                        "total_batches": total_train_batches,
                        "average_loss": total_loss / batches,
                        "optimizer_iterations": int(optimizer.iterations.numpy()),
                        "test_access": False,
                        "partial_epoch": True,
                    },
                )
        if batches == 0:
            raise CheckpointContinuationError(f"No training batches for epoch {epoch}")
        train_phase_sec = time.perf_counter() - epoch_started
        validation_started = time.perf_counter()
        val_metrics = evaluate_batches(
            model,
            val_data.as_dataset(epoch, prefetch=controls.tf_data_prefetch),
            evaluate_step=eval_step,
        )
        val_phase_sec = time.perf_counter() - validation_started
        telemetry.validation_sec.append(val_phase_sec)

        train_eval_started = time.perf_counter()
        train_metrics = evaluate_batches(
            model,
            train_eval_data.as_dataset(epoch, prefetch=controls.tf_data_prefetch),
            evaluate_step=eval_step,
        )
        train_eval_phase_sec = time.perf_counter() - train_eval_started
        signatures = {
            "config": canonical_config_hash(config),
            "graph": config["locked"]["graph_signature"],
            "feature": config["locked"]["feature_signature"],
            "prior": config["locked"]["prior_signature"],
            "dataset_split": config["locked"]["dataset_split_signature"],
        }
        metadata = {
            "seed": 42,
            "config_hash": signatures["config"],
            "package_checksum": config["locked"]["package_checksum"],
            "graph_signature": signatures["graph"],
            "feature_signature": signatures["feature"],
            "prior_signature": signatures["prior"],
            "dataset_split_signature": signatures["dataset_split"],
            "tensorflow_version": tf.__version__,
            "keras_version": keras_version(),
            "optimizer_state": optimizer_identity(optimizer),
            "scheduler_state": scheduler.get_state(),
            "mixed_precision_policy": tf.keras.mixed_precision.global_policy().name,
            "execution_mode": execution_state,
            "execution_contract_sha256": config["locked"]["execution_contract_sha256"],
            "resource_settings": controls.__dict__,
            "continuation_protocol_id": PROTOCOL_ID,
            "stochastic_continuation": STOCHASTIC_CONTINUATION,
            "test_access": False,
        }

        # Registered frozen order: early -> checkpoint -> scheduler -> history.
        stop = early.update(epoch, val_metrics["loss"])
        metadata["early_stopping_state"] = early.get_state()
        checkpoint = policy.update_best(model, optimizer, epoch, val_metrics, metadata)
        lr = scheduler.step(val_metrics["loss"])
        metadata["scheduler_state"] = scheduler.get_state()
        metadata["optimizer_state"]["learning_rate"] = lr
        train_macro = train_metrics["macro_f1"]
        early_state = early.get_state()
        row = {
            "epoch": epoch,
            "train_loss": total_loss / batches,
            "train_eval_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_macro,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "train_val_macro_gap_pp": 100.0 * (float(train_macro) - float(val_metrics["macro_f1"])),
            "lr": lr,
            "train_phase_sec": train_phase_sec,
            "val_phase_sec": val_phase_sec,
            "train_eval_phase_sec": train_eval_phase_sec,
            "train_eval_performed": True,
            "early_stopping_wait": int(early_state["epochs_without_improvement"]),
            "early_stopping_patience": int(early.patience),
            "stop_requested": bool(stop),
            "epoch_time_sec": time.perf_counter() - epoch_started,
            **checkpoint,
        }
        history.append(row)
        _persist_history(output_root, history, row)
        _print_epoch_summary(row, policy)
        if epoch in overlap_rows:
            _atomic_json(
                output_root / f"resume_overlap_epoch{epoch}.json",
                overlap_diagnostic(row, overlap_rows[epoch]),
            )
        latest_state_publisher(
            model=model,
            completed_epoch=epoch,
            scheduler=scheduler,
            early=early,
            policy=policy,
            history_path=output_root / "history.json",
            output_root=output_root,
            source_hashes=source_hashes,
        )
        if stop:
            break
    return history


def _verify_no_forbidden_artifacts(output_root: Path) -> None:
    forbidden = [name for name in FORBIDDEN_ARTIFACT_NAMES if (output_root / name).exists()]
    forbidden.extend(path.name for path in output_root.glob("test_metrics_*.json"))
    if forbidden:
        raise CheckpointContinuationError(
            f"Forbidden post-test artifacts found: {sorted(set(forbidden))}"
        )


def run_checkpoint_conditioned_continuation(
    *,
    config_path: str | Path,
    fer_csv: str | Path,
    prior_root: str | Path,
    source_archive: str | Path,
    output_root: str | Path,
    controls: ResourceControls,
) -> dict[str, Any]:
    """Run registered validation-only continuation; no final-test path exists."""

    source_hashes = verify_source_locks()
    input_config_path, config = verify_seed42_config(config_path)
    verify_resource_controls(controls)
    extraction_root = Path(output_root).expanduser().resolve().with_name(
        Path(output_root).name + ".source_evidence.tmp"
    )
    if extraction_root.exists():
        raise FileExistsError(f"Temporary source evidence path already exists: {extraction_root}")
    try:
        extracted = verify_and_extract_source_archive(source_archive, extraction_root)
        prefix, overlap_rows, _source_config = load_reviewed_source(extracted)
        output_root = initialize_output(
            output_root,
            extracted,
            prefix,
            overlap_rows,
            source_hashes,
            input_config_path,
        )
        config["data"]["prior_dir"] = str(Path(prior_root).expanduser().resolve())
        config["data"]["fer_csv"] = str(Path(fer_csv).expanduser().resolve())
        config["training"]["batch_size"] = int(controls.batch_size)
        config["resources"].update(controls.__dict__)
        (output_root / "resolved_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        _atomic_json(output_root / "resolved_config.json", config)
        signatures = {
            "config": canonical_config_hash(config),
            "graph": config["locked"]["graph_signature"],
            "feature": config["locked"]["feature_signature"],
            "prior": config["locked"]["prior_signature"],
            "dataset_split": config["locked"]["dataset_split_signature"],
        }
        write_provenance(output_root, config, signatures)

        # Seed setup occurs exactly once for the checkpoint-conditioned restart.
        seed_everything(42)
        controls.apply()
        execution_state = validate_execution_config(config["training"])
        telemetry = RuntimeTelemetry()
        train_data = GraphBatchGenerator(
            prior_root,
            "train",
            config,
            controls.batch_size,
            42,
            True,
            controls.graph_cache_size,
            telemetry,
            graph_workers=controls.graph_workers,
            clean_graph_cache_dir=controls.clean_graph_cache_dir,
        )
        val_data = GraphBatchGenerator(
            prior_root,
            "val",
            config,
            controls.eval_batch_size,
            42,
            False,
            controls.graph_cache_size,
            telemetry,
            graph_workers=controls.graph_workers,
            clean_graph_cache_dir=controls.clean_graph_cache_dir,
        )
        eval_config = json.loads(json.dumps(config))
        eval_config["graph"]["prior_corruption"]["enabled"] = False
        train_eval_data = GraphBatchGenerator(
            prior_root,
            "train",
            eval_config,
            controls.eval_batch_size,
            42,
            False,
            controls.graph_cache_size,
            telemetry,
            graph_workers=controls.graph_workers,
            clean_graph_cache_dir=controls.clean_graph_cache_dir,
        )
        model = load_resume_checkpoint(extracted[SOURCE_CHECKPOINT_MEMBER])
        optimizer = model.optimizer
        eval_step = build_compiled_evaluation_step(model)
        iterations_before_gate = int(optimizer.iterations.numpy())
        pretrain_metrics = evaluate_batches(
            model,
            val_data.as_dataset(RESUME_EPOCH, prefetch=controls.tf_data_prefetch),
            evaluate_step=eval_step,
        )
        gate = validate_pretrain_validation_gate(pretrain_metrics)
        if int(optimizer.iterations.numpy()) != iterations_before_gate:
            raise CheckpointContinuationError("Optimizer changed during pre-train validation gate")
        _atomic_json(output_root / "pretrain_validation_gate.json", gate)

        early, scheduler, policy = reconstruct_control_state(optimizer, config, output_root)
        execute_train_step = build_candidate_restricted_graph_train_step(
            model,
            optimizer,
            input_signature=GraphBatchGenerator.output_signature(),
        )
        history = run_continuation_epoch_loop(
            model=model,
            optimizer=optimizer,
            execute_train_step=execute_train_step,
            train_data=train_data,
            val_data=val_data,
            train_eval_data=train_eval_data,
            eval_step=eval_step,
            controls=controls,
            config=config,
            execution_state=execution_state,
            telemetry=telemetry,
            early=early,
            scheduler=scheduler,
            policy=policy,
            history=prefix,
            overlap_rows=overlap_rows,
            output_root=output_root,
            source_hashes=source_hashes,
            max_epoch=int(config["training"]["max_epochs"]),
        )
        _atomic_json(output_root / "telemetry.json", telemetry.to_dict())
        _verify_no_forbidden_artifacts(output_root)
        final_epoch = int(history[-1]["epoch"])
        marker = {
            "schema_version": 1,
            "continuation_protocol_id": PROTOCOL_ID,
            "implementation_base": IMPLEMENTATION_BASE,
            "stochastic_continuation": STOCHASTIC_CONTINUATION,
            "source_archive_sha256": EXPECTED_SOURCE_ARCHIVE_SHA256,
            "source_checkpoint_sha256": SOURCE_MEMBER_SHA256[SOURCE_CHECKPOINT_MEMBER],
            "continuation_completed": True,
            "completion_reason": (
                "early_stopping" if history[-1]["stop_requested"] else "max_epochs"
            ),
            "scientific_prefix_first_epoch": 1,
            "scientific_prefix_last_epoch": 30,
            "first_continuation_epoch": 31,
            "final_completed_epoch": final_epoch,
            "first_run_overlap_diagnostics": [31, 32],
            "original_step12_scientific_result_valid": False,
            "original_step12_scientific_interpretation": None,
            "scientific_result_valid": False,
            "scientific_interpretation": None,
            "training": True,
            "optimizer_gradient_updates": True,
            "test_access": False,
            "test_data_constructed": False,
            "final_test_skipped": True,
            "latest_state_path": str(output_root / LATEST_STATE_NAME),
            "latest_state_sha256": sha256_file(output_root / LATEST_STATE_NAME),
            "combined_history_sha256": sha256_file(output_root / "history.json"),
            "source_code_sha256": source_hashes,
        }
        _atomic_json(output_root / COMPLETION_MARKER_NAME, marker)
        return marker
    except Exception as exc:
        resolved_output = Path(output_root).expanduser().resolve()
        if resolved_output.is_dir():
            _atomic_json(
                resolved_output / FAILURE_MARKER_NAME,
                {
                    "schema_version": 1,
                    "continuation_protocol_id": PROTOCOL_ID,
                    "status": "TECHNICAL_CONTINUATION_FAILURE",
                    "error_type": type(exc).__name__,
                    "error_text": str(exc),
                    "scientific_result_valid": False,
                    "scientific_interpretation": None,
                    "automatic_retry": False,
                    "test_access": False,
                },
            )
        raise
    finally:
        if extraction_root.exists():
            shutil.rmtree(extraction_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the registered epoch30 checkpoint-conditioned validation-only continuation."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--fer-csv", required=True)
    parser.add_argument("--prior-root", required=True)
    parser.add_argument("--source-archive", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--graph-workers", type=int, default=2)
    parser.add_argument("--intra-op-threads", type=int, default=0)
    parser.add_argument("--inter-op-threads", type=int, default=0)
    parser.add_argument("--tf-data-prefetch", type=int, default=2)
    parser.add_argument("--tf-data-parallel-calls", type=int, default=1)
    parser.add_argument("--graph-cache-size", type=int, default=64)
    parser.add_argument("--clean-graph-cache-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--xla", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--memory-growth", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _resource_controls(args: argparse.Namespace) -> ResourceControls:
    return ResourceControls(
        intra_op_threads=args.intra_op_threads,
        inter_op_threads=args.inter_op_threads,
        graph_workers=args.graph_workers,
        tf_data_prefetch=args.tf_data_prefetch,
        tf_data_parallel_calls=args.tf_data_parallel_calls,
        graph_cache_size=args.graph_cache_size,
        clean_graph_cache_dir=args.clean_graph_cache_dir,
        memory_growth=args.memory_growth,
        mixed_precision=args.mixed_precision,
        xla=args.xla,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        device=args.device,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.device.lower().startswith("gpu"):
        if not tf.config.list_physical_devices("GPU"):
            parser.error("Registered GPU continuation requested but no GPU is available")
    marker = run_checkpoint_conditioned_continuation(
        config_path=args.config,
        fer_csv=args.fer_csv,
        prior_root=args.prior_root,
        source_archive=args.source_archive,
        output_root=args.output_root,
        controls=_resource_controls(args),
    )
    print(json.dumps(marker, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
