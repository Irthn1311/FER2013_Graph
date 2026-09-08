"""Build and support the preregistered Issue #31 Kaggle pre-run adapter.

The runtime helpers are deliberately standard-library only at import time so
the adapter regressions never import TensorFlow or execute FER training.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from typing import Any, Sequence
import zipfile


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    ROOT / "notebooks" / "kaggle-issue31-learned-local-residual-slots-seed42.ipynb"
)

REPOSITORY_URL = "https://github.com/Irthn1311/FER2013_Graph.git"
EXECUTION_COMMIT = "cc54ec045f2af0dad6aca4bf4b8b1710677ab1a4"
SCIENTIFIC_PAYLOAD_SHA256 = (
    "286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e"
)
BASELINE_EXECUTION_CONTRACT_SHA256 = (
    "14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22"
)
SOURCE_LOCKS = {
    "candidate_model": (
        "research/candidates/tf_learned_local_residual_slots/model.py",
        "0a7cfa315baf0d6666b7ab86139b328ab6d138985cfddd71180410675602fcca",
    ),
    "candidate_execution_adapter": (
        "research/candidates/tf_learned_local_residual_slots/candidate_execution.py",
        "48c0e5f8ad4676e17fb4127b3a30ad053beedca8e04e05cfb6fb24f2bb9236f9",
    ),
    "candidate_execution_contract": (
        "research/candidates/tf_learned_local_residual_slots/candidate_execution_contract.json",
        "331570bacd3ec97474c85f25e7e3cb461ef42b0aa3f442caf3dd1f52314bcbc7",
    ),
    "candidate_validation_harness": (
        "research/candidates/tf_learned_local_residual_slots/train_validation_only.py",
        "1b0707c41f30a9a5b9b9dba3995030ac50fccc90cf439d1ac26a31a32a878f2f",
    ),
    "frozen_execution": (
        "standalone/lap_gnn_tensorflow_ofix7_mid_candidate/src/lap_gnn_tf/training/execution.py",
        "2f0a579f51fb216d859b2a7e063614e7f76e5a74948067b7d7abd9f2d59e2f70",
    ),
    "frozen_validation_only_wrapper": (
        "standalone/lap_gnn_tensorflow_ofix7_mid_candidate/tools/train_validation_only.py",
        "c94c122066fdd19210c8ba64a2a61567b249fad4f69c69cb4236b68cce6ff7b4",
    ),
    "frozen_trainer": (
        "standalone/lap_gnn_tensorflow_ofix7_mid_candidate/src/lap_gnn_tf/training/trainer.py",
        "4c3cb1aa311578038ff656cb7d119103ae5a651135f8ee1c76e37c2c04c1fc75",
    ),
    "seed42_config": (
        "standalone/lap_gnn_tensorflow_ofix7_mid_candidate/configs/fer2013_ofix7_mid_tensorflow_seed42.yaml",
        "aa3bf2d3932bbad6c5f8cdcc347f4a9866e2c027d6135a60b5002a8f6a3b6908",
    ),
}

EXPECTED_PYTHON = "3.12.12"
EXPECTED_TENSORFLOW = "2.18.1"
EXPECTED_KERAS = "3.15.0"
EXPECTED_GPU_COUNT = 2
EXPECTED_GPU_TOKEN = "T4"
EXPECTED_CANDIDATE_CLASS = "LearnedLocalResidualSlotLapGNN"
EXPECTED_CANDIDATE_PARAMS = 1_061_576
EXPECTED_VARIABLE_COUNT = 128
EXPECTED_Q_INDEX = 127
EXPECTED_Q_SHAPE = [4, 96]
EXPECTED_Q_DTYPE = "float32"

BASELINE_BEST_VAL_MACRO_F1 = 0.601166548701511
BASELINE_BEST_VAL_MACRO_EPOCH = 26
BASELINE_TRAIN_MACRO_AT_BEST = 0.7562805286580438
BASELINE_TRAIN_VAL_GAP_PP = 15.511397995653287
BASELINE_BEST_VAL_ACCURACY = 0.6319308999721371
BASELINE_BEST_VAL_ACCURACY_EPOCH = 31
BASELINE_EPOCH31_VAL_MACRO_F1 = 0.5938407974340496
BASELINE_BEST_VAL_LOSS = 1.0625020856350924
BASELINE_BEST_VAL_LOSS_EPOCH = 17
PRACTICAL_EFFECT_THRESHOLD_PP = 1.0

FER_ROOT = Path("/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split")
FER_TRAIN_CSV = FER_ROOT / "train.csv"
FER_VAL_CSV = FER_ROOT / "val.csv"
PRIOR_ROOT = Path(
    "/kaggle/input/datasets/irthn1311/d16-mediapipe-pixel-priors-best-retry-rescue/outputs/d16_mediapipe_pixel_priors_best_retry_rescue"
)
CACHE_ROOT = Path("/kaggle/input/datasets/irthn1311/ofix7-mid-seed42-records")
WORKING_ROOT = Path("/kaggle/working")
CHECKOUT_ROOT = WORKING_ROOT / "FER2013_Graph_issue31"
RUN_ROOT = WORKING_ROOT / "tf_step12_learned_local_residual_slots_seed42"
TRAIN_OUTPUT_ROOT = RUN_ROOT / "run"
ADAPTER_ROOT = RUN_ROOT / "adapter"
PRE_RUN_MANIFEST_PATH = ADAPTER_ROOT / "pre_run_manifest.json"
SUBPROCESS_LOG_PATH = ADAPTER_ROOT / "subprocess.log"
WRAPPER_EXECUTION_PATH = ADAPTER_ROOT / "wrapper_execution.json"
FINAL_EVIDENCE_PATH = ADAPTER_ROOT / "final_evidence.json"
RUNTIME_PROGRESS_PATH = ADAPTER_ROOT / "runtime_progress.json"
FAILURE_REPORT_PATH = ADAPTER_ROOT / "technical_or_runtime_failure.md"
REPORT_PATH = WORKING_ROOT / "tf_step12_learned_local_residual_slots_seed42.md"
ARCHIVE_PATH = (
    WORKING_ROOT / "tf_step12_learned_local_residual_slots_seed42_kaggle_t4.zip"
)

TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
GRAPH_WORKERS = 2
TF_DATA_PREFETCH = 2
TF_DATA_PARALLEL_CALLS = 1
GRAPH_CACHE_SIZE = 64
EXPECTED_SPLIT_COUNTS = {"train": 28_709, "val": 3_589}

FROZEN_MARKER = "VALIDATION_ONLY_COMPLETE.json"
CANDIDATE_MARKER = "CANDIDATE_VALIDATION_ONLY_COMPLETE.json"
FORBIDDEN_OUTPUT_NAMES = {
    "TRAINING_COMPLETE.json",
    "run_summary.json",
    "predictions.csv",
    "per_class_metrics.csv",
    "confusion_matrix.csv",
    "confusion_matrix.png",
}
ROLLING_RELATIVE_FILES = (
    "history.json",
    "train_log.csv",
    "latest_epoch_summary.json",
    "telemetry.json",
    "resolved_config.json",
    "resolved_config.yaml",
    "provenance.json",
    "checkpoints/best_val_accuracy.keras",
    "checkpoints/best_val_accuracy.weights.h5",
    "checkpoints/best_val_accuracy.metadata.json",
    FROZEN_MARKER,
    CANDIDATE_MARKER,
)


class AdapterError(RuntimeError):
    """Raised when Issue #31 evidence cannot be established fail closed."""


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, destination)


def json_object(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdapterError(f"Malformed or unreadable {label}: {source}") from exc
    if not isinstance(payload, dict):
        raise AdapterError(f"{label} must be a JSON object: {source}")
    return payload


def run_checked(command: Sequence[Any], *, cwd: str | Path | None = None) -> str:
    actual = [str(item) for item in command]
    print("$", " ".join(actual), flush=True)
    result = subprocess.run(
        actual,
        cwd=None if cwd is None else str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.returncode:
        raise AdapterError(f"Command failed with exit code {result.returncode}: {actual}")
    return result.stdout


def verify_source_locks(checkout_root: str | Path) -> dict[str, str]:
    root = Path(checkout_root)
    actual: dict[str, str] = {}
    for label, (relative, expected) in SOURCE_LOCKS.items():
        path = root / relative
        if not path.is_file():
            raise AdapterError(f"Locked source missing: {label}: {path}")
        digest = sha256_file(path)
        if digest != expected:
            raise AdapterError(
                f"Locked source drift: {label}: expected {expected}, got {digest}"
            )
        actual[label] = digest
    package_root = root / "standalone/lap_gnn_tensorflow_ofix7_mid_candidate"
    manifest = json_object(package_root / "package_manifest.json", "package manifest")
    if manifest.get("scientific_payload_sha256") != SCIENTIFIC_PAYLOAD_SHA256:
        raise AdapterError("Frozen scientific payload manifest drift")
    if (
        manifest.get("execution_contract_sha256")
        != BASELINE_EXECUTION_CONTRACT_SHA256
    ):
        raise AdapterError("Inherited execution contract manifest drift")
    contract = json_object(
        root / SOURCE_LOCKS["candidate_execution_contract"][0],
        "candidate execution contract",
    )
    required_contract = {
        "selected_mode": "restricted_tf_function",
        "selected_grappler_profile": "G1-A",
        "expected_trainable_variable_count": EXPECTED_VARIABLE_COUNT,
        "baseline_variable_prefix_count": EXPECTED_Q_INDEX,
        "inherited_baseline_execution_contract_sha256": (
            BASELINE_EXECUTION_CONTRACT_SHA256
        ),
    }
    for key, expected in required_contract.items():
        if contract.get(key) != expected:
            raise AdapterError(f"Candidate execution contract drift: {key}")
    if contract.get("q") != {
        "dtype": EXPECTED_Q_DTYPE,
        "index": EXPECTED_Q_INDEX,
        "scalar_count": 384,
        "shape": EXPECTED_Q_SHAPE,
    }:
        raise AdapterError("Candidate execution contract Q identity drift")
    if contract.get("precision_boundary") != {
        "mixed_float16_supported": True,
        "official_global_cast": False,
        "raw_slot_diagnostics_dtype": "float32",
        "residual_input_dtype": "official_global_dtype",
        "slot_compute_dtype": "float32",
    }:
        raise AdapterError("Candidate mixed-precision boundary contract drift")
    return actual


def registered_decision(delta_macro_pp: float) -> str:
    value = float(delta_macro_pp)
    if value >= PRACTICAL_EFFECT_THRESHOLD_PP:
        return "PROMISING_SINGLE_SEED_VALIDATION_GAIN"
    if value <= -PRACTICAL_EFFECT_THRESHOLD_PP:
        return "SINGLE_SEED_VALIDATION_REGRESSION"
    return "NO_CLEAR_SINGLE_SEED_DIFFERENCE"


def registered_command(checkout_root: str | Path, output_root: str | Path) -> list[str]:
    root = Path(checkout_root)
    harness = root / SOURCE_LOCKS["candidate_validation_harness"][0]
    config = root / SOURCE_LOCKS["seed42_config"][0]
    command = [
        sys.executable,
        "-B",
        str(harness),
        "--config",
        str(config),
        "--fer-csv",
        str(FER_TRAIN_CSV),
        "--prior-root",
        str(PRIOR_ROOT),
        "--output-root",
        str(Path(output_root)),
        "--device",
        "gpu",
        "--graph-workers",
        str(GRAPH_WORKERS),
        "--tf-data-prefetch",
        str(TF_DATA_PREFETCH),
        "--tf-data-parallel-calls",
        str(TF_DATA_PARALLEL_CALLS),
        "--graph-cache-size",
        str(GRAPH_CACHE_SIZE),
        "--clean-graph-cache-dir",
        str(CACHE_ROOT),
        "--batch-size",
        str(TRAIN_BATCH_SIZE),
        "--eval-batch-size",
        str(EVAL_BATCH_SIZE),
        "--mixed-precision",
        "--no-xla",
        "--memory-growth",
        "--no-resume",
    ]
    if any(argument.startswith("--limit-") for argument in command):
        raise AdapterError("Registered command must not contain bounded limits")
    return command


def archive_members(
    run_root: str | Path,
    train_output_root: str | Path,
    report_path: str | Path,
) -> list[tuple[Path, str]]:
    run = Path(run_root)
    output = Path(train_output_root)
    members: list[tuple[Path, str]] = []
    adapter = run / "adapter"
    for path in sorted(adapter.glob("*")) if adapter.is_dir() else []:
        if path.is_file() and path.name != "final_evidence.json.tmp":
            members.append((path, f"adapter/{path.name}"))
    for relative in ROLLING_RELATIVE_FILES:
        path = output / relative
        if path.is_file():
            members.append((path, f"run/{Path(relative).as_posix()}"))
    report = Path(report_path)
    if report.is_file():
        members.append((report, report.name))
    names = [name for _path, name in members]
    forbidden = [
        name
        for name in names
        if Path(name).name in FORBIDDEN_OUTPUT_NAMES
        or Path(name).name.startswith("test_metrics_")
    ]
    if forbidden:
        raise AdapterError(f"Forbidden post-test archive members: {forbidden}")
    if len(names) != len(set(names)):
        raise AdapterError("Duplicate archive member names")
    return members


def publish_archive_atomic(
    archive_path: str | Path,
    run_root: str | Path,
    train_output_root: str | Path,
    report_path: str | Path,
) -> list[str]:
    destination = Path(archive_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    members = archive_members(run_root, train_output_root, report_path)
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path, name in members:
            archive.write(path, name)
    with zipfile.ZipFile(temporary, "r") as archive:
        names = archive.namelist()
        if archive.testzip() is not None:
            raise AdapterError("Rolling archive verification failed")
        if (Path(run_root) / "adapter" / "pre_run_manifest.json").is_file() and (
            "adapter/pre_run_manifest.json" not in names
        ):
            raise AdapterError("Rolling archive omitted pre-run manifest")
    os.replace(temporary, destination)
    return names


class RollingArchiveMonitor:
    """Read-only history monitor; it never mutates trainer-owned artifacts."""

    def __init__(
        self,
        *,
        history_path: str | Path,
        progress_path: str | Path,
        archive_path: str | Path,
        run_root: str | Path,
        train_output_root: str | Path,
        report_path: str | Path,
        archive_lock: threading.Lock | None = None,
        poll_seconds: float = 2.0,
    ) -> None:
        self.history_path = Path(history_path)
        self.progress_path = Path(progress_path)
        self.archive_path = Path(archive_path)
        self.run_root = Path(run_root)
        self.train_output_root = Path(train_output_root)
        self.report_path = Path(report_path)
        self.archive_lock = archive_lock or threading.Lock()
        self.poll_seconds = float(poll_seconds)
        self.last_epoch = 0

    def poll_once(self) -> bool:
        if not self.history_path.is_file():
            return False
        try:
            history = json.loads(self.history_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return False
        rows = history.get("epochs") if isinstance(history, dict) else None
        if not isinstance(rows, list) or not rows or not isinstance(rows[-1], dict):
            return False
        epoch = rows[-1].get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= self.last_epoch:
            return False
        atomic_json(
            self.progress_path,
            {
                "schema_version": 1,
                "status": "RUNNING",
                "latest_completed_epoch": epoch,
                "completed_epoch_count": len(rows),
                "scientific_result_valid": False,
                "scientific_interpretation": None,
                "monitor_read_only": True,
            },
        )
        with self.archive_lock:
            publish_archive_atomic(
                self.archive_path,
                self.run_root,
                self.train_output_root,
                self.report_path,
            )
        self.last_epoch = epoch
        return True

    def run(self, stop_event: threading.Event) -> None:
        while not stop_event.wait(self.poll_seconds):
            try:
                self.poll_once()
            except BaseException as exc:
                atomic_json(
                    self.progress_path,
                    {
                        "schema_version": 1,
                        "status": "MONITOR_ARCHIVE_RETRY",
                        "latest_completed_epoch": self.last_epoch,
                        "scientific_result_valid": False,
                        "scientific_interpretation": None,
                        "monitor_read_only": True,
                        "archive_error": f"{type(exc).__name__}: {exc}",
                    },
                )
        try:
            self.poll_once()
        except BaseException:
            pass


def _validated_rows(history: dict[str, Any]) -> list[dict[str, Any]]:
    rows = history.get("epochs")
    if not isinstance(rows, list) or not rows:
        raise AdapterError("Complete history requires a non-empty epochs list")
    required_numeric = (
        "epoch",
        "train_macro_f1",
        "val_macro_f1",
        "val_accuracy",
        "val_loss",
        "lr",
        "early_stopping_wait",
    )
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or row.get("epoch") != index:
            raise AdapterError("History epochs must be complete and sequential from one")
        for key in required_numeric:
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise AdapterError(f"History row {index} lacks numeric {key}")
    return rows


def derive_registered_metrics(history: dict[str, Any]) -> dict[str, Any]:
    rows = _validated_rows(history)
    best_macro_value = max(float(row["val_macro_f1"]) for row in rows)
    best_macro = next(
        row for row in rows if float(row["val_macro_f1"]) == best_macro_value
    )
    best_accuracy_value = max(float(row["val_accuracy"]) for row in rows)
    best_accuracy = next(
        row for row in rows if float(row["val_accuracy"]) == best_accuracy_value
    )
    best_loss_value = min(float(row["val_loss"]) for row in rows)
    best_loss = next(row for row in rows if float(row["val_loss"]) == best_loss_value)
    train_macro = float(best_macro["train_macro_f1"])
    gap_pp = 100.0 * (train_macro - best_macro_value)
    delta_pp = 100.0 * (best_macro_value - BASELINE_BEST_VAL_MACRO_F1)
    lr_trajectory = [
        {"epoch": int(row["epoch"]), "lr": float(row["lr"])} for row in rows
    ]
    lr_reductions = [
        current
        for previous, current in zip(lr_trajectory, lr_trajectory[1:])
        if current["lr"] < previous["lr"]
    ]
    return {
        "candidate_best_val_macro_f1": best_macro_value,
        "candidate_best_val_macro_epoch": int(best_macro["epoch"]),
        "candidate_train_macro_f1_at_best_macro_epoch": train_macro,
        "candidate_train_validation_macro_gap_pp": gap_pp,
        "gap_difference_vs_baseline_pp": gap_pp - BASELINE_TRAIN_VAL_GAP_PP,
        "delta_macro_pp": delta_pp,
        "registered_decision": registered_decision(delta_pp),
        "candidate_best_val_accuracy": best_accuracy_value,
        "candidate_best_val_accuracy_epoch": int(best_accuracy["epoch"]),
        "delta_best_val_accuracy": best_accuracy_value - BASELINE_BEST_VAL_ACCURACY,
        "candidate_best_val_loss": best_loss_value,
        "candidate_best_val_loss_epoch": int(best_loss["epoch"]),
        "delta_best_val_loss": best_loss_value - BASELINE_BEST_VAL_LOSS,
        "lr_trajectory": lr_trajectory,
        "lr_reductions": lr_reductions,
        "early_stopping_trajectory": [
            {
                "epoch": int(row["epoch"]),
                "wait": int(row["early_stopping_wait"]),
                "stop_requested": bool(row.get("stop_requested", False)),
            }
            for row in rows
        ],
        "final_observed_epoch": int(rows[-1]["epoch"]),
    }


def validate_completion(
    train_output_root: str | Path,
    source_hashes_before: dict[str, str],
    source_hashes_after: dict[str, str],
) -> dict[str, Any]:
    output = Path(train_output_root)
    if source_hashes_before != source_hashes_after:
        raise AdapterError("Reviewed source hashes changed during execution")
    forbidden = [name for name in FORBIDDEN_OUTPUT_NAMES if (output / name).exists()]
    forbidden.extend(path.name for path in output.glob("test_metrics_*.json"))
    if forbidden:
        raise AdapterError(f"Forbidden post-test output contamination: {forbidden}")
    frozen = json_object(output / FROZEN_MARKER, "frozen validation-only marker")
    candidate = json_object(output / CANDIDATE_MARKER, "candidate completion sidecar")
    required_frozen = {
        "training_validation_completed": True,
        "final_test_skipped": True,
        "test_accessed": False,
        "test_data_constructed": False,
        "test_checkpoint_loaded": False,
        "normal_full_training_completed": False,
        "boundary": "before_resolve_final_checkpoint",
        "trainer_revision_guard_passed": True,
        "intercepted_function_restored": True,
        "trainer_source_sha256": SOURCE_LOCKS["frozen_trainer"][1],
        "scientific_payload_sha256": SCIENTIFIC_PAYLOAD_SHA256,
        "input_config_sha256": SOURCE_LOCKS["seed42_config"][1],
        "seed": 42,
    }
    required_candidate = {
        "training_validation_completed": True,
        "final_test_skipped": True,
        "test_access": False,
        "original_constructor_restored": True,
        "original_restricted_builder_restored": True,
        "candidate_constructor_injected": True,
        "candidate_restricted_builder_injected": True,
        "candidate_class": EXPECTED_CANDIDATE_CLASS,
        "actual_candidate_parameter_count": EXPECTED_CANDIDATE_PARAMS,
        "candidate_trainable_variable_count": EXPECTED_VARIABLE_COUNT,
        "q_index": EXPECTED_Q_INDEX,
        "q_shape": EXPECTED_Q_SHAPE,
        "q_dtype": EXPECTED_Q_DTYPE,
        "candidate_harness_sha256": SOURCE_LOCKS["candidate_validation_harness"][1],
        "candidate_model_sha256": SOURCE_LOCKS["candidate_model"][1],
        "candidate_execution_adapter_sha256": SOURCE_LOCKS[
            "candidate_execution_adapter"
        ][1],
        "candidate_execution_contract_sha256": SOURCE_LOCKS[
            "candidate_execution_contract"
        ][1],
        "frozen_validation_only_wrapper_sha256": SOURCE_LOCKS[
            "frozen_validation_only_wrapper"
        ][1],
        "frozen_trainer_sha256": SOURCE_LOCKS["frozen_trainer"][1],
        "frozen_execution_sha256": SOURCE_LOCKS["frozen_execution"][1],
        "scientific_payload_sha256": SCIENTIFIC_PAYLOAD_SHA256,
        "input_config_sha256": SOURCE_LOCKS["seed42_config"][1],
        "inherited_baseline_execution_contract_sha256": (
            BASELINE_EXECUTION_CONTRACT_SHA256
        ),
    }
    for label, payload, required in (
        ("frozen marker", frozen, required_frozen),
        ("candidate sidecar", candidate, required_candidate),
    ):
        for key, expected in required.items():
            if payload.get(key) != expected:
                raise AdapterError(f"{label} validity gate drift: {key}")
    sidecar_keys = {
        "candidate_model",
        "candidate_execution_adapter",
        "candidate_execution_contract",
        "frozen_validation_only_wrapper",
        "frozen_trainer",
        "frozen_execution",
    }
    expected_sidecar_before = {
        key: value for key, value in source_hashes_before.items() if key in sidecar_keys
    }
    expected_sidecar_after = {
        key: value for key, value in source_hashes_after.items() if key in sidecar_keys
    }
    if candidate.get("source_artifact_sha256_before") != expected_sidecar_before:
        raise AdapterError("Candidate sidecar source-before hashes drift")
    if candidate.get("source_artifact_sha256_after") != expected_sidecar_after:
        raise AdapterError("Candidate sidecar source-after hashes drift")
    history_path = output / "history.json"
    history = json_object(history_path, "complete history")
    rows = _validated_rows(history)
    if frozen.get("final_observed_epoch") != rows[-1]["epoch"]:
        raise AdapterError("Frozen marker/history final epoch drift")
    if candidate.get("final_observed_epoch") != rows[-1]["epoch"]:
        raise AdapterError("Candidate sidecar/history final epoch drift")
    if frozen.get("history_sha256") != sha256_file(history_path):
        raise AdapterError("Frozen history digest drift")
    if candidate.get("history_sha256") != sha256_file(history_path):
        raise AdapterError("Candidate history digest drift")
    if candidate.get("validation_only_marker_sha256") != sha256_file(
        output / FROZEN_MARKER
    ):
        raise AdapterError("Candidate frozen-marker digest drift")
    resolved_path = output / "resolved_config.json"
    resolved = json_object(resolved_path, "resolved config")
    if frozen.get("resolved_config_sha256") != sha256_file(resolved_path):
        raise AdapterError("Frozen resolved-config digest drift")
    if candidate.get("resolved_config_sha256") != sha256_file(resolved_path):
        raise AdapterError("Candidate resolved-config digest drift")
    locked = resolved.get("locked", {})
    training = resolved.get("training", {})
    resources = resolved.get("resources", {})
    resolved_required = {
        "seed": (resolved.get("seed"), 42),
        "payload": (locked.get("package_checksum"), SCIENTIFIC_PAYLOAD_SHA256),
        "execution contract": (
            locked.get("execution_contract_sha256"),
            BASELINE_EXECUTION_CONTRACT_SHA256,
        ),
        "baseline parameter lock": (locked.get("parameter_count"), 1_061_192),
        "optimizer execution": (
            training.get("optimizer_execution_mode"),
            "restricted_tf_function",
        ),
        "gradient execution": (training.get("gradient_execution_mode"), "tf_function"),
        "Grappler profile": (training.get("grappler_profile"), "G1-A"),
        "train batch": (resources.get("batch_size"), TRAIN_BATCH_SIZE),
        "eval batch": (resources.get("eval_batch_size"), EVAL_BATCH_SIZE),
        "graph workers": (resources.get("graph_workers"), GRAPH_WORKERS),
        "tf-data prefetch": (resources.get("tf_data_prefetch"), TF_DATA_PREFETCH),
        "tf-data parallel calls": (
            resources.get("tf_data_parallel_calls"),
            TF_DATA_PARALLEL_CALLS,
        ),
        "graph cache": (resources.get("graph_cache_size"), GRAPH_CACHE_SIZE),
        "mixed precision": (resources.get("mixed_precision"), True),
        "XLA": (resources.get("xla"), False),
        "memory growth": (resources.get("memory_growth"), True),
    }
    for label, (actual, expected) in resolved_required.items():
        if actual != expected:
            raise AdapterError(f"Resolved registered runtime drift: {label}")
    if any(value is not None for value in frozen.get("bounded_limits", {}).values()):
        raise AdapterError("Registered run used a bounded limit")
    checkpoint = output / "checkpoints" / "best_val_accuracy.keras"
    weights = output / "checkpoints" / "best_val_accuracy.weights.h5"
    metadata_path = output / "checkpoints" / "best_val_accuracy.metadata.json"
    for required_path in (checkpoint, weights, metadata_path):
        if not required_path.is_file():
            raise AdapterError(f"Required candidate checkpoint artifact missing: {required_path}")
    if candidate.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise AdapterError("Candidate checkpoint digest drift")
    if candidate.get("checkpoint_class") != EXPECTED_CANDIDATE_CLASS:
        raise AdapterError("Candidate checkpoint class drift")
    if candidate.get("checkpoint_parameter_count") != EXPECTED_CANDIDATE_PARAMS:
        raise AdapterError("Candidate checkpoint parameter count drift")
    if candidate.get("checkpoint_q_shape") != EXPECTED_Q_SHAPE:
        raise AdapterError("Candidate checkpoint Q shape drift")
    if candidate.get("checkpoint_q_dtype") != EXPECTED_Q_DTYPE:
        raise AdapterError("Candidate checkpoint Q dtype drift")
    q_digest = candidate.get("learned_q_flat_float32_sha256")
    if not isinstance(q_digest, str) or len(q_digest) != 64:
        raise AdapterError("Candidate Q digest missing")
    metadata = json_object(metadata_path, "best-val-accuracy metadata")
    checkpoint_epoch = metadata.get("epoch")
    validation_metrics = metadata.get("validation_metrics")
    if isinstance(checkpoint_epoch, bool) or not isinstance(checkpoint_epoch, int):
        raise AdapterError("Checkpoint metadata epoch invalid")
    if not isinstance(validation_metrics, dict):
        raise AdapterError("Checkpoint validation metrics missing")
    derived = derive_registered_metrics(history)
    if checkpoint_epoch != derived["candidate_best_val_accuracy_epoch"]:
        raise AdapterError(
            "Checkpoint metadata epoch is not the earliest complete-history "
            "global maximum validation-accuracy epoch"
        )
    if float(validation_metrics.get("accuracy")) != derived[
        "candidate_best_val_accuracy"
    ]:
        raise AdapterError(
            "Checkpoint metadata accuracy is not the complete-history global maximum"
        )
    selected_row = next((row for row in rows if row["epoch"] == checkpoint_epoch), None)
    if selected_row is None:
        raise AdapterError("Checkpoint epoch absent from complete history")
    for key, history_key in (
        ("accuracy", "val_accuracy"),
        ("macro_f1", "val_macro_f1"),
        ("loss", "val_loss"),
    ):
        if float(validation_metrics.get(key)) != float(selected_row[history_key]):
            raise AdapterError(f"Checkpoint metadata/history metric drift: {key}")
    derived["checkpoint_metadata"] = {
        "epoch": checkpoint_epoch,
        "validation_accuracy": float(validation_metrics["accuracy"]),
        "validation_macro_f1": float(validation_metrics["macro_f1"]),
        "validation_loss": float(validation_metrics["loss"]),
        "selected_checkpoint_macro_f1_difference_vs_baseline_epoch31": (
            float(validation_metrics["macro_f1"])
            - BASELINE_EPOCH31_VAL_MACRO_F1
        ),
        "checkpoint_sha256": sha256_file(checkpoint),
        "weights_sha256": sha256_file(weights),
        "metadata_sha256": sha256_file(metadata_path),
        "learned_q_flat_float32_sha256": q_digest,
    }
    return derived


def write_success_outputs(
    *,
    derived: dict[str, Any],
    wrapper_path: str | Path,
    evidence_path: str | Path,
    report_path: str | Path,
    subprocess_return_code: int,
    source_hashes_before: dict[str, str],
    source_hashes_after: dict[str, str],
) -> dict[str, Any]:
    wrapper = {
        "schema_version": 1,
        "status": "COMPLETE",
        "subprocess_return_code": int(subprocess_return_code),
        "error_text": None,
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
        "scientific_result_valid": True,
        "scientific_interpretation": derived["registered_decision"],
        "training_invoked": True,
        "training_validation_completed": True,
        "test_access": False,
        "automatic_retry": False,
    }
    evidence = {
        "schema_version": 1,
        "issue": 31,
        "execution_commit": EXECUTION_COMMIT,
        "scientific_result_valid": True,
        "scientific_interpretation": derived["registered_decision"],
        "primary_endpoint": derived,
        "baseline_comparator": {
            "best_val_macro_f1": BASELINE_BEST_VAL_MACRO_F1,
            "best_val_macro_epoch": BASELINE_BEST_VAL_MACRO_EPOCH,
            "train_macro_f1_at_best_macro_epoch": BASELINE_TRAIN_MACRO_AT_BEST,
            "train_validation_macro_gap_pp": BASELINE_TRAIN_VAL_GAP_PP,
            "best_val_accuracy": BASELINE_BEST_VAL_ACCURACY,
            "best_val_accuracy_epoch": BASELINE_BEST_VAL_ACCURACY_EPOCH,
            "epoch31_val_macro_f1": BASELINE_EPOCH31_VAL_MACRO_F1,
            "best_val_loss": BASELINE_BEST_VAL_LOSS,
            "best_val_loss_epoch": BASELINE_BEST_VAL_LOSS_EPOCH,
        },
        "test_access": False,
    }
    atomic_json(wrapper_path, wrapper)
    atomic_json(evidence_path, evidence)
    report = textwrap.dedent(
        f"""\
        # TensorFlow Step 12 seed42 learned local residual-slot candidate

        - Status: `COMPLETE`.
        - Scientific result valid: `true`.
        - Registered single-seed label: `{derived['registered_decision']}`.
        - Candidate best validation macro-F1: `{derived['candidate_best_val_macro_f1']}` at epoch `{derived['candidate_best_val_macro_epoch']}`.
        - Delta versus locked baseline: `{derived['delta_macro_pp']}` pp.
        - Test access: `false`.

        This is a preregistered single-seed validation signal, not a significance
        test or evidence of universal architectural superiority. It does not
        authorize test-split access.
        """
    )
    Path(report_path).write_text(report, encoding="utf-8", newline="\n")
    return wrapper


def write_failure_outputs(
    *,
    wrapper_path: str | Path,
    evidence_path: str | Path,
    failure_report_path: str | Path,
    subprocess_return_code: int | None,
    error_text: str,
    source_hashes_before: dict[str, str],
    source_hashes_after: dict[str, str] | None,
) -> dict[str, Any]:
    final_evidence = Path(evidence_path)
    if final_evidence.exists():
        final_evidence.unlink()
    wrapper = {
        "schema_version": 1,
        "status": "TECHNICAL_OR_RUNTIME_FAILURE",
        "subprocess_return_code": subprocess_return_code,
        "error_text": str(error_text),
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
        "scientific_result_valid": False,
        "scientific_interpretation": None,
        "training_invoked": True,
        "training_validation_completed": False,
        "test_access": False,
        "automatic_retry": False,
    }
    atomic_json(wrapper_path, wrapper)
    report = textwrap.dedent(
        f"""\
        # TensorFlow Step 12 technical/runtime failure

        - Status: `TECHNICAL_OR_RUNTIME_FAILURE`.
        - Subprocess return code: `{subprocess_return_code}`.
        - Scientific result valid: `false`.
        - Scientific interpretation: `null`.
        - Error: `{error_text}`.

        Partial trajectories and checkpoints are diagnostic only. No registered
        candidate-performance label may be derived without both valid completion
        markers and every Issue #31 validity gate.
        """
    )
    Path(failure_report_path).write_text(report, encoding="utf-8", newline="\n")
    return wrapper


def run_subprocess_once(
    command: Sequence[str],
    *,
    cwd: str | Path,
    log_path: str | Path,
    monitor: RollingArchiveMonitor,
    popen_factory=subprocess.Popen,
) -> int:
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor.run,
        args=(stop_event,),
        name="issue31-read-only-rolling-archive-monitor",
        daemon=True,
    )
    destination = Path(log_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n", buffering=1) as log:
        process = popen_factory(
            [str(item) for item in command],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        monitor_thread.start()
        try:
            if process.stdout is None:
                raise AdapterError("Candidate subprocess stdout pipe unavailable")
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = process.wait()
        finally:
            stop_event.set()
            monitor_thread.join(timeout=max(10.0, monitor.poll_seconds * 3.0))
    return int(return_code)


def run_registered_adapter() -> dict[str, Any]:
    """Execute exactly one candidate harness subprocess and publish evidence."""

    if ARCHIVE_PATH.exists() or REPORT_PATH.exists():
        raise FileExistsError("Fresh Issue #31 output paths are required")
    RUN_ROOT.mkdir(parents=True, exist_ok=False)
    ADAPTER_ROOT.mkdir(parents=True, exist_ok=False)
    source_before = verify_source_locks(CHECKOUT_ROOT)
    command = registered_command(CHECKOUT_ROOT, TRAIN_OUTPUT_ROOT)
    manifest = {
        "schema_version": 1,
        "issue": 31,
        "execution_commit": EXECUTION_COMMIT,
        "source_locks": source_before,
        "scientific_payload_sha256": SCIENTIFIC_PAYLOAD_SHA256,
        "inherited_execution_contract_sha256": BASELINE_EXECUTION_CONTRACT_SHA256,
        "command": command,
        "candidate_harness_invocations": 1,
        "direct_frozen_trainer_invocations": 0,
        "automatic_retry": False,
        "seed": 42,
        "resources": {
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "graph_workers": GRAPH_WORKERS,
            "tf_data_prefetch": TF_DATA_PREFETCH,
            "tf_data_parallel_calls": TF_DATA_PARALLEL_CALLS,
            "graph_cache_size": GRAPH_CACHE_SIZE,
            "mixed_precision": True,
            "xla": False,
            "memory_growth": True,
            "resume": False,
            "op_determinism_changed": False,
            "bounded_limits": None,
        },
        "test_access": False,
    }
    atomic_json(PRE_RUN_MANIFEST_PATH, manifest)
    SUBPROCESS_LOG_PATH.touch()
    archive_lock = threading.Lock()
    monitor = RollingArchiveMonitor(
        history_path=TRAIN_OUTPUT_ROOT / "history.json",
        progress_path=RUNTIME_PROGRESS_PATH,
        archive_path=ARCHIVE_PATH,
        run_root=RUN_ROOT,
        train_output_root=TRAIN_OUTPUT_ROOT,
        report_path=REPORT_PATH,
        archive_lock=archive_lock,
    )
    with archive_lock:
        publish_archive_atomic(ARCHIVE_PATH, RUN_ROOT, TRAIN_OUTPUT_ROOT, REPORT_PATH)
    return_code: int | None = None
    source_after: dict[str, str] | None = None
    try:
        return_code = run_subprocess_once(
            command,
            cwd=CHECKOUT_ROOT,
            log_path=SUBPROCESS_LOG_PATH,
            monitor=monitor,
        )
        source_after = verify_source_locks(CHECKOUT_ROOT)
        dirty = run_checked(["git", "status", "--porcelain"], cwd=CHECKOUT_ROOT).strip()
        if dirty:
            raise AdapterError("Execution checkout became dirty")
        if return_code != 0:
            raise AdapterError(f"Candidate harness exited with code {return_code}")
        derived = validate_completion(TRAIN_OUTPUT_ROOT, source_before, source_after)
        wrapper = write_success_outputs(
            derived=derived,
            wrapper_path=WRAPPER_EXECUTION_PATH,
            evidence_path=FINAL_EVIDENCE_PATH,
            report_path=REPORT_PATH,
            subprocess_return_code=return_code,
            source_hashes_before=source_before,
            source_hashes_after=source_after,
        )
    except BaseException as exc:
        if source_after is None:
            try:
                source_after = verify_source_locks(CHECKOUT_ROOT)
            except BaseException:
                source_after = None
        wrapper = write_failure_outputs(
            wrapper_path=WRAPPER_EXECUTION_PATH,
            evidence_path=FINAL_EVIDENCE_PATH,
            failure_report_path=FAILURE_REPORT_PATH,
            subprocess_return_code=return_code,
            error_text=f"{type(exc).__name__}: {exc}",
            source_hashes_before=source_before,
            source_hashes_after=source_after,
        )
    with archive_lock:
        names = publish_archive_atomic(
            ARCHIVE_PATH, RUN_ROOT, TRAIN_OUTPUT_ROOT, REPORT_PATH
        )
    wrapper["archive_path"] = str(ARCHIVE_PATH)
    wrapper["archive_sha256"] = sha256_file(ARCHIVE_PATH)
    wrapper["archive_members"] = len(names)
    atomic_json(WRAPPER_EXECUTION_PATH, wrapper)
    with archive_lock:
        names = publish_archive_atomic(
            ARCHIVE_PATH, RUN_ROOT, TRAIN_OUTPUT_ROOT, REPORT_PATH
        )
    print(json.dumps(wrapper, indent=2, sort_keys=True), flush=True)
    return wrapper


def _source(text: str) -> list[str]:
    return textwrap.dedent(text).lstrip("\n").splitlines(keepends=True)


def _markdown(text: str, cell_id: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": _source(text)}


def _code(text: str, cell_id: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


def _runtime_definitions_source() -> str:
    objects = (
        AdapterError,
        sha256_file,
        atomic_json,
        json_object,
        run_checked,
        verify_source_locks,
        registered_decision,
        registered_command,
        archive_members,
        publish_archive_atomic,
        RollingArchiveMonitor,
        _validated_rows,
        derive_registered_metrics,
        validate_completion,
        write_success_outputs,
        write_failure_outputs,
        run_subprocess_once,
        run_registered_adapter,
    )
    return "\n\n".join(inspect.getsource(item).rstrip() for item in objects) + "\n"


def _constants_source() -> str:
    names = (
        "REPOSITORY_URL",
        "EXECUTION_COMMIT",
        "SCIENTIFIC_PAYLOAD_SHA256",
        "BASELINE_EXECUTION_CONTRACT_SHA256",
        "SOURCE_LOCKS",
        "EXPECTED_PYTHON",
        "EXPECTED_TENSORFLOW",
        "EXPECTED_KERAS",
        "EXPECTED_GPU_COUNT",
        "EXPECTED_GPU_TOKEN",
        "EXPECTED_CANDIDATE_CLASS",
        "EXPECTED_CANDIDATE_PARAMS",
        "EXPECTED_VARIABLE_COUNT",
        "EXPECTED_Q_INDEX",
        "EXPECTED_Q_SHAPE",
        "EXPECTED_Q_DTYPE",
        "BASELINE_BEST_VAL_MACRO_F1",
        "BASELINE_BEST_VAL_MACRO_EPOCH",
        "BASELINE_TRAIN_MACRO_AT_BEST",
        "BASELINE_TRAIN_VAL_GAP_PP",
        "BASELINE_BEST_VAL_ACCURACY",
        "BASELINE_BEST_VAL_ACCURACY_EPOCH",
        "BASELINE_EPOCH31_VAL_MACRO_F1",
        "BASELINE_BEST_VAL_LOSS",
        "BASELINE_BEST_VAL_LOSS_EPOCH",
        "PRACTICAL_EFFECT_THRESHOLD_PP",
        "FER_ROOT",
        "FER_TRAIN_CSV",
        "FER_VAL_CSV",
        "PRIOR_ROOT",
        "CACHE_ROOT",
        "WORKING_ROOT",
        "CHECKOUT_ROOT",
        "RUN_ROOT",
        "TRAIN_OUTPUT_ROOT",
        "ADAPTER_ROOT",
        "PRE_RUN_MANIFEST_PATH",
        "SUBPROCESS_LOG_PATH",
        "WRAPPER_EXECUTION_PATH",
        "FINAL_EVIDENCE_PATH",
        "RUNTIME_PROGRESS_PATH",
        "FAILURE_REPORT_PATH",
        "REPORT_PATH",
        "ARCHIVE_PATH",
        "TRAIN_BATCH_SIZE",
        "EVAL_BATCH_SIZE",
        "GRAPH_WORKERS",
        "TF_DATA_PREFETCH",
        "TF_DATA_PARALLEL_CALLS",
        "GRAPH_CACHE_SIZE",
        "EXPECTED_SPLIT_COUNTS",
        "FROZEN_MARKER",
        "CANDIDATE_MARKER",
        "FORBIDDEN_OUTPUT_NAMES",
        "ROLLING_RELATIVE_FILES",
    )
    lines = []
    for name in names:
        value = globals()[name]
        if isinstance(value, Path):
            lines.append(f"{name} = Path({value.as_posix()!r})")
        elif isinstance(value, set):
            lines.append(f"{name} = set({sorted(value)!r})")
        elif isinstance(value, tuple) and all(isinstance(item, str) for item in value):
            lines.append(f"{name} = {value!r}")
        else:
            lines.append(f"{name} = {value!r}")
    return "\n".join(lines) + "\n"


def build_notebook() -> dict[str, Any]:
    imports = textwrap.dedent("""\
    from __future__ import annotations
    import csv
    import hashlib
    import importlib.metadata
    import importlib.util
    import json
    import os
    from pathlib import Path
    import platform
    import shutil
    import subprocess
    import sys
    import textwrap
    import threading
    import time
    import zipfile
    sys.dont_write_bytecode = True
    """)
    clone_cell = """\
    if CHECKOUT_ROOT.exists():
        raise FileExistsError(f"Fresh detached checkout required: {CHECKOUT_ROOT}")
    run_checked(["git", "clone", "--no-checkout", REPOSITORY_URL, CHECKOUT_ROOT])
    run_checked(["git", "checkout", "--detach", EXECUTION_COMMIT], cwd=CHECKOUT_ROOT)
    actual_commit = run_checked(["git", "rev-parse", "HEAD"], cwd=CHECKOUT_ROOT).strip()
    dirty = run_checked(["git", "status", "--porcelain"], cwd=CHECKOUT_ROOT).strip()
    if actual_commit != EXECUTION_COMMIT or dirty:
        raise AdapterError(f"Detached source lock failed: {actual_commit}, dirty={bool(dirty)}")
    source_hashes = verify_source_locks(CHECKOUT_ROOT)
    package_root = CHECKOUT_ROOT / "standalone/lap_gnn_tensorflow_ofix7_mid_candidate"
    run_checked([sys.executable, "-B", package_root / "tools/verify_checksums.py"], cwd=package_root)
    print(json.dumps({"commit": actual_commit, "source_hashes": source_hashes}, indent=2))
    """
    preflight_cell = """\
    if platform.python_version() != EXPECTED_PYTHON:
        raise AdapterError(f"Python runtime drift: {platform.python_version()}")
    gpu_names = [
        line.strip() for line in run_checked(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
        ).splitlines() if line.strip()
    ]
    if len(gpu_names) != EXPECTED_GPU_COUNT or not all(
        EXPECTED_GPU_TOKEN in name for name in gpu_names
    ):
        raise AdapterError(f"Issue #31 requires exactly two Tesla T4 GPUs: {gpu_names}")
    if "tensorflow" in sys.modules or "keras" in sys.modules:
        raise AdapterError("TensorFlow/Keras imported before exact-version verification")
    def installed_version(distribution):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return None
    installed_tf = installed_version("tensorflow")
    installed_keras = installed_version("keras")
    if installed_tf != EXPECTED_TENSORFLOW or installed_keras != EXPECTED_KERAS:
        requirements = package_root / "requirements-kaggle.txt"
        run_checked([sys.executable, "-m", "pip", "install", "-q", "--no-warn-conflicts", "-r", requirements])
        importlib.invalidate_caches()
    missing_runtime = [
        requirement for module, requirement in (
            ("yaml", "PyYAML==6.0.2"),
            ("sklearn", "scikit-learn==1.6.1"),
            ("psutil", "psutil==6.1.1"),
            ("matplotlib", "matplotlib==3.10.0"),
        ) if importlib.util.find_spec(module) is None
    ]
    if missing_runtime:
        run_checked([sys.executable, "-m", "pip", "install", "-q", *missing_runtime])
        importlib.invalidate_caches()
    import tensorflow as tf
    if tf.__version__ != EXPECTED_TENSORFLOW or tf.keras.__version__ != EXPECTED_KERAS:
        raise AdapterError(f"TensorFlow/Keras runtime drift: {tf.__version__}/{tf.keras.__version__}")
    if len(tf.config.list_physical_devices("GPU")) != EXPECTED_GPU_COUNT:
        raise AdapterError("TensorFlow does not expose exactly two registered GPUs")
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
        csv_path = {"train": FER_TRAIN_CSV, "val": FER_VAL_CSV}[split]
        with csv_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            header = [value.strip().lower() for value in next(reader)]
            row_count = sum(1 for _ in reader)
        if row_count != expected_count or not {"emotion", "pixels"}.issubset(header):
            raise AdapterError(f"Registered {split} CSV drift")
        prior_files = len(list((PRIOR_ROOT / split).glob("*.npz")))
        if prior_files != expected_count:
            raise AdapterError(f"Registered {split} prior count drift")
        cache_index_path = CACHE_ROOT / split / "index.json"
        cache_index = json_object(cache_index_path, f"{split} cache index")
        if cache_index.get("schema_version") != "tf_clean_graph_cache_v2_records":
            raise AdapterError(f"Registered {split} cache schema drift")
        if cache_index.get("sample_count") != expected_count:
            raise AdapterError(f"Registered {split} cache count drift")
        for shard in cache_index.get("shards", []):
            shard_path = CACHE_ROOT / split / shard["path"]
            if not shard_path.is_file():
                raise AdapterError(f"Registered cache shard missing: {shard_path}")
    cache_complete = json_object(CACHE_ROOT / "CACHE_COMPLETE.json", "shared cache aggregate marker")
    if cache_complete.get("schema_version") != "tf_clean_graph_cache_v2_records":
        raise AdapterError("Shared cache aggregate marker schema drift")

    package_src = package_root / "src"
    for path in (CHECKOUT_ROOT, package_src):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from lap_gnn_tf.graph.batch import load_golden_batch
    from research.candidates.tf_learned_local_residual_slots.model import LearnedLocalResidualSlotLapGNN
    golden = load_golden_batch(str(package_root / "validation_assets/golden/graph_batch.npz"))
    candidate = LearnedLocalResidualSlotLapGNN()
    candidate(golden, training=False)
    q = candidate.learned_local_residual_slots.Q
    if type(candidate).__name__ != EXPECTED_CANDIDATE_CLASS:
        raise AdapterError("Candidate exact class drift")
    if candidate.count_params() != EXPECTED_CANDIDATE_PARAMS:
        raise AdapterError("Candidate parameter count drift")
    if len(candidate.trainable_variables) != EXPECTED_VARIABLE_COUNT:
        raise AdapterError("Candidate variable count drift")
    if candidate.trainable_variables[EXPECTED_Q_INDEX] is not q:
        raise AdapterError("Candidate Q ordered identity drift")
    if q.shape.as_list() != EXPECTED_Q_SHAPE or str(q.dtype) != EXPECTED_Q_DTYPE:
        raise AdapterError("Candidate Q shape/dtype drift")
    del candidate, golden
    dirty = run_checked(["git", "status", "--porcelain"], cwd=CHECKOUT_ROOT).strip()
    if dirty:
        raise AdapterError("Preflight altered the detached execution checkout")
    print(json.dumps({
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "keras": tf.keras.__version__,
        "gpu_names": gpu_names,
        "allowed_sample_splits": ["train", "val"],
        "shared_cache_aggregate_metadata": True,
        "test_access": False,
        "candidate_class": EXPECTED_CANDIDATE_CLASS,
        "candidate_params": EXPECTED_CANDIDATE_PARAMS,
        "candidate_variables": EXPECTED_VARIABLE_COUNT,
        "q": {"index": EXPECTED_Q_INDEX, "shape": EXPECTED_Q_SHAPE, "dtype": EXPECTED_Q_DTYPE},
    }, indent=2))
    """
    notebook = {
        "cells": [
            _markdown(
                """
                # Issue #31: seed42 learned local residual-slot candidate adapter

                This is the unexecuted preregistered Step 12 pre-run adapter. It
                performs exactly one future candidate validation-only training run
                after research-lead approval. It never runs final-test inference.

                Required Kaggle Inputs:

                - FER: `/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split`;
                  resolved sample files are `train.csv` and `val.csv` only.
                - MediaPipe priors:
                  `/kaggle/input/datasets/irthn1311/d16-mediapipe-pixel-priors-best-retry-rescue/outputs/d16_mediapipe_pixel_priors_best_retry_rescue`;
                  resolved record directories are `train/` and `val/` only.
                - Clean graph cache:
                  `/kaggle/input/datasets/irthn1311/ofix7-mid-seed42-records`;
                  resolved sample indexes/shards are `train/` and `val/`, plus the
                  shared non-sample aggregate `CACHE_COMPLETE.json` required by the
                  frozen loader.

                Internet is required to clone the exact Git commit and only if the
                Kaggle image needs the pinned TensorFlow/Keras dependencies. All
                FER, prior, and cache assets are offline Kaggle Inputs.

                Rolling/final output:
                `/kaggle/working/tf_step12_learned_local_residual_slots_seed42_kaggle_t4.zip`.
                """,
                "issue31-00",
            ),
            _markdown("## 1. Locked protocol and failure-safe helpers\n", "issue31-01"),
            _code(imports + "\n" + _constants_source() + "\n" + _runtime_definitions_source(), "issue31-02"),
            _markdown("## 2. Exact detached source checkout and SHA verification\n", "issue31-03"),
            _code(clone_cell, "issue31-04"),
            _markdown("## 3. Registered Kaggle environment, inputs, and candidate identity\n", "issue31-05"),
            _code(preflight_cell, "issue31-06"),
            _markdown("## 4. Exactly one registered candidate validation-only subprocess\n", "issue31-07"),
            _code(
                "wrapper_execution = run_registered_adapter()\n"
                "if not ARCHIVE_PATH.is_file():\n"
                "    raise AdapterError('Failure-safe archive was not published')\n"
                "print('archive:', ARCHIVE_PATH)\n"
                "print('status:', wrapper_execution['status'])\n",
                "issue31-08",
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": EXPECTED_PYTHON},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return notebook


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(
        json.dumps(build_notebook(), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
