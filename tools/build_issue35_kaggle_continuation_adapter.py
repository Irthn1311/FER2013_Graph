"""Build the deterministic Issue #35 Kaggle continuation adapter notebook.

The imported builder is standard-library only. TensorFlow is imported only by
the generated Kaggle notebook or by the reviewed continuation harness.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import textwrap
import threading
from typing import Any, Callable, Sequence
import zipfile


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks/kaggle-issue35-step12c-checkpoint-continuation.ipynb"
REPOSITORY_URL = "https://github.com/Irthn1311/FER2013_Graph.git"
EXECUTION_COMMIT = "0f4fde1d4e6645096711a800509f4db2deedf38f"
SCIENTIFIC_PAYLOAD_SHA256 = "286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e"
BASELINE_EXECUTION_CONTRACT_SHA256 = "14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22"
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
    "continuation_harness": (
        "research/candidates/tf_learned_local_residual_slots/resume_validation_only.py",
        "dba0d749b9a8e05b3cd67dad0749ef4235fc06f2a389b552229c76f691edde40",
    ),
    "step12c_evidence": (
        "research/evidence/tf_step12c_checkpoint_continuation_harness.md",
        "4e48652c4c75cbdcf985b596e04c5658483825ab1e6900f97d52d1cf7ee7f29f",
    ),
    "seed42_config": (
        "standalone/lap_gnn_tensorflow_ofix7_mid_candidate/configs/fer2013_ofix7_mid_tensorflow_seed42.yaml",
        "aa3bf2d3932bbad6c5f8cdcc347f4a9866e2c027d6135a60b5002a8f6a3b6908",
    ),
}

EXPECTED_PYTHON = "3.12.12"
EXPECTED_TENSORFLOW = "2.18.1"
EXPECTED_KERAS = "3.15.0"
EXPECTED_CUDA = "12.5.1"
EXPECTED_CUDNN_MAJOR = "9"
EXPECTED_GPU_COUNT = 2
EXPECTED_GPU_TOKEN = "T4"
EXPECTED_CANDIDATE_CLASS = "LearnedLocalResidualSlotLapGNN"
EXPECTED_CANDIDATE_PARAMS = 1_061_576
EXPECTED_VARIABLE_COUNT = 128
EXPECTED_Q_INDEX = 127
EXPECTED_Q_SHAPE = [4, 96]
EXPECTED_Q_DTYPE = "float32"
EXPECTED_RESUME_Q_SHA256 = "166f6e09191f94c52c17af81c2d9ba357c765b2077aab5fc809563a9de6d6270"
EXPECTED_RESUME_OPTIMIZER_ITERATIONS = 53_822
EXPECTED_RESUME_OPTIMIZER_VARIABLES = 262
EXPECTED_RESUME_LR = 0.0001500000071246177

SOURCE_ARCHIVE_NAME = "tf_step12_learned_local_residual_slots_seed42_kaggle_t4.zip"
SOURCE_ARCHIVE_SHA256 = "2ada6cfd1ce1c07f6d7ae36264a1f14840a0936e9448a72e6bb464ae6ab71357"
SOURCE_TRANSPORT_NAME = f"{SOURCE_ARCHIVE_NAME}.b64"
SOURCE_TRANSPORT_SHA256 = "66bc813bd3e3dcc38a1dd4c0c36e41ddb794831895f15e099cec566d1ad51b8d"
SOURCE_HISTORY_SHA256 = "0a2edffbc595f09660e01ccacc5338656aef06892949aad4a9e209aac280789c"
SOURCE_CHECKPOINT_SHA256 = "818450d56cb480cf08637bee01061e8028a3d58c0f13346716618f0ee186d932"
CONTINUATION_PROTOCOL_ID = "tf-step12c-checkpoint-conditioned-continuation-v1"
CONTINUATION_ROW_ORIGIN = "CHECKPOINT_CONDITIONED_CONTINUATION"
BASELINE_BEST_VAL_MACRO_F1 = 0.601166548701511
PRACTICAL_EFFECT_THRESHOLD_PP = 1.0

FER_ROOT = Path("/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split")
FER_TRAIN_CSV = FER_ROOT / "train.csv"
FER_VAL_CSV = FER_ROOT / "val.csv"
PRIOR_ROOT = Path("/kaggle/input/datasets/irthn1311/d16-mediapipe-pixel-priors-best-retry-rescue/outputs/d16_mediapipe_pixel_priors_best_retry_rescue")
CACHE_ROOT = Path("/kaggle/input/datasets/irthn1311/ofix7-mid-seed42-records")
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")
MATERIALIZED_SOURCE_ARCHIVE_PATH = WORKING_ROOT / SOURCE_ARCHIVE_NAME
CHECKOUT_ROOT = WORKING_ROOT / "FER2013_Graph_issue35"
RUN_ROOT = WORKING_ROOT / "tf_step12c_checkpoint_continuation"
TRAIN_OUTPUT_ROOT = RUN_ROOT / "run"
ADAPTER_ROOT = RUN_ROOT / "adapter"
PRE_RUN_MANIFEST_PATH = ADAPTER_ROOT / "pre_run_manifest.json"
SUBPROCESS_LOG_PATH = ADAPTER_ROOT / "subprocess.log"
WRAPPER_EXECUTION_PATH = ADAPTER_ROOT / "wrapper_execution.json"
FINAL_EVIDENCE_PATH = ADAPTER_ROOT / "final_evidence.json"
RUNTIME_PROGRESS_PATH = ADAPTER_ROOT / "runtime_progress.json"
FAILURE_REPORT_PATH = ADAPTER_ROOT / "technical_continuation_failure.md"
REPORT_PATH = WORKING_ROOT / "tf_step12c_checkpoint_continuation.md"
ARCHIVE_PATH = WORKING_ROOT / "tf_step12c_checkpoint_continuation_kaggle_t4.zip"

TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
GRAPH_WORKERS = 2
TF_DATA_PREFETCH = 2
TF_DATA_PARALLEL_CALLS = 1
GRAPH_CACHE_SIZE = 64
MAX_EPOCHS = 90
EARLY_STOPPING_PATIENCE = 15
EXPECTED_SPLIT_COUNTS = {"train": 28_709, "val": 3_589}
COMPLETION_MARKER = "CHECKPOINT_CONTINUATION_VALIDATION_ONLY_COMPLETE.json"
TECHNICAL_FAILURE_MARKER = "CHECKPOINT_CONTINUATION_TECHNICAL_FAILURE.json"
LATEST_STATE_MANIFEST = "latest_state_manifest.json"
OVERLAP_SOURCE = "FIRST_RUN_OVERLAP_DIAGNOSTICS.json"
FORBIDDEN_OUTPUT_NAMES = {
    "TRAINING_COMPLETE.json", "run_summary.json", "predictions.csv",
    "per_class_metrics.csv", "confusion_matrix.csv", "confusion_matrix.png",
}
ROLLING_RELATIVE_FILES = (
    "continuation_pre_run_manifest.json", "pretrain_validation_gate.json",
    "resolved_config.json", "resolved_config.yaml", "provenance.json",
    "history.json", "train_log.csv", "latest_epoch_summary.json",
    "runtime_progress.json", "telemetry.json", OVERLAP_SOURCE,
    "resume_overlap_epoch31.json", "resume_overlap_epoch32.json",
    "checkpoints/best_val_accuracy.keras",
    "checkpoints/best_val_accuracy.weights.h5",
    "checkpoints/best_val_accuracy.metadata.json", LATEST_STATE_MANIFEST,
    COMPLETION_MARKER, TECHNICAL_FAILURE_MARKER,
)


class AdapterError(RuntimeError):
    """Issue #35 evidence failed closed."""


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    os.replace(temporary, destination)


def json_object(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdapterError(f"Malformed or unreadable {label}: {source}") from exc
    if not isinstance(value, dict):
        raise AdapterError(f"{label} must be a JSON object")
    return value


def run_checked(command: Sequence[Any], *, cwd: str | Path | None = None) -> str:
    actual = [str(item) for item in command]
    result = subprocess.run(
        actual, cwd=None if cwd is None else str(cwd), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.returncode:
        raise AdapterError(f"Command failed ({result.returncode}): {actual}")
    return result.stdout


def verify_source_locks(checkout_root: str | Path) -> dict[str, str]:
    root = Path(checkout_root)
    actual = {}
    for label, (relative, expected) in SOURCE_LOCKS.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise AdapterError(f"Locked source drift: {label}")
        actual[label] = expected
    manifest = json_object(
        root / "standalone/lap_gnn_tensorflow_ofix7_mid_candidate/package_manifest.json",
        "frozen package manifest",
    )
    if manifest.get("scientific_payload_sha256") != SCIENTIFIC_PAYLOAD_SHA256:
        raise AdapterError("Frozen scientific payload drift")
    if manifest.get("execution_contract_sha256") != BASELINE_EXECUTION_CONTRACT_SHA256:
        raise AdapterError("Frozen execution contract drift")
    return actual


def discover_source_transport(input_root: str | Path) -> Path:
    root = Path(input_root)
    matches = sorted(root.glob(f"**/{SOURCE_ARCHIVE_NAME}"))
    matches.extend(sorted(root.glob(f"**/{SOURCE_TRANSPORT_NAME}")))
    if len(matches) != 1:
        raise AdapterError(f"Expected exactly one censored source transport; found {len(matches)}")
    source = matches[0].resolve()
    expected = (
        SOURCE_ARCHIVE_SHA256
        if source.name == SOURCE_ARCHIVE_NAME
        else SOURCE_TRANSPORT_SHA256
    )
    if sha256_file(source) != expected:
        raise AdapterError("Censored source transport SHA drift")
    return source


def materialize_source_archive(
    source_transport: str | Path, materialized_path: str | Path
) -> Path:
    source = Path(source_transport).resolve()
    if source.name == SOURCE_ARCHIVE_NAME:
        if sha256_file(source) != SOURCE_ARCHIVE_SHA256:
            raise AdapterError("Censored source archive SHA drift")
        return source
    if source.name != SOURCE_TRANSPORT_NAME:
        raise AdapterError("Unregistered censored source transport")
    if sha256_file(source) != SOURCE_TRANSPORT_SHA256:
        raise AdapterError("Censored source transport SHA drift")
    destination = Path(materialized_path).resolve()
    if destination.exists():
        raise FileExistsError(f"Fresh materialized source path required: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        try:
            payload = base64.b64decode(source.read_bytes(), validate=True)
        except (OSError, ValueError) as exc:
            raise AdapterError("Malformed Base64 source transport") from exc
        temporary.write_bytes(payload)
        if sha256_file(temporary) != SOURCE_ARCHIVE_SHA256:
            raise AdapterError("Decoded censored source archive SHA drift")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def discover_source_archive(
    input_root: str | Path,
    materialized_path: str | Path = MATERIALIZED_SOURCE_ARCHIVE_PATH,
) -> Path:
    source = discover_source_transport(input_root)
    archive = materialize_source_archive(source, materialized_path)
    if sha256_file(archive) != SOURCE_ARCHIVE_SHA256:
        raise AdapterError("Censored source archive SHA drift")
    return archive


def locked_source_history(source_archive: str | Path) -> list[dict[str, Any]]:
    path = Path(source_archive)
    if sha256_file(path) != SOURCE_ARCHIVE_SHA256:
        raise AdapterError("Censored source archive SHA drift")
    try:
        with zipfile.ZipFile(path) as archive:
            payload = archive.read("run/history.json")
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise AdapterError("Locked source history unavailable") from exc
    if hashlib.sha256(payload).hexdigest() != SOURCE_HISTORY_SHA256:
        raise AdapterError("Locked source history SHA drift")
    history = json.loads(payload)
    rows = history.get("epochs") if isinstance(history, dict) else None
    if not isinstance(rows, list) or len(rows) < 32:
        raise AdapterError("Locked source history is incomplete")
    if [row.get("epoch") for row in rows] != list(range(1, len(rows) + 1)):
        raise AdapterError("Locked source history epochs drift")
    return rows


def registered_decision(delta_macro_pp: float) -> str:
    if (
        isinstance(delta_macro_pp, bool)
        or not isinstance(delta_macro_pp, (int, float))
        or not math.isfinite(float(delta_macro_pp))
    ):
        raise AdapterError("Registered decision requires a finite numerical delta")
    if float(delta_macro_pp) >= PRACTICAL_EFFECT_THRESHOLD_PP:
        return "CHECKPOINT_CONTINUATION_PROMISING_SINGLE_SEED_VALIDATION_GAIN"
    if float(delta_macro_pp) <= -PRACTICAL_EFFECT_THRESHOLD_PP:
        return "CHECKPOINT_CONTINUATION_SINGLE_SEED_VALIDATION_REGRESSION"
    return "CHECKPOINT_CONTINUATION_NO_CLEAR_SINGLE_SEED_DIFFERENCE"


def registered_command(
    checkout_root: str | Path, output_root: str | Path, source_archive: str | Path
) -> list[str]:
    root = Path(checkout_root)
    command = [
        sys.executable, "-B", str(root / SOURCE_LOCKS["continuation_harness"][0]),
        "--config", str(root / SOURCE_LOCKS["seed42_config"][0]),
        "--fer-csv", str(FER_TRAIN_CSV), "--prior-root", str(PRIOR_ROOT),
        "--source-archive", str(Path(source_archive)),
        "--output-root", str(Path(output_root)), "--device", "gpu",
        "--graph-workers", str(GRAPH_WORKERS),
        "--tf-data-prefetch", str(TF_DATA_PREFETCH),
        "--tf-data-parallel-calls", str(TF_DATA_PARALLEL_CALLS),
        "--graph-cache-size", str(GRAPH_CACHE_SIZE),
        "--clean-graph-cache-dir", str(CACHE_ROOT),
        "--batch-size", str(TRAIN_BATCH_SIZE),
        "--eval-batch-size", str(EVAL_BATCH_SIZE),
        "--mixed-precision", "--no-xla", "--memory-growth",
    ]
    if any(item.startswith("--limit-") for item in command):
        raise AdapterError("Registered command contains a limit")
    return command


def _canonical_json_sha256(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _resolve_under(root: Path, relative: Any, label: str) -> Path:
    candidate = Path(relative) if isinstance(relative, str) else Path("..")
    if candidate.is_absolute() or ".." in candidate.parts:
        raise AdapterError(f"Rejected {label} path")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise AdapterError(f"{label} escaped output root") from exc
    return resolved


def _validated_rows(history: dict[str, Any]) -> list[dict[str, Any]]:
    rows = history.get("epochs")
    if not isinstance(rows, list) or not rows:
        raise AdapterError("History must contain epochs")
    finite_numeric = (
        "train_macro_f1", "val_macro_f1", "val_accuracy", "val_loss", "lr",
    )
    for epoch, row in enumerate(rows, start=1):
        if (
            not isinstance(row, dict)
            or isinstance(row.get("epoch"), bool)
            or not isinstance(row.get("epoch"), int)
            or row.get("epoch") != epoch
        ):
            raise AdapterError("History has a gap, duplicate, or invalid epoch")
        for key in finite_numeric:
            value = row.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise AdapterError(f"History epoch {epoch} lacks finite numeric {key}")
        wait = row.get("early_stopping_wait")
        if isinstance(wait, bool) or not isinstance(wait, int) or wait < 0:
            raise AdapterError(f"History epoch {epoch} has invalid early_stopping_wait")
        patience = row.get("early_stopping_patience")
        if (
            isinstance(patience, bool)
            or not isinstance(patience, int)
            or patience != EARLY_STOPPING_PATIENCE
        ):
            raise AdapterError(f"History epoch {epoch} has invalid early_stopping_patience")
        if not isinstance(row.get("stop_requested"), bool):
            raise AdapterError(f"History epoch {epoch} has invalid stop_requested")
        if epoch >= 31 and (
            row.get("row_origin") != CONTINUATION_ROW_ORIGIN
            or row.get("continuation_protocol_id") != CONTINUATION_PROTOCOL_ID
        ):
            raise AdapterError(f"Continuation row provenance drift at epoch {epoch}")
    return rows


def _is_forbidden_test_artifact(relative: str | Path) -> bool:
    candidate = Path(relative)
    parts = tuple(part.lower() for part in candidate.parts)
    basename = candidate.name.lower()
    exact_names = {name.lower() for name in FORBIDDEN_OUTPUT_NAMES}
    return (
        any(part == "test" for part in parts)
        or basename == "test.csv"
        or basename.startswith("test_")
        or basename.startswith("test-")
        or basename in exact_names
    )


def _forbidden_output_paths(output: Path) -> list[str]:
    return [
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if _is_forbidden_test_artifact(path.relative_to(output))
    ]


def validate_canonical_generation(output_root: str | Path) -> dict[str, Any]:
    output = Path(output_root).resolve()
    manifest = json_object(output / LATEST_STATE_MANIFEST, "latest-state manifest")
    epoch = manifest.get("completed_epoch")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("continuation_protocol_id") != CONTINUATION_PROTOCOL_ID
        or manifest.get("partial_epoch") is not False
        or manifest.get("test_access") is not False
        or isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 31
        or manifest.get("next_epoch") != epoch + 1
    ):
        raise AdapterError("Canonical manifest contract drift")
    generation = f"latest_states/epoch_{epoch:04d}"
    expected_paths = {
        "generation_id": f"epoch_{epoch:04d}",
        "generation_relative_path": generation,
        "model_relative_path": f"{generation}/state.keras",
        "metadata_relative_path": f"{generation}/state.metadata.json",
        "history_relative_path": f"{generation}/history.json",
    }
    for key, expected in expected_paths.items():
        if manifest.get(key) != expected:
            raise AdapterError(f"Canonical manifest path drift: {key}")
    paths = {
        "model": _resolve_under(output, manifest["model_relative_path"], "model"),
        "metadata": _resolve_under(output, manifest["metadata_relative_path"], "metadata"),
        "history": _resolve_under(output, manifest["history_relative_path"], "history"),
    }
    expected_hashes = {
        "model": manifest.get("model_sha256"),
        "metadata": manifest.get("metadata_sha256"),
        "history": manifest.get("combined_history_sha256"),
    }
    for key, path in paths.items():
        if not path.is_file() or sha256_file(path) != expected_hashes[key]:
            raise AdapterError(f"Canonical generation {key} SHA drift")
    metadata = json_object(paths["metadata"], "canonical metadata")
    required_metadata = {
        "schema_version": 2,
        "continuation_protocol_id": CONTINUATION_PROTOCOL_ID,
        "completed_epoch": epoch, "next_epoch": epoch + 1,
        "generation_relative_path": generation,
        "state_keras_sha256": expected_hashes["model"],
        "combined_history_sha256": expected_hashes["history"],
        "optimizer_state_sha256": manifest.get("optimizer_state_sha256"),
        "model_class": EXPECTED_CANDIDATE_CLASS,
        "model_parameter_count": EXPECTED_CANDIDATE_PARAMS,
        "model_trainable_variable_count": EXPECTED_VARIABLE_COUNT,
        "q_index": EXPECTED_Q_INDEX, "q_shape": EXPECTED_Q_SHAPE,
        "q_dtype": EXPECTED_Q_DTYPE, "optimizer_class": "LossScaleOptimizer",
        "partial_epoch": False, "test_access": False,
        "scientific_result_valid": False, "scientific_interpretation": None,
    }
    for key, expected in required_metadata.items():
        if metadata.get(key) != expected:
            raise AdapterError(f"Canonical metadata drift: {key}")
    fingerprint = metadata.get("optimizer_state_fingerprint")
    variable_count = metadata.get("optimizer_variable_count")
    if (
        not isinstance(fingerprint, dict)
        or fingerprint.get("sha256") != manifest.get("optimizer_state_sha256")
        or fingerprint.get("variable_count") != variable_count
        or not isinstance(fingerprint.get("variables"), list)
        or len(fingerprint["variables"]) != variable_count
    ):
        raise AdapterError("Canonical optimizer fingerprint drift")
    for index, variable in enumerate(fingerprint["variables"]):
        if (
            not isinstance(variable, dict) or variable.get("index") != index
            or not isinstance(variable.get("shape"), list)
            or not isinstance(variable.get("dtype"), str)
            or not isinstance(variable.get("value_sha256"), str)
            or len(variable["value_sha256"]) != 64
        ):
            raise AdapterError("Canonical optimizer fingerprint variable drift")
    history = json_object(paths["history"], "canonical history")
    if len(_validated_rows(history)) != epoch:
        raise AdapterError("Canonical history final epoch drift")
    generation_files = [
        (paths["model"], manifest["model_relative_path"]),
        (paths["metadata"], manifest["metadata_relative_path"]),
        (paths["history"], manifest["history_relative_path"]),
    ]
    best_hashes = metadata.get("best_val_accuracy_artifact_sha256")
    best_names = (
        "best_val_accuracy.keras", "best_val_accuracy.weights.h5",
        "best_val_accuracy.metadata.json",
    )
    if not isinstance(best_hashes, dict) or len(best_hashes) != len(best_names):
        raise AdapterError("Canonical best checkpoint inventory drift")
    for name in best_names:
        relative = f"{generation}/best_val_accuracy/{name}"
        snapshot = _resolve_under(output, relative, f"best {name}")
        root = output / "checkpoints" / name
        expected = best_hashes.get(relative)
        if (
            not snapshot.is_file() or sha256_file(snapshot) != expected
            or not root.is_file() or sha256_file(root) != expected
        ):
            raise AdapterError(f"Canonical/root best checkpoint drift: {name}")
        generation_files.append((snapshot, relative))
    return {
        "completed_epoch": epoch, "next_epoch": epoch + 1,
        "generation_relative_path": generation,
        "model_sha256": expected_hashes["model"],
        "metadata_sha256": expected_hashes["metadata"],
        "history_sha256": expected_hashes["history"],
        "optimizer_state_sha256": manifest["optimizer_state_sha256"],
        "q_sha256": metadata.get("q_flat_float32_sha256"),
        "generation_files": generation_files,
    }


def archive_members(
    run_root: str | Path, train_output_root: str | Path, report_path: str | Path
) -> list[tuple[Path, str]]:
    run, output = Path(run_root), Path(train_output_root)
    forbidden_outputs = _forbidden_output_paths(output)
    if forbidden_outputs:
        raise AdapterError(f"Forbidden test output contamination: {forbidden_outputs}")
    members = []
    adapter = run / "adapter"
    for path in sorted(adapter.glob("*")) if adapter.is_dir() else []:
        if path.is_file() and not path.name.endswith(".tmp"):
            members.append((path, f"adapter/{path.name}"))
    for relative in ROLLING_RELATIVE_FILES:
        path = output / relative
        if path.is_file():
            members.append((path, f"run/{Path(relative).as_posix()}"))
    if (output / LATEST_STATE_MANIFEST).is_file():
        canonical = validate_canonical_generation(output)
        members.extend((path, f"run/{relative}") for path, relative in canonical["generation_files"])
    report = Path(report_path)
    if report.is_file():
        members.append((report, report.name))
    names = [name for _path, name in members]
    if len(names) != len(set(names)):
        raise AdapterError("Duplicate archive members")
    forbidden = [name for name in names if _is_forbidden_test_artifact(name)]
    if forbidden:
        raise AdapterError(f"Forbidden test archive members: {forbidden}")
    return members


def publish_archive_atomic(
    archive_path: str | Path, run_root: str | Path,
    train_output_root: str | Path, report_path: str | Path,
    *, replace: Callable[[Any, Any], None] = os.replace,
) -> list[str]:
    destination = Path(archive_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.unlink(missing_ok=True)
    members = archive_members(run_root, train_output_root, report_path)
    expected = {name: sha256_file(path) for path, name in members}
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path, name in members:
                archive.write(path, name)
        with zipfile.ZipFile(temporary) as archive:
            names = archive.namelist()
            if archive.testzip() is not None or names != list(expected):
                raise AdapterError("Rolling ZIP integrity/inventory drift")
            for name, digest in expected.items():
                if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                    raise AdapterError(f"Rolling ZIP member hash drift: {name}")
        if not {"adapter/pre_run_manifest.json", "adapter/subprocess.log"}.issubset(names):
            raise AdapterError("Rolling ZIP lacks adapter provenance/log")
        if (Path(train_output_root) / LATEST_STATE_MANIFEST).is_file():
            canonical = validate_canonical_generation(train_output_root)
            required = {f"run/{LATEST_STATE_MANIFEST}"}
            required.update(f"run/{relative}" for _path, relative in canonical["generation_files"])
            if not required.issubset(names):
                raise AdapterError("Rolling ZIP lacks canonical generation")
        replace(temporary, destination)
        return names
    finally:
        temporary.unlink(missing_ok=True)


class RollingArchiveMonitor:
    """Refresh only after a newly committed canonical generation appears."""

    def __init__(
        self, *, manifest_path: str | Path, progress_path: str | Path,
        archive_path: str | Path, run_root: str | Path,
        train_output_root: str | Path, report_path: str | Path,
        archive_lock: threading.Lock | None = None, poll_seconds: float = 2.0,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.progress_path = Path(progress_path)
        self.archive_path = Path(archive_path)
        self.run_root = Path(run_root)
        self.train_output_root = Path(train_output_root)
        self.report_path = Path(report_path)
        self.archive_lock = archive_lock or threading.Lock()
        self.poll_seconds = float(poll_seconds)
        self.last_epoch = 0

    def poll_once(self) -> bool:
        if not self.manifest_path.is_file():
            return False
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return False
        epoch = manifest.get("completed_epoch") if isinstance(manifest, dict) else None
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= self.last_epoch:
            return False
        canonical = validate_canonical_generation(self.train_output_root)
        if canonical["completed_epoch"] != epoch:
            raise AdapterError("Manifest changed during rolling validation")
        atomic_json(self.progress_path, {
            "schema_version": 1, "status": "RUNNING",
            "latest_completed_epoch": epoch,
            "canonical_generation": canonical["generation_relative_path"],
            "scientific_result_valid": False, "scientific_interpretation": None,
            "monitor_read_only": True,
        })
        with self.archive_lock:
            publish_archive_atomic(
                self.archive_path, self.run_root, self.train_output_root, self.report_path
            )
        self.last_epoch = epoch
        return True

    def run(self, stop_event: threading.Event) -> None:
        while not stop_event.wait(self.poll_seconds):
            try:
                self.poll_once()
            except BaseException as exc:
                atomic_json(self.progress_path, {
                    "schema_version": 1, "status": "MONITOR_ARCHIVE_RETRY",
                    "latest_completed_epoch": self.last_epoch,
                    "scientific_result_valid": False,
                    "scientific_interpretation": None, "monitor_read_only": True,
                    "archive_error": f"{type(exc).__name__}: {exc}",
                })
        try:
            self.poll_once()
        except BaseException:
            pass


def derive_registered_metrics(history: dict[str, Any]) -> dict[str, Any]:
    rows = _validated_rows(history)
    best_macro_value = max(float(row["val_macro_f1"]) for row in rows)
    best_macro = next(row for row in rows if float(row["val_macro_f1"]) == best_macro_value)
    best_accuracy_value = max(float(row["val_accuracy"]) for row in rows)
    best_accuracy = next(row for row in rows if float(row["val_accuracy"]) == best_accuracy_value)
    best_loss_value = min(float(row["val_loss"]) for row in rows)
    best_loss = next(row for row in rows if float(row["val_loss"]) == best_loss_value)
    train_macro = float(best_macro["train_macro_f1"])
    gap_pp = 100.0 * (train_macro - best_macro_value)
    delta_pp = 100.0 * (best_macro_value - BASELINE_BEST_VAL_MACRO_F1)
    lr = [{"epoch": row["epoch"], "lr": float(row["lr"])} for row in rows]
    return {
        "candidate_best_val_macro_f1": best_macro_value,
        "candidate_best_val_macro_epoch": best_macro["epoch"],
        "candidate_train_macro_f1_at_best_macro_epoch": train_macro,
        "candidate_train_validation_macro_gap_pp": gap_pp,
        "delta_macro_pp": delta_pp,
        "registered_decision": registered_decision(delta_pp),
        "candidate_best_val_accuracy": best_accuracy_value,
        "candidate_best_val_accuracy_epoch": best_accuracy["epoch"],
        "candidate_best_val_loss": best_loss_value,
        "candidate_best_val_loss_epoch": best_loss["epoch"],
        "final_epoch": rows[-1]["epoch"],
        "lr_reductions": [
            current for previous, current in zip(lr, lr[1:])
            if current["lr"] < previous["lr"]
        ],
    }


def _validate_runtime_manifest(manifest: dict[str, Any], output: Path) -> None:
    required = {
        "schema_version": 1, "issue": 35, "execution_commit": EXECUTION_COMMIT,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "continuation_harness_invocations": 1,
        "direct_frozen_trainer_invocations": 0,
        "initial_step12_harness_invocations": 0,
        "chained_latest_state_invocations": 0, "automatic_retry": False,
        "seed": 42, "test_access": False,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise AdapterError(f"Adapter pre-run manifest drift: {key}")
    source_archive = manifest.get("source_archive_path")
    source_transport = manifest.get("source_transport_path")
    checkout = manifest.get("checkout_root")
    if (
        not isinstance(source_archive, str)
        or not isinstance(source_transport, str)
        or not isinstance(checkout, str)
    ):
        raise AdapterError("Adapter path identity missing")
    transport_name = Path(source_transport).name
    expected_transport_sha = (
        SOURCE_ARCHIVE_SHA256
        if transport_name == SOURCE_ARCHIVE_NAME
        else SOURCE_TRANSPORT_SHA256
        if transport_name == SOURCE_TRANSPORT_NAME
        else None
    )
    if (
        expected_transport_sha is None
        or manifest.get("source_transport_filename") != transport_name
        or manifest.get("source_transport_sha256") != expected_transport_sha
    ):
        raise AdapterError("Adapter source transport identity drift")
    if manifest.get("command") != registered_command(checkout, output, source_archive):
        raise AdapterError("Registered command drift")
    expected_resources = {
        "train_batch_size": 16, "eval_batch_size": 32, "graph_workers": 2,
        "tf_data_prefetch": 2, "tf_data_parallel_calls": 1,
        "graph_cache_size": 64, "mixed_precision": True, "xla": False,
        "memory_growth": True, "op_determinism_changed": False,
        "bounded_limits": None,
    }
    if manifest.get("resources") != expected_resources:
        raise AdapterError("Registered resource drift")
    expected_anchor = {
        "epoch": 30, "checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "q_sha256": EXPECTED_RESUME_Q_SHA256,
        "optimizer_iterations": EXPECTED_RESUME_OPTIMIZER_ITERATIONS,
        "optimizer_variables": EXPECTED_RESUME_OPTIMIZER_VARIABLES,
        "learning_rate": EXPECTED_RESUME_LR,
    }
    if manifest.get("resume_anchor") != expected_anchor:
        raise AdapterError("Registered epoch-30 resume anchor drift")
    runtime = manifest.get("runtime")
    expected_runtime = {
        "python": EXPECTED_PYTHON, "tensorflow": EXPECTED_TENSORFLOW,
        "keras": EXPECTED_KERAS, "cuda": EXPECTED_CUDA,
        "cudnn_major": EXPECTED_CUDNN_MAJOR, "gpu_count": 2,
    }
    if not isinstance(runtime, dict):
        raise AdapterError("Runtime evidence missing")
    for key, expected in expected_runtime.items():
        if runtime.get(key) != expected:
            raise AdapterError(f"Registered runtime drift: {key}")
    names = runtime.get("gpu_names")
    if not isinstance(names, list) or len(names) != 2 or not all("T4" in name for name in names):
        raise AdapterError("Registered GPU identity drift")


def _validate_pretrain_gate(output: Path) -> dict[str, Any]:
    gate = json_object(output / "pretrain_validation_gate.json", "pretrain gate")
    references = {
        "sample_count": 3589, "accuracy": 0.603789356366676,
        "macro_f1": 0.5634445160028113, "loss": 1.1265364869505958,
    }
    tolerances = {"accuracy": 0.001, "macro_f1": 0.001, "loss": 0.005}
    if (
        gate.get("status") != "PASS" or gate.get("sample_count") != 3589
        or gate.get("optimizer_updates_before_gate") != 0
        or gate.get("references") != references or gate.get("tolerances") != tolerances
    ):
        raise AdapterError("Pretrain validation gate failed or drifted")
    observed, differences = gate.get("observed"), gate.get("absolute_differences")
    if not isinstance(observed, dict) or not isinstance(differences, dict):
        raise AdapterError("Pretrain gate numerical evidence missing")
    for key, tolerance in tolerances.items():
        difference = abs(float(observed.get(key)) - references[key])
        if difference > tolerance or float(differences.get(key)) != difference:
            raise AdapterError(f"Pretrain gate metric drift: {key}")
    return gate


def _validate_overlap(
    output: Path, rows: list[dict[str, Any]], source_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    payload = json_object(output / OVERLAP_SOURCE, "overlap source")
    originals = payload.get("rows")
    if (
        payload.get("classification") != "FIRST_RUN_OVERLAP_DIAGNOSTICS"
        or payload.get("descriptive_only") is not True
        or payload.get("excluded_from_combined_scientific_history") is not True
        or not isinstance(originals, dict) or set(originals) != {"31", "32"}
    ):
        raise AdapterError("Overlap source drift")
    evidence = {}
    for epoch in (31, 32):
        original, resumed = originals[str(epoch)], rows[epoch - 1]
        if original != source_rows[epoch - 1]:
            raise AdapterError(f"Locked overlap row drift: {epoch}")
        diagnostic = json_object(output / f"resume_overlap_epoch{epoch}.json", "overlap")
        required = {
            "classification": "FIRST_RUN_OVERLAP_DIAGNOSTICS", "epoch": epoch,
            "descriptive_only": True, "affects_training": False,
            "affects_stopping": False, "affects_scheduler": False,
            "affects_checkpoint_selection": False, "affects_primary_endpoint": False,
            "triggers_retry": False,
        }
        if any(diagnostic.get(key) != value for key, value in required.items()):
            raise AdapterError(f"Overlap diagnostic contract drift: {epoch}")
        if diagnostic.get("first_run_row") != original or diagnostic.get("resumed_row") != resumed:
            raise AdapterError(f"Overlap diagnostic row drift: {epoch}")
        evidence[str(epoch)] = diagnostic
    return evidence


def deep_validate_latest_state(checkout_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    root = Path(checkout_root)
    for path in (root, root / "standalone/lap_gnn_tensorflow_ofix7_mid_candidate/src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from research.candidates.tf_learned_local_residual_slots import resume_validation_only as resume
    state = resume.load_latest_completed_state(output_root)
    identity = resume.model_identity(state.model)
    fingerprint = resume.optimizer_state_fingerprint(state.optimizer)
    return {
        "status": "PASS", "completed_epoch": state.completed_epoch,
        "next_epoch": state.next_epoch, "model_class": identity["class"],
        "model_parameter_count": identity["parameter_count"],
        "model_trainable_variable_count": identity["trainable_variable_count"],
        "q_index": identity["q_index"], "q_shape": identity["q_shape"],
        "q_dtype": identity["q_dtype"], "q_sha256": identity["q_flat_float32_sha256"],
        "optimizer_state_sha256": fingerprint["sha256"], "test_access": False,
    }


def validate_completion(
    train_output_root: str | Path, adapter_manifest_path: str | Path,
    source_hashes_before: dict[str, str], source_hashes_after: dict[str, str],
    deep_state_evidence: dict[str, Any],
) -> dict[str, Any]:
    output = Path(train_output_root).resolve()
    if source_hashes_before != source_hashes_after:
        raise AdapterError("Reviewed source changed during execution")
    forbidden = _forbidden_output_paths(output)
    if forbidden:
        raise AdapterError(f"Forbidden test output contamination: {forbidden}")
    if (output / TECHNICAL_FAILURE_MARKER).exists():
        raise AdapterError("Technical continuation failure marker exists")
    adapter_manifest = json_object(adapter_manifest_path, "adapter pre-run manifest")
    _validate_runtime_manifest(adapter_manifest, output)
    if adapter_manifest.get("source_locks") != source_hashes_before:
        raise AdapterError("Adapter pre-run source lock evidence drift")
    if adapter_manifest.get("scientific_payload_sha256") != SCIENTIFIC_PAYLOAD_SHA256:
        raise AdapterError("Adapter frozen payload identity drift")
    if (
        adapter_manifest.get("inherited_execution_contract_sha256")
        != BASELINE_EXECUTION_CONTRACT_SHA256
    ):
        raise AdapterError("Adapter execution contract identity drift")
    source_rows = locked_source_history(adapter_manifest["source_archive_path"])
    gate = _validate_pretrain_gate(output)

    completion_path = output / COMPLETION_MARKER
    completion = json_object(completion_path, "completion marker")
    required_completion = {
        "schema_version": 1,
        "continuation_protocol_id": CONTINUATION_PROTOCOL_ID,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "continuation_completed": True, "scientific_prefix_first_epoch": 1,
        "scientific_prefix_last_epoch": 30, "first_continuation_epoch": 31,
        "original_step12_scientific_result_valid": False,
        "original_step12_scientific_interpretation": None,
        "scientific_result_valid": False, "scientific_interpretation": None,
        "training": True, "optimizer_gradient_updates": True,
        "test_access": False, "test_data_constructed": False,
        "final_test_skipped": True,
    }
    for key, expected in required_completion.items():
        if completion.get(key) != expected:
            raise AdapterError(f"Completion gate drift: {key}")
    if completion.get("completion_reason") not in {"early_stopping", "max_epochs"}:
        raise AdapterError("Completion reason is not natural")

    history_path = output / "history.json"
    history = json_object(history_path, "combined history")
    rows = _validated_rows(history)
    final_epoch = rows[-1]["epoch"]
    if final_epoch < 32 or completion.get("final_completed_epoch") != final_epoch:
        raise AdapterError("Completion/history final epoch drift")
    if rows[:30] != source_rows[:30]:
        raise AdapterError("Immutable source prefix drift")
    final_row = rows[-1]
    if completion["completion_reason"] == "early_stopping":
        if (
            final_row["stop_requested"] is not True
            or final_row["early_stopping_wait"] < EARLY_STOPPING_PATIENCE
            or final_row["early_stopping_patience"] != EARLY_STOPPING_PATIENCE
        ):
            raise AdapterError("Early-stopping completion state is not natural")
    elif final_epoch != MAX_EPOCHS or final_row["stop_requested"] is not False:
        raise AdapterError("Max-epochs completion state is not natural")
    continuation_manifest = json_object(
        output / "continuation_pre_run_manifest.json", "continuation pre-run manifest"
    )
    if (
        continuation_manifest.get("schema_version") != 1
        or continuation_manifest.get("protocol_id") != CONTINUATION_PROTOCOL_ID
        or continuation_manifest.get("source_archive_sha256") != SOURCE_ARCHIVE_SHA256
        or continuation_manifest.get("source_history_sha256") != SOURCE_HISTORY_SHA256
        or continuation_manifest.get("immutable_scientific_prefix_sha256")
        != _canonical_json_sha256(rows[:30])
    ):
        raise AdapterError("Continuation provenance/prefix drift")
    overlap = _validate_overlap(output, rows, source_rows)

    canonical = validate_canonical_generation(output)
    if canonical["completed_epoch"] != final_epoch:
        raise AdapterError("Final epoch/latest-state manifest mismatch")
    if completion.get("latest_state_generation") != canonical["generation_relative_path"]:
        raise AdapterError("Completion canonical generation drift")
    if completion.get("latest_state_model_sha256") != canonical["model_sha256"]:
        raise AdapterError("Completion canonical model SHA drift")
    if completion.get("latest_state_optimizer_sha256") != canonical["optimizer_state_sha256"]:
        raise AdapterError("Completion optimizer fingerprint SHA drift")
    if completion.get("combined_history_sha256") != sha256_file(history_path):
        raise AdapterError("Completion root history SHA drift")
    if completion.get("latest_state_manifest_sha256") != sha256_file(
        output / LATEST_STATE_MANIFEST
    ):
        raise AdapterError("Completion latest-state manifest SHA drift")
    generation_history = json_object(
        output / canonical["generation_relative_path"] / "history.json",
        "canonical history",
    )
    if generation_history != history:
        raise AdapterError("Root/canonical history mismatch")
    required_deep = {
        "status": "PASS", "completed_epoch": final_epoch,
        "next_epoch": final_epoch + 1, "model_class": EXPECTED_CANDIDATE_CLASS,
        "model_parameter_count": EXPECTED_CANDIDATE_PARAMS,
        "model_trainable_variable_count": EXPECTED_VARIABLE_COUNT,
        "q_index": EXPECTED_Q_INDEX, "q_shape": EXPECTED_Q_SHAPE,
        "q_dtype": EXPECTED_Q_DTYPE, "q_sha256": canonical["q_sha256"],
        "optimizer_state_sha256": canonical["optimizer_state_sha256"],
        "test_access": False,
    }
    for key, expected in required_deep.items():
        if deep_state_evidence.get(key) != expected:
            raise AdapterError(f"Deep canonical state validation drift: {key}")

    metadata_path = output / "checkpoints/best_val_accuracy.metadata.json"
    checkpoint_metadata = json_object(metadata_path, "root best metadata")
    checkpoint_epoch = checkpoint_metadata.get("epoch")
    metrics = checkpoint_metadata.get("validation_metrics")
    if isinstance(checkpoint_epoch, bool) or not isinstance(checkpoint_epoch, int):
        raise AdapterError("Root best checkpoint epoch invalid")
    if not isinstance(metrics, dict):
        raise AdapterError("Root best checkpoint metrics missing")
    best_accuracy = max(float(row["val_accuracy"]) for row in rows)
    earliest_best = next(row for row in rows if float(row["val_accuracy"]) == best_accuracy)
    if checkpoint_epoch != earliest_best["epoch"]:
        raise AdapterError("Root best checkpoint is not earliest global max val_accuracy")
    for key, history_key in (
        ("accuracy", "val_accuracy"), ("macro_f1", "val_macro_f1"),
        ("loss", "val_loss"),
    ):
        if float(metrics.get(key)) != float(earliest_best[history_key]):
            raise AdapterError(f"Root best checkpoint metric drift: {key}")

    derived = derive_registered_metrics(history)
    derived.update({
        "completion_reason": completion["completion_reason"],
        "pretrain_validation_gate": gate, "overlap_diagnostics": overlap,
        "canonical_latest_state": {
            key: value for key, value in canonical.items() if key != "generation_files"
        },
        "root_best_val_accuracy_checkpoint": {
            "epoch": checkpoint_epoch, "accuracy": float(metrics["accuracy"]),
            "macro_f1": float(metrics["macro_f1"]), "loss": float(metrics["loss"]),
            "checkpoint_sha256": sha256_file(output / "checkpoints/best_val_accuracy.keras"),
            "weights_sha256": sha256_file(output / "checkpoints/best_val_accuracy.weights.h5"),
            "metadata_sha256": sha256_file(metadata_path),
        },
    })
    return derived


def partial_disposition(status: str = "RUNTIME_HARD_CENSORED") -> dict[str, Any]:
    return {
        "status": status, "scientific_result_valid": False,
        "scientific_interpretation": None, "automatic_retry": False,
        "test_access": False,
    }


def write_success_outputs(
    derived: dict[str, Any], *, subprocess_return_code: int,
    source_hashes_before: dict[str, str], source_hashes_after: dict[str, str],
    wrapper_path: str | Path, evidence_path: str | Path, report_path: str | Path,
) -> dict[str, Any]:
    wrapper = {
        "schema_version": 1, "status": "COMPLETE",
        "subprocess_return_code": int(subprocess_return_code), "error_text": None,
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
        "scientific_result_valid": True,
        "scientific_interpretation": derived["registered_decision"],
        "training": True, "test_access": False, "automatic_retry": False,
    }
    evidence = {
        "schema_version": 1, "issue": 35, "execution_commit": EXECUTION_COMMIT,
        "checkpoint_conditioned_restart": True,
        "scientific_result_valid": True,
        "scientific_interpretation": derived["registered_decision"],
        "primary_endpoint": derived, "test_access": False,
    }
    atomic_json(wrapper_path, wrapper)
    atomic_json(evidence_path, evidence)
    Path(report_path).write_text(textwrap.dedent(f"""\
        # TF Step 12D checkpoint-conditioned continuation

        - Status: `COMPLETE`
        - Scientific result valid: `true`
        - Registered label: `{derived['registered_decision']}`
        - Best validation macro-F1: `{derived['candidate_best_val_macro_f1']}`
        - Delta versus locked baseline: `{derived['delta_macro_pp']}` pp
        - Test access: `false`

        This is single-seed descriptive/practical validation evidence from a
        checkpoint-conditioned restart, not an uninterrupted Step-12 result.
        """), encoding="utf-8", newline="\n")
    return wrapper


def write_failure_outputs(
    *, subprocess_return_code: int | None, error_text: str,
    source_hashes_before: dict[str, str],
    source_hashes_after: dict[str, str] | None,
    wrapper_path: str | Path, evidence_path: str | Path,
    failure_report_path: str | Path,
) -> dict[str, Any]:
    Path(evidence_path).unlink(missing_ok=True)
    wrapper = {
        "schema_version": 1, "status": "TECHNICAL_CONTINUATION_FAILURE",
        "subprocess_return_code": subprocess_return_code, "error_text": error_text,
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
        "scientific_result_valid": False, "scientific_interpretation": None,
        "training": True, "test_access": False, "automatic_retry": False,
    }
    atomic_json(wrapper_path, wrapper)
    Path(failure_report_path).write_text(textwrap.dedent(f"""\
        # TF Step 12D technical continuation failure

        - Status: `TECHNICAL_CONTINUATION_FAILURE`
        - Subprocess return code: `{subprocess_return_code}`
        - Scientific result valid: `false`
        - Scientific interpretation: `null`
        - Error: `{error_text}`

        Partial evidence is diagnostic only. No retry or scientific label is authorized.
        """), encoding="utf-8", newline="\n")
    return wrapper


def run_subprocess_once(
    command: Sequence[str], *, cwd: str | Path, log_path: str | Path,
    monitor: RollingArchiveMonitor, popen_factory: Any = subprocess.Popen,
) -> int:
    stop_event = threading.Event()
    thread = threading.Thread(
        target=monitor.run, args=(stop_event,),
        name="issue35-canonical-generation-monitor", daemon=True,
    )
    destination = Path(log_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n", buffering=1) as log:
        process = popen_factory(
            [str(item) for item in command], cwd=str(cwd),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        thread.start()
        try:
            if process.stdout is None:
                raise AdapterError("Continuation subprocess stdout unavailable")
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = process.wait()
        finally:
            stop_event.set()
            thread.join(timeout=max(10.0, monitor.poll_seconds * 3.0))
    return int(return_code)


def run_registered_adapter(
    runtime_evidence: dict[str, Any],
    source_transport: str | Path,
    source_archive: str | Path,
) -> dict[str, Any]:
    """Invoke exactly one initial checkpoint-continuation subprocess."""

    if RUN_ROOT.exists() or ARCHIVE_PATH.exists() or REPORT_PATH.exists():
        raise FileExistsError("Fresh Issue #35 output paths are required")
    source_transport = Path(source_transport).resolve()
    source_archive = Path(source_archive).resolve()
    if source_transport != discover_source_transport(KAGGLE_INPUT_ROOT):
        raise AdapterError("Source transport changed after preflight")
    if sha256_file(source_archive) != SOURCE_ARCHIVE_SHA256:
        raise AdapterError("Materialized source archive SHA drift")
    source_before = verify_source_locks(CHECKOUT_ROOT)
    command = registered_command(CHECKOUT_ROOT, TRAIN_OUTPUT_ROOT, source_archive)
    RUN_ROOT.mkdir(parents=True, exist_ok=False)
    ADAPTER_ROOT.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1, "issue": 35, "execution_commit": EXECUTION_COMMIT,
        "checkout_root": str(CHECKOUT_ROOT),
        "source_transport_path": str(source_transport),
        "source_transport_filename": source_transport.name,
        "source_transport_sha256": sha256_file(source_transport),
        "source_archive_path": str(source_archive),
        "source_archive_filename": SOURCE_ARCHIVE_NAME,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "source_locks": source_before,
        "scientific_payload_sha256": SCIENTIFIC_PAYLOAD_SHA256,
        "inherited_execution_contract_sha256": BASELINE_EXECUTION_CONTRACT_SHA256,
        "command": command, "continuation_harness_invocations": 1,
        "direct_frozen_trainer_invocations": 0,
        "initial_step12_harness_invocations": 0,
        "chained_latest_state_invocations": 0, "automatic_retry": False,
        "seed": 42,
        "resources": {
            "train_batch_size": 16, "eval_batch_size": 32, "graph_workers": 2,
            "tf_data_prefetch": 2, "tf_data_parallel_calls": 1,
            "graph_cache_size": 64, "mixed_precision": True, "xla": False,
            "memory_growth": True, "op_determinism_changed": False,
            "bounded_limits": None,
        },
        "resume_anchor": {
            "epoch": 30, "checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "q_sha256": EXPECTED_RESUME_Q_SHA256,
            "optimizer_iterations": EXPECTED_RESUME_OPTIMIZER_ITERATIONS,
            "optimizer_variables": EXPECTED_RESUME_OPTIMIZER_VARIABLES,
            "learning_rate": EXPECTED_RESUME_LR,
        },
        "runtime": runtime_evidence, "test_access": False,
        "scientific_result_valid": False, "scientific_interpretation": None,
    }
    atomic_json(PRE_RUN_MANIFEST_PATH, manifest)
    SUBPROCESS_LOG_PATH.touch()
    archive_lock = threading.Lock()
    monitor = RollingArchiveMonitor(
        manifest_path=TRAIN_OUTPUT_ROOT / LATEST_STATE_MANIFEST,
        progress_path=RUNTIME_PROGRESS_PATH, archive_path=ARCHIVE_PATH,
        run_root=RUN_ROOT, train_output_root=TRAIN_OUTPUT_ROOT,
        report_path=REPORT_PATH, archive_lock=archive_lock,
    )
    with archive_lock:
        publish_archive_atomic(ARCHIVE_PATH, RUN_ROOT, TRAIN_OUTPUT_ROOT, REPORT_PATH)
    return_code = None
    source_after = None
    try:
        return_code = run_subprocess_once(
            command, cwd=CHECKOUT_ROOT, log_path=SUBPROCESS_LOG_PATH, monitor=monitor
        )
        source_after = verify_source_locks(CHECKOUT_ROOT)
        if run_checked(["git", "status", "--porcelain"], cwd=CHECKOUT_ROOT).strip():
            raise AdapterError("Execution checkout became dirty")
        if return_code != 0:
            raise AdapterError(f"Continuation harness exited with code {return_code}")
        # Shallow validation occurs before the reviewed loader can recover root files.
        validate_canonical_generation(TRAIN_OUTPUT_ROOT)
        deep = deep_validate_latest_state(CHECKOUT_ROOT, TRAIN_OUTPUT_ROOT)
        derived = validate_completion(
            TRAIN_OUTPUT_ROOT, PRE_RUN_MANIFEST_PATH,
            source_before, source_after, deep,
        )
        wrapper = write_success_outputs(
            derived, subprocess_return_code=return_code,
            source_hashes_before=source_before, source_hashes_after=source_after,
            wrapper_path=WRAPPER_EXECUTION_PATH, evidence_path=FINAL_EVIDENCE_PATH,
            report_path=REPORT_PATH,
        )
    except BaseException as exc:
        if source_after is None:
            try:
                source_after = verify_source_locks(CHECKOUT_ROOT)
            except BaseException:
                source_after = None
        wrapper = write_failure_outputs(
            subprocess_return_code=return_code,
            error_text=f"{type(exc).__name__}: {exc}",
            source_hashes_before=source_before, source_hashes_after=source_after,
            wrapper_path=WRAPPER_EXECUTION_PATH, evidence_path=FINAL_EVIDENCE_PATH,
            failure_report_path=FAILURE_REPORT_PATH,
        )
    try:
        with archive_lock:
            publish_archive_atomic(ARCHIVE_PATH, RUN_ROOT, TRAIN_OUTPUT_ROOT, REPORT_PATH)
    except BaseException as exc:
        # The previous verified rolling ZIP is intentionally retained.
        wrapper["final_archive_refresh_error"] = f"{type(exc).__name__}: {exc}"
        atomic_json(WRAPPER_EXECUTION_PATH, wrapper)
    print(json.dumps(wrapper, indent=2, sort_keys=True), flush=True)
    return wrapper


def _source(text: str) -> list[str]:
    return textwrap.dedent(text).lstrip("\n").splitlines(keepends=True)


def _markdown(text: str, cell_id: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": _source(text)}


def _code(text: str, cell_id: str) -> dict[str, Any]:
    return {
        "cell_type": "code", "execution_count": None, "id": cell_id,
        "metadata": {}, "outputs": [], "source": _source(text),
    }


def _runtime_definitions_source() -> str:
    objects = (
        AdapterError, sha256_file, atomic_json, json_object, run_checked,
        verify_source_locks, discover_source_transport, materialize_source_archive,
        discover_source_archive, locked_source_history,
        registered_decision, registered_command, _canonical_json_sha256,
        _resolve_under, _validated_rows, _is_forbidden_test_artifact,
        _forbidden_output_paths, validate_canonical_generation,
        archive_members, publish_archive_atomic, RollingArchiveMonitor,
        derive_registered_metrics, _validate_runtime_manifest,
        _validate_pretrain_gate, _validate_overlap, deep_validate_latest_state,
        validate_completion, partial_disposition, write_success_outputs,
        write_failure_outputs, run_subprocess_once, run_registered_adapter,
    )
    return "\n\n".join(inspect.getsource(item).rstrip() for item in objects) + "\n"


def _constants_source() -> str:
    names = (
        "REPOSITORY_URL", "EXECUTION_COMMIT", "SCIENTIFIC_PAYLOAD_SHA256",
        "BASELINE_EXECUTION_CONTRACT_SHA256", "SOURCE_LOCKS",
        "EXPECTED_PYTHON", "EXPECTED_TENSORFLOW", "EXPECTED_KERAS",
        "EXPECTED_CUDA", "EXPECTED_CUDNN_MAJOR", "EXPECTED_GPU_COUNT",
        "EXPECTED_GPU_TOKEN", "EXPECTED_CANDIDATE_CLASS",
        "EXPECTED_CANDIDATE_PARAMS", "EXPECTED_VARIABLE_COUNT",
        "EXPECTED_Q_INDEX", "EXPECTED_Q_SHAPE", "EXPECTED_Q_DTYPE",
        "EXPECTED_RESUME_Q_SHA256", "EXPECTED_RESUME_OPTIMIZER_ITERATIONS",
        "EXPECTED_RESUME_OPTIMIZER_VARIABLES", "EXPECTED_RESUME_LR",
        "SOURCE_ARCHIVE_NAME", "SOURCE_ARCHIVE_SHA256", "SOURCE_TRANSPORT_NAME",
        "SOURCE_TRANSPORT_SHA256", "SOURCE_HISTORY_SHA256",
        "SOURCE_CHECKPOINT_SHA256", "CONTINUATION_PROTOCOL_ID",
        "CONTINUATION_ROW_ORIGIN", "BASELINE_BEST_VAL_MACRO_F1",
        "PRACTICAL_EFFECT_THRESHOLD_PP", "FER_ROOT", "FER_TRAIN_CSV",
        "FER_VAL_CSV", "PRIOR_ROOT", "CACHE_ROOT", "KAGGLE_INPUT_ROOT",
        "WORKING_ROOT", "MATERIALIZED_SOURCE_ARCHIVE_PATH", "CHECKOUT_ROOT",
        "RUN_ROOT", "TRAIN_OUTPUT_ROOT",
        "ADAPTER_ROOT", "PRE_RUN_MANIFEST_PATH", "SUBPROCESS_LOG_PATH",
        "WRAPPER_EXECUTION_PATH", "FINAL_EVIDENCE_PATH",
        "RUNTIME_PROGRESS_PATH", "FAILURE_REPORT_PATH", "REPORT_PATH",
        "ARCHIVE_PATH", "TRAIN_BATCH_SIZE", "EVAL_BATCH_SIZE", "GRAPH_WORKERS",
        "TF_DATA_PREFETCH", "TF_DATA_PARALLEL_CALLS", "GRAPH_CACHE_SIZE",
        "MAX_EPOCHS", "EARLY_STOPPING_PATIENCE",
        "EXPECTED_SPLIT_COUNTS", "COMPLETION_MARKER", "TECHNICAL_FAILURE_MARKER",
        "LATEST_STATE_MANIFEST", "OVERLAP_SOURCE", "FORBIDDEN_OUTPUT_NAMES",
        "ROLLING_RELATIVE_FILES",
    )
    lines = []
    for name in names:
        value = globals()[name]
        if isinstance(value, Path):
            lines.append(f"{name} = Path({value.as_posix()!r})")
        elif isinstance(value, set):
            lines.append(f"{name} = set({sorted(value)!r})")
        else:
            lines.append(f"{name} = {value!r}")
    return "\n".join(lines) + "\n"


def build_notebook() -> dict[str, Any]:
    imports = textwrap.dedent("""\
    from __future__ import annotations
    import base64
    import csv
    import hashlib
    import importlib.metadata
    import json
    import math
    import os
    from pathlib import Path
    import platform
    import subprocess
    import sys
    import textwrap
    import threading
    import zipfile
    from typing import Any, Callable, Sequence
    sys.dont_write_bytecode = True
    """)
    checkout = """\
    if CHECKOUT_ROOT.exists():
        raise FileExistsError(f"Fresh detached checkout required: {CHECKOUT_ROOT}")
    run_checked(["git", "clone", "--no-checkout", REPOSITORY_URL, CHECKOUT_ROOT])
    run_checked(["git", "checkout", "--detach", EXECUTION_COMMIT], cwd=CHECKOUT_ROOT)
    actual_commit = run_checked(["git", "rev-parse", "HEAD"], cwd=CHECKOUT_ROOT).strip()
    dirty = run_checked(["git", "status", "--porcelain"], cwd=CHECKOUT_ROOT).strip()
    if actual_commit != EXECUTION_COMMIT or dirty:
        raise AdapterError(f"Detached checkout drift: {actual_commit}, dirty={bool(dirty)}")
    source_hashes = verify_source_locks(CHECKOUT_ROOT)
    package_root = CHECKOUT_ROOT / "standalone/lap_gnn_tensorflow_ofix7_mid_candidate"
    run_checked([sys.executable, "-B", package_root / "tools/verify_checksums.py"], cwd=package_root)
    print(json.dumps({"commit": actual_commit, "source_hashes": source_hashes}, indent=2))
    """
    preflight = """\
    if platform.python_version() != EXPECTED_PYTHON:
        raise AdapterError(f"Python drift: {platform.python_version()}")
    source_transport = discover_source_transport(KAGGLE_INPUT_ROOT)
    source_archive = materialize_source_archive(
        source_transport, MATERIALIZED_SOURCE_ARCHIVE_PATH
    )
    source_rows = locked_source_history(source_archive)
    with zipfile.ZipFile(source_archive) as archive:
        checkpoint_digest = hashlib.sha256(
            archive.read("run/checkpoints/best_val_accuracy.keras")
        ).hexdigest()
    if checkpoint_digest != SOURCE_CHECKPOINT_SHA256:
        raise AdapterError("Epoch-30 checkpoint SHA drift")
    gpu_names = [
        line.strip() for line in run_checked(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
        ).splitlines() if line.strip()
    ]
    if len(gpu_names) != 2 or not all("T4" in name for name in gpu_names):
        raise AdapterError(f"Exactly two Tesla T4 GPUs required: {gpu_names}")
    def installed_version(distribution):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return None
    installed_tf = installed_version("tensorflow")
    installed_keras = installed_version("keras")
    if installed_tf != EXPECTED_TENSORFLOW or installed_keras != EXPECTED_KERAS:
        run_checked([
            sys.executable, "-m", "pip", "install", "-q", "--no-warn-conflicts",
            "-r", package_root / "requirements-kaggle.txt",
        ])
    import tensorflow as tf
    if tf.__version__ != EXPECTED_TENSORFLOW or tf.keras.__version__ != EXPECTED_KERAS:
        raise AdapterError("TensorFlow/Keras runtime drift")
    build_info = tf.sysconfig.get_build_info()
    cuda = str(build_info.get("cuda_version"))
    cudnn_major = str(build_info.get("cudnn_version")).split(".")[0]
    if cuda != EXPECTED_CUDA or cudnn_major != EXPECTED_CUDNN_MAJOR:
        raise AdapterError(f"CUDA/cuDNN drift: {cuda}/{cudnn_major}")
    if len(tf.config.list_physical_devices("GPU")) != 2:
        raise AdapterError("TensorFlow GPU count drift")
    for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
        csv_path = {"train": FER_TRAIN_CSV, "val": FER_VAL_CSV}[split]
        with csv_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            header = [value.strip().lower() for value in next(reader)]
            row_count = sum(1 for _ in reader)
        if row_count != expected_count or not {"emotion", "pixels"}.issubset(header):
            raise AdapterError(f"Registered {split} CSV drift")
        if len(list((PRIOR_ROOT / split).glob("*.npz"))) != expected_count:
            raise AdapterError(f"Registered {split} prior count drift")
        cache = json_object(CACHE_ROOT / split / "index.json", f"{split} cache index")
        if cache.get("schema_version") != "tf_clean_graph_cache_v2_records":
            raise AdapterError(f"Registered {split} cache schema drift")
        if cache.get("sample_count") != expected_count:
            raise AdapterError(f"Registered {split} cache count drift")
        for shard in cache.get("shards", []):
            if not (CACHE_ROOT / split / shard["path"]).is_file():
                raise AdapterError(f"Registered {split} cache shard missing")
    runtime_evidence = {
        "python": platform.python_version(), "tensorflow": tf.__version__,
        "keras": tf.keras.__version__, "cuda": cuda, "cudnn_major": cudnn_major,
        "gpu_count": len(gpu_names), "gpu_names": gpu_names,
    }
    print(json.dumps({
        "runtime": runtime_evidence, "source_archive": str(source_archive),
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "resume_anchor": {
            "epoch": 30, "checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "q_sha256": EXPECTED_RESUME_Q_SHA256,
            "optimizer_iterations": EXPECTED_RESUME_OPTIMIZER_ITERATIONS,
            "optimizer_variables": EXPECTED_RESUME_OPTIMIZER_VARIABLES,
            "learning_rate": EXPECTED_RESUME_LR,
        },
        "allowed_splits": ["train", "val"], "test_access": False,
    }, indent=2))
    """
    run_cell = """\
    wrapper_execution = run_registered_adapter(
        runtime_evidence, source_transport, source_archive
    )
    if not ARCHIVE_PATH.is_file():
        raise AdapterError("Failure-safe continuation archive was not published")
    print("archive:", ARCHIVE_PATH)
    print("status:", wrapper_execution["status"])
    """
    return {
        "cells": [
            _markdown("""
                # Issue #35: Step 12D checkpoint-conditioned continuation adapter

                Pre-run adapter only. Do not execute until separate one-run approval.
                The original Step-12 run remains hard-censored and scientifically invalid.

                Required offline Kaggle Inputs:

                - FER root: `/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split`;
                  only `train.csv` and `val.csv` are opened.
                - MediaPipe priors: `/kaggle/input/datasets/irthn1311/d16-mediapipe-pixel-priors-best-retry-rescue/outputs/d16_mediapipe_pixel_priors_best_retry_rescue`;
                  only registered train/validation records are used.
                - Clean cache: `/kaggle/input/datasets/irthn1311/ofix7-mid-seed42-records`;
                  only train/validation indexes and shards are used.
                - Censored source transport: exactly one direct reviewed ZIP or
                  `tf_step12_learned_local_residual_slots_seed42_kaggle_t4.zip.b64`
                  anywhere below `/kaggle/input`. The Base64 transport is locked,
                  decoded atomically in `/kaggle/working`, and must reproduce the
                  exact reviewed source archive SHA-256 before harness invocation.

                Internet is used only for the exact Git clone and pinned dependency
                installation if the Kaggle image differs. Scientific inputs are offline.

                Rolling/final output:
                `/kaggle/working/tf_step12c_checkpoint_continuation_kaggle_t4.zip`.
                """, "issue35-00"),
            _markdown("## 1. Locked failure-safe adapter definitions\n", "issue35-01"),
            _code(imports + "\n" + _constants_source() + "\n" + _runtime_definitions_source(), "issue35-02"),
            _markdown("## 2. Exact detached merged Step-12C checkout\n", "issue35-03"),
            _code(checkout, "issue35-04"),
            _markdown("## 3. Runtime, inputs, source archive, and resume anchor\n", "issue35-05"),
            _code(preflight, "issue35-06"),
            _markdown("## 4. Exactly one first-continuation subprocess\n", "issue35-07"),
            _code(run_cell, "issue35-08"),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": EXPECTED_PYTHON},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(
        json.dumps(build_notebook(), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
