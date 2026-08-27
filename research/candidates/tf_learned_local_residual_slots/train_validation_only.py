"""Validation-only training entry point for the reviewed Step-10 candidate.

The frozen wrapper and trainer remain the lifecycle owners.  Protocol
Amendment A authorizes exactly two temporary trainer bindings: the candidate
constructor and its selected 128-variable restricted G1-A train-step builder.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
FROZEN_PACKAGE_ROOT = (
    REPOSITORY_ROOT / "standalone" / "lap_gnn_tensorflow_ofix7_mid_candidate"
)
FROZEN_PACKAGE_SRC = FROZEN_PACKAGE_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(FROZEN_PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(FROZEN_PACKAGE_SRC))

from lap_gnn_tf.config import load_config  # noqa: E402
from lap_gnn_tf.model import LapGNN as FrozenLapGNN  # noqa: E402
from lap_gnn_tf.signatures import scientific_payload_checksum  # noqa: E402
from lap_gnn_tf.training import trainer  # noqa: E402
from lap_gnn_tf.training.execution import (  # noqa: E402
    build_restricted_graph_train_step as frozen_build_restricted_graph_train_step,
)
from research.candidates.tf_learned_local_residual_slots.candidate_execution import (  # noqa: E402
    build_candidate_restricted_graph_train_step,
)
from research.candidates.tf_learned_local_residual_slots.model import (  # noqa: E402
    LearnedLocalResidualSlotLapGNN,
)


HARNESS_VERSION = "tf-step11-candidate-validation-only-amendment-a-v1"
ARCHITECTURE_ID = "learned_local_residual_slots_k4_qdot_v1"
CANDIDATE_EXECUTION_CONTRACT_ID = (
    "learned_local_residual_slots_candidate_execution_v1"
)
IMPLEMENTATION_BASE = "572885a0bb650434f5b36bd3be2049524377067b"
EXPECTED_CANDIDATE_MODEL_SHA256 = (
    "0a7cfa315baf0d6666b7ab86139b328ab6d138985cfddd71180410675602fcca"
)
EXPECTED_CANDIDATE_EXECUTION_SHA256 = (
    "48c0e5f8ad4676e17fb4127b3a30ad053beedca8e04e05cfb6fb24f2bb9236f9"
)
EXPECTED_CANDIDATE_EXECUTION_CONTRACT_SHA256 = (
    "331570bacd3ec97474c85f25e7e3cb461ef42b0aa3f442caf3dd1f52314bcbc7"
)
EXPECTED_WRAPPER_SHA256 = (
    "c94c122066fdd19210c8ba64a2a61567b249fad4f69c69cb4236b68cce6ff7b4"
)
EXPECTED_TRAINER_SHA256 = (
    "4c3cb1aa311578038ff656cb7d119103ae5a651135f8ee1c76e37c2c04c1fc75"
)
EXPECTED_FROZEN_EXECUTION_SHA256 = (
    "2f0a579f51fb216d859b2a7e063614e7f76e5a74948067b7d7abd9f2d59e2f70"
)
EXPECTED_SCIENTIFIC_PAYLOAD_SHA256 = (
    "286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e"
)
INHERITED_BASELINE_EXECUTION_CONTRACT_SHA256 = (
    "14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22"
)
BASELINE_PARAMETER_COUNT = 1_061_192
CANDIDATE_PARAMETER_COUNT = 1_061_576
PARAMETER_DELTA = 384
BASELINE_VARIABLE_PREFIX_COUNT = 127
CANDIDATE_VARIABLE_COUNT = 128
EXPECTED_Q_INDEX = 127
EXPECTED_Q_SHAPE = (4, 96)
EXPECTED_Q_DTYPE = "float32"

CANDIDATE_DIRECTORY = Path(__file__).resolve().parent
CANDIDATE_MODEL_PATH = CANDIDATE_DIRECTORY / "model.py"
CANDIDATE_EXECUTION_PATH = CANDIDATE_DIRECTORY / "candidate_execution.py"
CANDIDATE_EXECUTION_CONTRACT_PATH = (
    CANDIDATE_DIRECTORY / "candidate_execution_contract.json"
)
FROZEN_WRAPPER_PATH = FROZEN_PACKAGE_ROOT / "tools" / "train_validation_only.py"
FROZEN_TRAINER_PATH = (
    FROZEN_PACKAGE_SRC / "lap_gnn_tf" / "training" / "trainer.py"
)
FROZEN_EXECUTION_PATH = (
    FROZEN_PACKAGE_SRC / "lap_gnn_tf" / "training" / "execution.py"
)
FROZEN_MANIFEST_PATH = FROZEN_PACKAGE_ROOT / "package_manifest.json"
FROZEN_MARKER_NAME = "VALIDATION_ONLY_COMPLETE.json"
CANDIDATE_MARKER_NAME = "CANDIDATE_VALIDATION_ONLY_COMPLETE.json"
CANDIDATE_CHECKPOINT_RELATIVE = Path("checkpoints") / "best_val_accuracy.keras"
FORBIDDEN_POST_TEST_ARTIFACTS = (
    "TRAINING_COMPLETE.json",
    "run_summary.json",
    "predictions.csv",
    "per_class_metrics.csv",
    "confusion_matrix.csv",
    "confusion_matrix.png",
)


class CandidateValidationOnlyError(RuntimeError):
    """Raised when Step-11 provenance cannot be established fail-closed."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CandidateValidationOnlyError(
            f"Malformed or unreadable {label}: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise CandidateValidationOnlyError(f"{label} must be a JSON object: {path}")
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="",
    )
    os.replace(temporary, path)


def _source_hashes() -> dict[str, str]:
    paths = {
        "candidate_model": CANDIDATE_MODEL_PATH,
        "candidate_execution_adapter": CANDIDATE_EXECUTION_PATH,
        "candidate_execution_contract": CANDIDATE_EXECUTION_CONTRACT_PATH,
        "frozen_validation_only_wrapper": FROZEN_WRAPPER_PATH,
        "frozen_trainer": FROZEN_TRAINER_PATH,
        "frozen_execution": FROZEN_EXECUTION_PATH,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise CandidateValidationOnlyError(f"Required reviewed sources missing: {missing}")
    return {name: _sha256(path) for name, path in paths.items()}


def _verify_candidate_execution_contract() -> dict[str, Any]:
    contract = _json_object(
        CANDIDATE_EXECUTION_CONTRACT_PATH, "candidate execution contract"
    )
    required = {
        "contract_id": CANDIDATE_EXECUTION_CONTRACT_ID,
        "implementation_base": IMPLEMENTATION_BASE,
        "candidate_model_sha256": EXPECTED_CANDIDATE_MODEL_SHA256,
        "frozen_execution_source_sha256": EXPECTED_FROZEN_EXECUTION_SHA256,
        "inherited_baseline_execution_contract_sha256": (
            INHERITED_BASELINE_EXECUTION_CONTRACT_SHA256
        ),
        "selected_mode": "restricted_tf_function",
        "selected_grappler_profile": "G1-A",
        "expected_trainable_variable_count": CANDIDATE_VARIABLE_COUNT,
        "baseline_variable_prefix_count": BASELINE_VARIABLE_PREFIX_COUNT,
    }
    for key, expected in required.items():
        if contract.get(key) != expected:
            raise CandidateValidationOnlyError(
                f"Candidate execution contract field {key!r} drift"
            )
    if contract.get("q") != {
        "dtype": EXPECTED_Q_DTYPE,
        "index": EXPECTED_Q_INDEX,
        "scalar_count": PARAMETER_DELTA,
        "shape": list(EXPECTED_Q_SHAPE),
    }:
        raise CandidateValidationOnlyError("Candidate execution contract Q identity drift")
    if contract.get("eager_exact") != {"status": "unsupported_out_of_scope"}:
        raise CandidateValidationOnlyError("Candidate eager_exact scope drift")
    expected_semantics = {
        "clipping_and_update_arithmetic": "unchanged",
        "loss": "unchanged",
        "mixed_precision": (
            "unchanged_outside_explicit_candidate_residual_boundary_cast"
        ),
        "optimizer": "unchanged",
    }
    if contract.get("inherited_semantics") != expected_semantics:
        raise CandidateValidationOnlyError("Inherited candidate execution semantics drift")
    expected_precision_boundary = {
        "mixed_float16_supported": True,
        "official_global_cast": False,
        "raw_slot_diagnostics_dtype": "float32",
        "residual_input_dtype": "official_global_dtype",
        "slot_compute_dtype": "float32",
    }
    if contract.get("precision_boundary") != expected_precision_boundary:
        raise CandidateValidationOnlyError("Candidate precision boundary drift")
    return contract


def _verify_source_locks(*, require_original_bindings: bool) -> dict[str, str]:
    actual = _source_hashes()
    expected = {
        "candidate_model": EXPECTED_CANDIDATE_MODEL_SHA256,
        "candidate_execution_adapter": EXPECTED_CANDIDATE_EXECUTION_SHA256,
        "candidate_execution_contract": EXPECTED_CANDIDATE_EXECUTION_CONTRACT_SHA256,
        "frozen_validation_only_wrapper": EXPECTED_WRAPPER_SHA256,
        "frozen_trainer": EXPECTED_TRAINER_SHA256,
        "frozen_execution": EXPECTED_FROZEN_EXECUTION_SHA256,
    }
    drift = {
        name: {"expected": expected[name], "actual": digest}
        for name, digest in actual.items()
        if digest != expected[name]
    }
    if drift:
        raise CandidateValidationOnlyError(f"Reviewed source SHA drift: {drift}")
    _verify_candidate_execution_contract()
    manifest = _json_object(FROZEN_MANIFEST_PATH, "frozen package manifest")
    if manifest.get("scientific_payload_sha256") != EXPECTED_SCIENTIFIC_PAYLOAD_SHA256:
        raise CandidateValidationOnlyError("Frozen manifest scientific payload drift")
    if (
        manifest.get("execution_contract_sha256")
        != INHERITED_BASELINE_EXECUTION_CONTRACT_SHA256
    ):
        raise CandidateValidationOnlyError("Inherited execution contract drift")
    if scientific_payload_checksum(FROZEN_PACKAGE_ROOT) != EXPECTED_SCIENTIFIC_PAYLOAD_SHA256:
        raise CandidateValidationOnlyError("Frozen runtime scientific payload drift")
    if require_original_bindings:
        if trainer.LapGNN is not FrozenLapGNN:
            raise CandidateValidationOnlyError(
                "trainer.LapGNN is not the exact reviewed frozen baseline class"
            )
        if (
            trainer.build_restricted_graph_train_step
            is not frozen_build_restricted_graph_train_step
        ):
            raise CandidateValidationOnlyError(
                "trainer restricted builder is not the exact reviewed frozen function"
            )
    return actual


def _verify_run_config(config_path: str | Path) -> tuple[Path, str]:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Candidate validation-only config missing: {path}")
    config = load_config(path)
    locked = config.get("locked", {})
    training = config.get("training", {})
    required = {
        "scientific payload": (
            locked.get("package_checksum"),
            EXPECTED_SCIENTIFIC_PAYLOAD_SHA256,
        ),
        "inherited execution contract": (
            locked.get("execution_contract_sha256"),
            INHERITED_BASELINE_EXECUTION_CONTRACT_SHA256,
        ),
        "baseline parameter lock": (
            locked.get("parameter_count"),
            BASELINE_PARAMETER_COUNT,
        ),
        "gradient execution mode": (
            training.get("gradient_execution_mode"),
            "tf_function",
        ),
        "optimizer execution mode": (
            training.get("optimizer_execution_mode"),
            "restricted_tf_function",
        ),
        "Grappler profile": (training.get("grappler_profile"), "G1-A"),
    }
    drift = {
        name: {"actual": actual, "expected": expected}
        for name, (actual, expected) in required.items()
        if actual != expected
    }
    if drift:
        raise CandidateValidationOnlyError(f"Selected candidate config drift: {drift}")
    return path, _sha256(path)


def _load_frozen_wrapper() -> ModuleType:
    if _sha256(FROZEN_WRAPPER_PATH) != EXPECTED_WRAPPER_SHA256:
        raise CandidateValidationOnlyError("Frozen validation-only wrapper SHA drift")
    spec = importlib.util.spec_from_file_location(
        "_tf_step11_reviewed_train_validation_only", FROZEN_WRAPPER_PATH
    )
    if spec is None or spec.loader is None:
        raise CandidateValidationOnlyError("Cannot load reviewed validation-only wrapper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "run_validation_only", None)):
        raise CandidateValidationOnlyError("Reviewed wrapper entry point is unavailable")
    if module.trainer is not trainer:
        raise CandidateValidationOnlyError("Reviewed wrapper resolved a different trainer module")
    return module


def _verify_no_post_test_artifacts(output_root: Path) -> None:
    forbidden = [
        name
        for name in FORBIDDEN_POST_TEST_ARTIFACTS
        if (output_root / name).exists()
    ]
    forbidden.extend(path.name for path in output_root.glob("test_metrics_*.json"))
    if forbidden:
        raise CandidateValidationOnlyError(
            f"Post-test artifact contamination detected: {sorted(set(forbidden))}"
        )


def _validate_frozen_marker(
    output_root: Path,
    returned_marker: dict[str, Any],
    input_config_path: Path,
    input_config_sha256: str,
) -> tuple[dict[str, Any], Path]:
    marker_path = output_root / FROZEN_MARKER_NAME
    if not marker_path.is_file():
        raise CandidateValidationOnlyError("Frozen validation-only marker missing")
    marker = _json_object(marker_path, "frozen validation-only marker")
    if marker != returned_marker:
        raise CandidateValidationOnlyError("Frozen marker differs from wrapper return value")
    required = {
        "training_validation_completed": True,
        "final_test_skipped": True,
        "test_accessed": False,
        "test_data_constructed": False,
        "test_checkpoint_loaded": False,
        "normal_full_training_completed": False,
        "boundary": "before_resolve_final_checkpoint",
        "trainer_revision_guard_passed": True,
        "intercepted_function_restored": True,
        "trainer_source_sha256": EXPECTED_TRAINER_SHA256,
        "scientific_payload_sha256": EXPECTED_SCIENTIFIC_PAYLOAD_SHA256,
        "input_config_sha256": input_config_sha256,
    }
    for key, expected in required.items():
        if marker.get(key) != expected:
            raise CandidateValidationOnlyError(
                f"Frozen marker field {key!r} is not the reviewed value"
            )
    try:
        marked_config_path = Path(marker.get("input_config_path", "")).resolve()
    except (TypeError, OSError) as exc:
        raise CandidateValidationOnlyError("Frozen marker input config path invalid") from exc
    if marked_config_path != input_config_path:
        raise CandidateValidationOnlyError("Frozen marker input config path drift")
    final_epoch = marker.get("final_observed_epoch")
    if isinstance(final_epoch, bool) or not isinstance(final_epoch, int) or final_epoch < 1:
        raise CandidateValidationOnlyError("Frozen marker final epoch is invalid")
    return marker, marker_path


def _verified_output_json(
    output_root: Path,
    filename: str,
    expected_sha256: Any,
    label: str,
) -> tuple[dict[str, Any], Path, str]:
    path = output_root / filename
    if not path.is_file():
        raise CandidateValidationOnlyError(f"Required {label} missing: {path}")
    digest = _sha256(path)
    if expected_sha256 != digest:
        raise CandidateValidationOnlyError(f"{label} SHA differs from frozen marker")
    return _json_object(path, label), path, digest


def _checkpoint_provenance(output_root: Path) -> dict[str, Any]:
    checkpoint = output_root / CANDIDATE_CHECKPOINT_RELATIVE
    if not checkpoint.is_file():
        raise CandidateValidationOnlyError(f"Candidate checkpoint missing: {checkpoint}")

    import tensorflow as tf

    restored = tf.keras.models.load_model(checkpoint, compile=False)
    if type(restored) is not LearnedLocalResidualSlotLapGNN:
        raise CandidateValidationOnlyError(
            "Candidate checkpoint deserialized as the wrong exact class"
        )
    if restored.count_params() != CANDIDATE_PARAMETER_COUNT:
        raise CandidateValidationOnlyError("Candidate checkpoint parameter count drift")
    slot_layer = getattr(restored, "learned_local_residual_slots", None)
    q = getattr(slot_layer, "Q", None)
    if q is None or tuple(q.shape) != EXPECTED_Q_SHAPE:
        raise CandidateValidationOnlyError("Candidate checkpoint Q shape drift")
    if str(q.dtype) != EXPECTED_Q_DTYPE:
        raise CandidateValidationOnlyError("Candidate checkpoint Q dtype drift")
    q_values = np.asarray(q.numpy(), dtype=np.float32).reshape(-1)
    if q_values.size != PARAMETER_DELTA:
        raise CandidateValidationOnlyError("Candidate checkpoint Q element-count drift")
    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "learned_q_flat_float32_sha256": hashlib.sha256(
            q_values.tobytes(order="C")
        ).hexdigest(),
        "checkpoint_class": type(restored).__name__,
        "checkpoint_parameter_count": restored.count_params(),
        "checkpoint_q_shape": list(q.shape),
        "checkpoint_q_dtype": str(q.dtype),
    }


def _write_candidate_sidecar(
    *,
    output_root: Path,
    returned_marker: dict[str, Any],
    input_config_path: Path,
    input_config_sha256: str,
    source_hashes_before: dict[str, str],
    source_hashes_after: dict[str, str],
    candidate_constructor_injected: bool,
    candidate_builder_injected: bool,
    original_constructor_restored: bool,
    original_builder_restored: bool,
) -> dict[str, Any]:
    sidecar_path = output_root / CANDIDATE_MARKER_NAME
    if sidecar_path.exists():
        raise CandidateValidationOnlyError(f"Candidate sidecar already exists: {sidecar_path}")
    if not all(
        (
            candidate_constructor_injected,
            candidate_builder_injected,
            original_constructor_restored,
            original_builder_restored,
        )
    ):
        raise CandidateValidationOnlyError("Candidate binding injection/restoration not proven")
    if source_hashes_before != source_hashes_after:
        raise CandidateValidationOnlyError("Reviewed sources changed during execution")
    _verify_no_post_test_artifacts(output_root)
    marker, marker_path = _validate_frozen_marker(
        output_root, returned_marker, input_config_path, input_config_sha256
    )
    history, history_path, history_sha = _verified_output_json(
        output_root, "history.json", marker.get("history_sha256"), "history"
    )
    resolved, resolved_path, resolved_sha = _verified_output_json(
        output_root,
        "resolved_config.json",
        marker.get("resolved_config_sha256"),
        "resolved config",
    )
    locked = resolved.get("locked", {})
    if locked.get("package_checksum") != EXPECTED_SCIENTIFIC_PAYLOAD_SHA256:
        raise CandidateValidationOnlyError("Resolved config scientific payload drift")
    if (
        locked.get("execution_contract_sha256")
        != INHERITED_BASELINE_EXECUTION_CONTRACT_SHA256
    ):
        raise CandidateValidationOnlyError("Resolved inherited execution contract drift")
    if locked.get("parameter_count") != BASELINE_PARAMETER_COUNT:
        raise CandidateValidationOnlyError("Resolved baseline parameter lock drift")
    epochs = history.get("epochs")
    if not isinstance(epochs, list) or not epochs or not isinstance(epochs[-1], dict):
        raise CandidateValidationOnlyError("History lacks a valid final epoch")
    if epochs[-1].get("epoch") != marker["final_observed_epoch"]:
        raise CandidateValidationOnlyError("History and frozen marker final epoch differ")
    checkpoint = _checkpoint_provenance(output_root)

    harness_path = Path(__file__).resolve()
    sidecar = {
        "schema_version": 1,
        "harness_version": HARNESS_VERSION,
        "candidate_architecture_id": ARCHITECTURE_ID,
        "candidate_execution_contract_id": CANDIDATE_EXECUTION_CONTRACT_ID,
        "repository_implementation_base": IMPLEMENTATION_BASE,
        "candidate_harness_path": str(harness_path),
        "candidate_harness_sha256": _sha256(harness_path),
        "candidate_model_path": str(CANDIDATE_MODEL_PATH),
        "candidate_model_sha256": EXPECTED_CANDIDATE_MODEL_SHA256,
        "candidate_execution_adapter_path": str(CANDIDATE_EXECUTION_PATH),
        "candidate_execution_adapter_sha256": EXPECTED_CANDIDATE_EXECUTION_SHA256,
        "candidate_execution_contract_path": str(
            CANDIDATE_EXECUTION_CONTRACT_PATH
        ),
        "candidate_execution_contract_sha256": (
            EXPECTED_CANDIDATE_EXECUTION_CONTRACT_SHA256
        ),
        "inherited_baseline_execution_contract_sha256": (
            INHERITED_BASELINE_EXECUTION_CONTRACT_SHA256
        ),
        "candidate_class": LearnedLocalResidualSlotLapGNN.__name__,
        "baseline_config_parameter_lock": BASELINE_PARAMETER_COUNT,
        "actual_candidate_parameter_count": CANDIDATE_PARAMETER_COUNT,
        "parameter_delta": PARAMETER_DELTA,
        "candidate_trainable_variable_count": CANDIDATE_VARIABLE_COUNT,
        "baseline_variable_prefix_count": BASELINE_VARIABLE_PREFIX_COUNT,
        "q_index": EXPECTED_Q_INDEX,
        "q_shape": list(EXPECTED_Q_SHAPE),
        "q_dtype": EXPECTED_Q_DTYPE,
        "frozen_validation_only_wrapper_path": str(FROZEN_WRAPPER_PATH),
        "frozen_validation_only_wrapper_sha256": EXPECTED_WRAPPER_SHA256,
        "frozen_trainer_path": str(FROZEN_TRAINER_PATH),
        "frozen_trainer_sha256": EXPECTED_TRAINER_SHA256,
        "frozen_execution_path": str(FROZEN_EXECUTION_PATH),
        "frozen_execution_sha256": EXPECTED_FROZEN_EXECUTION_SHA256,
        "scientific_payload_sha256": EXPECTED_SCIENTIFIC_PAYLOAD_SHA256,
        "source_artifact_sha256_before": source_hashes_before,
        "source_artifact_sha256_after": source_hashes_after,
        "input_config_path": str(input_config_path),
        "input_config_sha256": input_config_sha256,
        "resolved_config_path": str(resolved_path),
        "resolved_config_sha256": resolved_sha,
        "history_path": str(history_path),
        "history_sha256": history_sha,
        "validation_only_marker_path": str(marker_path),
        "validation_only_marker_sha256": _sha256(marker_path),
        "final_observed_epoch": marker["final_observed_epoch"],
        "candidate_constructor_injected": True,
        "candidate_restricted_builder_injected": True,
        "original_constructor_restored": True,
        "original_restricted_builder_restored": True,
        "training_validation_completed": True,
        "final_test_skipped": True,
        "test_access": False,
        **checkpoint,
    }
    _atomic_json(sidecar_path, sidecar)
    return sidecar


def _run_candidate_validation_only(wrapper: ModuleType, args: Any) -> dict[str, Any]:
    """Inject two authorized bindings and delegate to the reviewed wrapper."""

    original_constructor = trainer.LapGNN
    original_builder = trainer.build_restricted_graph_train_step
    constructed_models: list[LearnedLocalResidualSlotLapGNN] = []
    constructor_injected = False
    builder_injected = False
    input_config_path: Path | None = None
    input_config_sha256: str | None = None
    source_hashes_before: dict[str, str] | None = None
    returned_marker: dict[str, Any] | None = None

    def construct_candidate() -> LearnedLocalResidualSlotLapGNN:
        candidate = LearnedLocalResidualSlotLapGNN()
        constructed_models.append(candidate)
        return candidate

    try:
        source_hashes_before = _verify_source_locks(require_original_bindings=True)
        input_config_path, input_config_sha256 = _verify_run_config(args.config)
        trainer.LapGNN = construct_candidate
        constructor_injected = True
        trainer.build_restricted_graph_train_step = (
            build_candidate_restricted_graph_train_step
        )
        builder_injected = True
        returned_marker = wrapper.run_validation_only(
            args.config,
            args.fer_csv,
            args.prior_root,
            args.output_root,
            wrapper._resource_controls(args),
            no_resume=args.no_resume,
            limit_epochs=args.limit_epochs,
            limit_train_batches=args.limit_train_batches,
            limit_val_batches=args.limit_val_batches,
            limit_train_eval_batches=args.limit_train_eval_batches,
        )
    finally:
        trainer.build_restricted_graph_train_step = original_builder
        trainer.LapGNN = original_constructor

    constructor_restored = trainer.LapGNN is original_constructor
    builder_restored = trainer.build_restricted_graph_train_step is original_builder
    if not constructor_restored or not builder_restored:
        raise CandidateValidationOnlyError("Original trainer bindings were not restored")
    if original_constructor is not FrozenLapGNN:
        raise CandidateValidationOnlyError("Original trainer constructor identity drift")
    if original_builder is not frozen_build_restricted_graph_train_step:
        raise CandidateValidationOnlyError("Original trainer builder identity drift")
    if (
        len(constructed_models) != 1
        or type(constructed_models[0]) is not LearnedLocalResidualSlotLapGNN
    ):
        raise CandidateValidationOnlyError(
            "Frozen trainer did not construct exactly one candidate model"
        )
    constructed = constructed_models[0]
    if constructed.built:
        if constructed.count_params() != CANDIDATE_PARAMETER_COUNT:
            raise CandidateValidationOnlyError("Constructed candidate parameter count drift")
        if len(constructed.trainable_variables) != CANDIDATE_VARIABLE_COUNT:
            raise CandidateValidationOnlyError("Constructed candidate variable count drift")
        if constructed.trainable_variables[EXPECTED_Q_INDEX] is not (
            constructed.learned_local_residual_slots.Q
        ):
            raise CandidateValidationOnlyError("Constructed candidate Q order drift")
    if not isinstance(returned_marker, dict):
        raise CandidateValidationOnlyError("Frozen wrapper did not return a completion marker")
    if source_hashes_before is None or input_config_path is None or input_config_sha256 is None:
        raise CandidateValidationOnlyError("Pre-injection provenance was not established")
    source_hashes_after = _verify_source_locks(require_original_bindings=True)
    return _write_candidate_sidecar(
        output_root=Path(args.output_root).expanduser().resolve(),
        returned_marker=returned_marker,
        input_config_path=input_config_path,
        input_config_sha256=input_config_sha256,
        source_hashes_before=source_hashes_before,
        source_hashes_after=source_hashes_after,
        candidate_constructor_injected=constructor_injected,
        candidate_builder_injected=builder_injected,
        original_constructor_restored=constructor_restored,
        original_builder_restored=builder_restored,
    )


def main(argv: Sequence[str] | None = None) -> int:
    wrapper = _load_frozen_wrapper()
    parser = wrapper.build_parser()
    args = parser.parse_args(argv)
    if args.device.lower().startswith("gpu"):
        import tensorflow as tf

        if not tf.config.list_physical_devices("GPU") and not args.allow_cpu_training:
            parser.error(
                "GPU requested but unavailable; pass --allow-cpu-training explicitly "
                "to override"
            )
    sidecar = _run_candidate_validation_only(wrapper, args)
    print(json.dumps(sidecar, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
