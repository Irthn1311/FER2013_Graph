"""Lean validation-only LS0.05 lifecycle preserving frozen training state.

The frozen trainer remains the lifecycle owner. The registered runtime-only
configuration disables its observational per-epoch clean-train pass. After the
frozen wrapper stops before test construction, this adapter reloads the selected
checkpoint and performs exactly one clean-train and one validation hard-CE pass.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
FROZEN_PACKAGE_ROOT = (
    REPOSITORY_ROOT / "standalone" / "lap_gnn_tensorflow_ofix7_mid_candidate"
)
FROZEN_PACKAGE_SRC = FROZEN_PACKAGE_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(FROZEN_PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(FROZEN_PACKAGE_SRC))

import tensorflow as tf  # noqa: E402

from lap_gnn_tf.config import load_config  # noqa: E402
from lap_gnn_tf.data.graph_generator import GraphBatchGenerator  # noqa: E402
from lap_gnn_tf.resources import ResourceControls  # noqa: E402
from lap_gnn_tf.signatures import scientific_payload_checksum  # noqa: E402
from lap_gnn_tf.training import evaluator, execution  # noqa: E402
from lap_gnn_tf.training.evaluator import (  # noqa: E402
    build_compiled_evaluation_step,
    evaluate_batches,
)
from lap_gnn_tf.training.losses import sparse_cross_entropy as frozen_hard_ce  # noqa: E402
from research.candidates.tf_lap_gnn_ls005_rescue.loss_adapter import (  # noqa: E402
    LABEL_SMOOTHING,
    training_loss_binding,
)
from research.candidates.tf_lap_gnn_ls005_rescue.train_validation_only import (  # noqa: E402
    BASELINE_MACRO_GAP_PP,
    BASELINE_VAL_ACCURACY,
    BASELINE_VAL_MACRO_F1,
    classify_outcome,
)
from research.candidates.tf_lap_gnn_ls005_runtime_equivalent.continuation import (  # noqa: E402
    EpochBoundaryContinuationManager,
)
from research.candidates.tf_lap_gnn_ls005_runtime_equivalent.trainer_continuation_adapter import (  # noqa: E402
    continuation_enabled_trainer,
)


IMPLEMENTATION_BASE = "6d89d17b2d3c39b7bf57084f9de4b707a92eb73e"
IMPLEMENTATION_STATUS = "LAP_LS005_RUNTIME_EQUIVALENT_PREPARATION_ONLY"
EXPECTED_SCIENTIFIC_PAYLOAD_SHA256 = (
    "286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e"
)
EXPECTED_EXECUTION_CONTRACT_SHA256 = (
    "14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22"
)
EXPECTED_TRAINER_SHA256 = (
    "4c3cb1aa311578038ff656cb7d119103ae5a651135f8ee1c76e37c2c04c1fc75"
)
EXPECTED_FROZEN_WRAPPER_SHA256 = (
    "c94c122066fdd19210c8ba64a2a61567b249fad4f69c69cb4236b68cce6ff7b4"
)
EXPECTED_LS005_LOSS_ADAPTER_SHA256 = (
    "72435f59cbc138f4e2184b384d0a4e2c85cdd0082655a6371e0a61ee6362e1e5"
)
EXPECTED_PARAMETER_COUNT = 1_061_192
EXPECTED_TRAINABLE_VARIABLE_COUNT = 127
EXPECTED_TRAIN_SAMPLES = 28_709
EXPECTED_VALIDATION_SAMPLES = 3_589
EXPECTED_GPU_COUNT = 2
EXPECTED_GPU_NAME_TOKEN = "T4"

CANDIDATE_ROOT = Path(__file__).resolve().parent
CANDIDATE_CONFIG_PATH = (
    CANDIDATE_ROOT / "configs" / "fer2013_ofix7_mid_seed42_ls005_lean.yaml"
)
FROZEN_CONFIG_PATH = (
    FROZEN_PACKAGE_ROOT / "configs" / "fer2013_ofix7_mid_tensorflow_seed42.yaml"
)
FROZEN_TRAINER_PATH = (
    FROZEN_PACKAGE_SRC / "lap_gnn_tf" / "training" / "trainer.py"
)
FROZEN_WRAPPER_PATH = FROZEN_PACKAGE_ROOT / "tools" / "train_validation_only.py"
LS005_LOSS_ADAPTER_PATH = (
    REPOSITORY_ROOT / "research" / "candidates" / "tf_lap_gnn_ls005_rescue"
    / "loss_adapter.py"
)
CANDIDATE_MARKER_NAME = "LS005_LEAN_VALIDATION_ONLY_COMPLETE.json"
CONTINUATION_IMPLEMENTATION_STATUS = (
    "LAP_LS005_EPOCH_BOUNDARY_EXACT_CONTINUATION_PREPARATION_ONLY"
)

EXPECTED_CONFIG_DIFF = {
    "loss.label_smoothing": (0.0, LABEL_SMOOTHING),
    "run_name": (
        "ofix7_mid_seed42",
        "ofix7_mid_seed42_ls005_runtime_equivalent",
    ),
    "training.eval_train_metrics": (True, False),
}


class RuntimeEquivalentError(RuntimeError):
    """Fail-closed runtime-equivalent lifecycle error."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeEquivalentError(f"Unreadable or malformed {label}: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeEquivalentError(f"{label} must be a JSON object: {path}")
    return value


def _deep_diff(
    baseline: Any, candidate: Any, prefix: str = ""
) -> dict[str, tuple[Any, Any]]:
    if isinstance(baseline, dict) and isinstance(candidate, dict):
        result: dict[str, tuple[Any, Any]] = {}
        for key in sorted(set(baseline) | set(candidate)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in baseline:
                result[path] = (None, candidate[key])
            elif key not in candidate:
                result[path] = (baseline[key], None)
            else:
                result.update(_deep_diff(baseline[key], candidate[key], path))
        return result
    if baseline != candidate:
        return {prefix: (baseline, candidate)}
    return {}


def verify_candidate_config(
    config_path: str | Path = CANDIDATE_CONFIG_PATH,
) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    if path != CANDIDATE_CONFIG_PATH.resolve():
        raise RuntimeEquivalentError(
            f"Only the registered lean config is accepted: {CANDIDATE_CONFIG_PATH}"
        )
    frozen = load_config(FROZEN_CONFIG_PATH)
    candidate = load_config(path)
    diff = _deep_diff(frozen, candidate)
    if diff != EXPECTED_CONFIG_DIFF:
        raise RuntimeEquivalentError(f"Lean config semantic drift: {diff}")
    if candidate["training"].get("eval_train_metrics") is not False:
        raise RuntimeEquivalentError("Per-epoch clean-train evaluation must be disabled")
    if candidate["loss"].get("label_smoothing") != LABEL_SMOOTHING:
        raise RuntimeEquivalentError("Training label smoothing must remain exactly 0.05")
    return candidate


def verify_frozen_guards() -> dict[str, Any]:
    actual = {
        "trainer_sha256": _sha256(FROZEN_TRAINER_PATH),
        "frozen_validation_only_wrapper_sha256": _sha256(FROZEN_WRAPPER_PATH),
        "ls005_loss_adapter_sha256": _sha256(LS005_LOSS_ADAPTER_PATH),
        "scientific_payload_sha256": scientific_payload_checksum(
            FROZEN_PACKAGE_ROOT
        ),
    }
    expected = {
        "trainer_sha256": EXPECTED_TRAINER_SHA256,
        "frozen_validation_only_wrapper_sha256": EXPECTED_FROZEN_WRAPPER_SHA256,
        "ls005_loss_adapter_sha256": EXPECTED_LS005_LOSS_ADAPTER_SHA256,
        "scientific_payload_sha256": EXPECTED_SCIENTIFIC_PAYLOAD_SHA256,
    }
    if actual != expected:
        raise RuntimeEquivalentError(f"Frozen source identity drift: {actual}")
    if execution.sparse_cross_entropy is not frozen_hard_ce:
        raise RuntimeEquivalentError("Training execution binding is not frozen hard CE")
    if evaluator.sparse_cross_entropy is not frozen_hard_ce:
        raise RuntimeEquivalentError("Evaluator binding is not frozen hard CE")
    return actual


def verify_registered_runtime() -> dict[str, Any]:
    """Fail closed unless the future scientific CLI sees exactly two T4 GPUs."""

    devices = tf.config.list_physical_devices("GPU")
    names = [
        str(tf.config.experimental.get_device_details(device).get("device_name", ""))
        for device in devices
    ]
    if len(devices) != EXPECTED_GPU_COUNT or any(
        EXPECTED_GPU_NAME_TOKEN not in name for name in names
    ):
        raise RuntimeEquivalentError(
            "Registered runtime requires exactly two Tesla T4 GPUs; "
            f"observed {names}"
        )
    return {"gpu_count": len(devices), "gpu_names": names}


def reject_explicit_test_path(path: str | Path, label: str) -> None:
    parts = tuple(
        part.casefold()
        for part in os.fspath(path).replace("\\", "/").split("/")
        if part
    )
    name = parts[-1] if parts else ""
    if {"test", "testing", "test_split", "test-split"}.intersection(parts):
        raise RuntimeEquivalentError(f"{label} must not identify a test path")
    if name in {"test", "test.csv"} or name.startswith(("test_", "test-")):
        raise RuntimeEquivalentError(f"{label} must not identify a test path")


def reject_issue60_artifact_path(path: str | Path) -> None:
    normalized = os.fspath(path).replace("\\", "/").casefold()
    if "issue60" in normalized or "issue-60" in normalized:
        raise RuntimeEquivalentError("Issue #60 artifacts are permanently forbidden")


def continuation_scientific_identity(
    controls: ResourceControls | None = None,
) -> dict[str, Any]:
    runtime_resources = None
    if controls is not None:
        runtime_resources = {
            key: controls.__dict__[key]
            for key in sorted(controls.__dict__)
            if key != "clean_graph_cache_dir"
        }
        runtime_resources["clean_graph_cache_dir"] = (
            None
            if controls.clean_graph_cache_dir is None
            else str(Path(controls.clean_graph_cache_dir).expanduser().resolve())
        )
    return {
        "issue61_parent": IMPLEMENTATION_BASE,
        "scientific_payload_sha256": EXPECTED_SCIENTIFIC_PAYLOAD_SHA256,
        "execution_contract_sha256": EXPECTED_EXECUTION_CONTRACT_SHA256,
        "frozen_trainer_sha256": EXPECTED_TRAINER_SHA256,
        "frozen_wrapper_sha256": EXPECTED_FROZEN_WRAPPER_SHA256,
        "ls005_loss_adapter_sha256": EXPECTED_LS005_LOSS_ADAPTER_SHA256,
        "candidate_config_sha256": _sha256(CANDIDATE_CONFIG_PATH),
        "seed": 42,
        "label_smoothing": LABEL_SMOOTHING,
        "checkpoint_policy": "earliest_strict_max_validation_accuracy",
        "runtime_resources": runtime_resources,
        "test_access": False,
        "issue60_artifacts_used": False,
    }


def _load_frozen_wrapper() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "lap_gnn_tf_frozen_validation_only_ls005_lean", FROZEN_WRAPPER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeEquivalentError("Cannot load frozen validation-only wrapper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model_variables_sha256(model: tf.keras.Model) -> str:
    digest = hashlib.sha256()
    for variable in model.variables:
        value = variable.numpy()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _selected_checkpoint_proof(output_root: Path) -> tuple[Path, int, dict[str, Any]]:
    history_payload = _json_object(output_root / "history.json", "history")
    rows = history_payload.get("epochs")
    if not isinstance(rows, list) or not rows:
        raise RuntimeEquivalentError("Complete training history is unavailable")
    if [row.get("epoch") for row in rows] != list(range(1, len(rows) + 1)):
        raise RuntimeEquivalentError("Training history epoch sequence drift")
    if any(row.get("train_eval_performed") is not False for row in rows):
        raise RuntimeEquivalentError("Per-epoch clean-train evaluation unexpectedly ran")
    if any(
        row.get(key) is not None
        for row in rows
        for key in ("train_eval_loss", "train_accuracy", "train_macro_f1")
    ):
        raise RuntimeEquivalentError("Per-epoch clean-train metrics unexpectedly exist")
    best_accuracy = max(float(row["val_accuracy"]) for row in rows)
    selected = next(row for row in rows if float(row["val_accuracy"]) == best_accuracy)
    checkpoint_path = output_root / "checkpoints" / "best_val_accuracy.keras"
    metadata_path = output_root / "checkpoints" / "best_val_accuracy.metadata.json"
    if not checkpoint_path.is_file() or not metadata_path.is_file():
        raise RuntimeEquivalentError("Selected checkpoint artifacts are missing")
    metadata = _json_object(metadata_path, "selected checkpoint metadata")
    if metadata.get("epoch") != selected["epoch"]:
        raise RuntimeEquivalentError(
            "Selected checkpoint is not the earliest strict maximum validation accuracy"
        )
    metrics = metadata.get("validation_metrics")
    if not isinstance(metrics, dict):
        raise RuntimeEquivalentError("Selected checkpoint validation metrics are missing")
    for metric, row_key in (
        ("accuracy", "val_accuracy"),
        ("macro_f1", "val_macro_f1"),
        ("loss", "val_loss"),
    ):
        if float(metrics.get(metric, float("nan"))) != float(selected[row_key]):
            raise RuntimeEquivalentError(
                f"Selected checkpoint metadata/history mismatch: {metric}"
            )
    return checkpoint_path, int(selected["epoch"]), metadata


def evaluate_selected_checkpoint_once(
    config: dict[str, Any],
    prior_root: str | Path,
    output_root: str | Path,
    controls: ResourceControls,
) -> dict[str, Any]:
    """Reload selected checkpoint, then run one clean-train and one val hard CE."""

    output = Path(output_root).resolve()
    checkpoint_path, selected_epoch, metadata = _selected_checkpoint_proof(output)
    if evaluator.sparse_cross_entropy is not frozen_hard_ce:
        raise RuntimeEquivalentError("Final evaluator is not frozen hard CE")
    model = tf.keras.models.load_model(checkpoint_path, compile=False)
    if model.count_params() != EXPECTED_PARAMETER_COUNT:
        raise RuntimeEquivalentError("Reloaded selected checkpoint parameter drift")
    if len(model.trainable_variables) != EXPECTED_TRAINABLE_VARIABLE_COUNT:
        raise RuntimeEquivalentError("Reloaded selected checkpoint variable-count drift")
    weights_before = _model_variables_sha256(model)
    clean_config = copy.deepcopy(config)
    clean_config["graph"]["prior_corruption"]["enabled"] = False
    clean_train = GraphBatchGenerator(
        prior_root,
        "train",
        clean_config,
        controls.eval_batch_size,
        int(config["seed"]),
        False,
        controls.graph_cache_size,
        graph_workers=controls.graph_workers,
        clean_graph_cache_dir=controls.clean_graph_cache_dir,
    )
    validation = GraphBatchGenerator(
        prior_root,
        "val",
        config,
        controls.eval_batch_size,
        int(config["seed"]),
        False,
        controls.graph_cache_size,
        graph_workers=controls.graph_workers,
        clean_graph_cache_dir=controls.clean_graph_cache_dir,
    )
    if len(clean_train.dataset) != EXPECTED_TRAIN_SAMPLES:
        raise RuntimeEquivalentError("Final clean-train sample count drift")
    if len(validation.dataset) != EXPECTED_VALIDATION_SAMPLES:
        raise RuntimeEquivalentError("Final validation sample count drift")
    evaluate_step = build_compiled_evaluation_step(model)
    clean_metrics = evaluate_batches(
        model,
        clean_train.as_dataset(0, prefetch=controls.tf_data_prefetch),
        evaluate_step=evaluate_step,
    )
    validation_metrics = evaluate_batches(
        model,
        validation.as_dataset(0, prefetch=controls.tf_data_prefetch),
        evaluate_step=evaluate_step,
    )
    if sum(clean_metrics.get("support_per_class", [])) != EXPECTED_TRAIN_SAMPLES:
        raise RuntimeEquivalentError("Final clean-train evaluation is incomplete")
    if sum(validation_metrics.get("support_per_class", [])) != EXPECTED_VALIDATION_SAMPLES:
        raise RuntimeEquivalentError("Final validation evaluation is incomplete")
    weights_after = _model_variables_sha256(model)
    if weights_after != weights_before:
        raise RuntimeEquivalentError("Final evaluation changed selected model variables")
    train_macro = float(clean_metrics["macro_f1"])
    val_macro = float(validation_metrics["macro_f1"])
    val_accuracy = float(validation_metrics["accuracy"])
    clean_train_accuracy = float(clean_metrics["accuracy"])
    accuracy_gap_pp = 100.0 * (clean_train_accuracy - val_accuracy)
    macro_gap_pp = 100.0 * (train_macro - val_macro)
    return {
        "selected_checkpoint": "best_val_accuracy.keras",
        "selected_epoch": selected_epoch,
        "selection_policy": "earliest_strict_max_val_accuracy",
        "checkpoint_metadata": metadata,
        "checkpoint_reloaded_compile_false": True,
        "final_clean_train_hard_ce_evaluations": 1,
        "final_validation_hard_ce_evaluations": 1,
        "clean_train_augmentation": False,
        "clean_train": clean_metrics,
        "validation": validation_metrics,
        "accuracy_gap_pp": accuracy_gap_pp,
        "macro_gap_pp": macro_gap_pp,
        "delta_val_accuracy_pp": 100.0 * (
            val_accuracy - BASELINE_VAL_ACCURACY
        ),
        "delta_val_macro_f1_pp": 100.0 * (
            val_macro - BASELINE_VAL_MACRO_F1
        ),
        "macro_gap_improvement_pp": BASELINE_MACRO_GAP_PP - macro_gap_pp,
        "registered_decision": classify_outcome(
            val_accuracy=val_accuracy,
            val_macro_f1=val_macro,
            clean_train_macro_f1=train_macro,
        ),
        "model_variables_sha256_before": weights_before,
        "model_variables_sha256_after": weights_after,
        "model_variables_immutable": True,
        "test_access": False,
    }


def run_validation_only(
    config_path: str | Path,
    fer_csv: str | Path,
    prior_root: str | Path,
    output_root: str | Path,
    controls: ResourceControls,
    *,
    continuation_root: str | Path,
    resume_capsule_root: str | Path | None = None,
    no_resume: bool = True,
    limit_epochs: int | None = None,
    limit_train_batches: int | None = None,
    limit_val_batches: int | None = None,
) -> dict[str, Any]:
    """Run frozen training without per-epoch clean eval, then final hard-CE endpoints."""

    if not no_resume:
        raise RuntimeEquivalentError("Resume is forbidden")
    if resume_capsule_root is not None:
        reject_issue60_artifact_path(resume_capsule_root)
    registered_runtime = verify_registered_runtime()
    verify_candidate_config(config_path)
    reject_explicit_test_path(fer_csv, "fer_csv")
    reject_explicit_test_path(prior_root, "prior_root")
    if controls.clean_graph_cache_dir is not None:
        reject_explicit_test_path(
            controls.clean_graph_cache_dir, "controls.clean_graph_cache_dir"
        )
    guards = verify_frozen_guards()
    frozen_wrapper = _load_frozen_wrapper()
    continuation_manager = EpochBoundaryContinuationManager(
        continuation_root,
        continuation_scientific_identity(controls),
        resume_root=resume_capsule_root,
    )
    original_execution = execution.sparse_cross_entropy
    original_evaluator = evaluator.sparse_cross_entropy
    with training_loss_binding() as binding_state:
        with continuation_enabled_trainer(continuation_manager):
            frozen_marker = frozen_wrapper.run_validation_only(
                CANDIDATE_CONFIG_PATH,
                fer_csv,
                prior_root,
                output_root,
                controls,
                no_resume=True,
                limit_epochs=limit_epochs,
                limit_train_batches=limit_train_batches,
                limit_val_batches=limit_val_batches,
                limit_train_eval_batches=None,
            )
    if execution.sparse_cross_entropy is not original_execution:
        raise RuntimeEquivalentError("Training loss binding was not restored")
    if evaluator.sparse_cross_entropy is not original_evaluator:
        raise RuntimeEquivalentError("Evaluator hard-CE binding changed")
    config = verify_candidate_config()
    selected = evaluate_selected_checkpoint_once(
        config, prior_root, output_root, controls
    )
    marker = {
        "schema_version": 1,
        "status": "LS005_LEAN_VALIDATION_ONLY_COMPLETE",
        "implementation_status": IMPLEMENTATION_STATUS,
        "continuation_implementation_status": CONTINUATION_IMPLEMENTATION_STATUS,
        "implementation_base": IMPLEMENTATION_BASE,
        "runtime_only_change": "per_epoch_clean_train_evaluation_removed",
        "per_epoch_clean_train_evaluations": 0,
        "validation_evaluation_every_epoch": True,
        "training_label_smoothing": LABEL_SMOOTHING,
        "training_target_formula": "(1 - 0.05) * one_hot(label, 7) + 0.05 / 7",
        "training_execution_binding_replaced": binding_state[
            "execution_binding_replaced"
        ],
        "training_execution_binding_restored": True,
        "evaluator_hard_ce_binding_unchanged": binding_state[
            "evaluator_binding_unchanged"
        ],
        "frozen_guards": guards,
        "registered_runtime": registered_runtime,
        "continuation": {
            "write_root": str(Path(continuation_root).expanduser().resolve()),
            "resumed_from_capsule": resume_capsule_root is not None,
            "resume_root": (
                None
                if resume_capsule_root is None
                else str(Path(resume_capsule_root).expanduser().resolve())
            ),
            "latest": _json_object(
                Path(continuation_root).expanduser().resolve() / "LATEST.json",
                "continuation latest pointer",
            ),
            "exact_state_required": True,
            "partial_epoch_resume_forbidden": True,
            "issue60_artifacts_used": False,
        },
        "candidate_config_sha256": _sha256(CANDIDATE_CONFIG_PATH),
        "candidate_config_semantic_diff": {
            key: {"before": before, "after": after}
            for key, (before, after) in EXPECTED_CONFIG_DIFF.items()
        },
        "frozen_validation_only_marker": frozen_marker,
        "selected_checkpoint_evaluation": selected,
        "training_validation_completed": True,
        "final_test_skipped": True,
        "test_accessed": False,
        "issue60_partial_checkpoint_loaded": False,
    }
    _atomic_json(Path(output_root).resolve() / CANDIDATE_MARKER_NAME, marker)
    return marker


def build_parser() -> argparse.ArgumentParser:
    frozen_parser = _load_frozen_wrapper().build_parser()
    parser = argparse.ArgumentParser(
        description=(
            "Runtime-equivalent LS0.05 validation-only lifecycle: no per-epoch "
            "clean-train evaluation; selected-checkpoint clean-train/val hard CE once."
        ),
        parents=[frozen_parser],
        add_help=False,
        conflict_handler="resolve",
    )
    removed_destinations = {"limit_train_eval_batches", "allow_cpu_training"}
    for action in list(parser._actions):
        if action.dest not in removed_destinations:
            continue
        for option in action.option_strings:
            parser._option_string_actions.pop(option, None)
        parser._actions.remove(action)
        for group in parser._action_groups:
            if action in group._group_actions:
                group._group_actions.remove(action)
    parser.add_argument("--continuation-root", required=True)
    parser.add_argument("--resume-capsule-root", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    verify_candidate_config(config_path)
    if not args.device.lower().startswith("gpu"):
        parser.error("Registered execution requires the GPU device class")
    controls = ResourceControls(
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
    try:
        marker = run_validation_only(
            config_path,
            args.fer_csv,
            args.prior_root,
            args.output_root,
            controls,
            continuation_root=args.continuation_root,
            resume_capsule_root=args.resume_capsule_root,
            no_resume=args.no_resume,
            limit_epochs=args.limit_epochs,
            limit_train_batches=args.limit_train_batches,
            limit_val_batches=args.limit_val_batches,
        )
    except RuntimeEquivalentError as exc:
        parser.error(str(exc))
    print(json.dumps(marker, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
