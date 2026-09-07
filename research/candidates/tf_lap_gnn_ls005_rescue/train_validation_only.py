"""Validation-only entry point for the registered LAP-GNN LS005 rescue.

The frozen trainer and validation-only wrapper remain lifecycle owners.  This
adapter changes only the loss binding resolved by the registered restricted
TensorFlow train step, and restores it before returning or propagating failure.
"""

from __future__ import annotations

import argparse
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

from lap_gnn_tf.config import load_config, validate_locked_config  # noqa: E402
from lap_gnn_tf.resources import ResourceControls  # noqa: E402
from lap_gnn_tf.signatures import scientific_payload_checksum  # noqa: E402
from lap_gnn_tf.training import evaluator, execution  # noqa: E402
from lap_gnn_tf.training.losses import sparse_cross_entropy as frozen_hard_ce  # noqa: E402
from research.candidates.tf_lap_gnn_ls005_rescue.loss_adapter import (  # noqa: E402
    LABEL_SMOOTHING,
    NUM_CLASSES,
    smoothed_sparse_cross_entropy,
    training_loss_binding,
)


IMPLEMENTATION_BASE = "f8a183520ab1ef7d949e66b48ee7ce7e352f1cb6"
IMPLEMENTATION_STATUS = "LAP_GNN_LS005_RESCUE_IMPLEMENTATION_ONLY"
EXPECTED_SCIENTIFIC_PAYLOAD_SHA256 = (
    "286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e"
)
EXPECTED_TRAINER_SHA256 = (
    "4c3cb1aa311578038ff656cb7d119103ae5a651135f8ee1c76e37c2c04c1fc75"
)
EXPECTED_FROZEN_WRAPPER_SHA256 = (
    "c94c122066fdd19210c8ba64a2a61567b249fad4f69c69cb4236b68cce6ff7b4"
)
EXPECTED_PARAMETER_COUNT = 1_061_192
EXPECTED_TRAINABLE_VARIABLE_COUNT = 127
EXPECTED_CONFIG_DIFF = {
    "loss.label_smoothing": (0.0, LABEL_SMOOTHING),
    "run_name": ("ofix7_mid_seed42", "ofix7_mid_seed42_ls005_rescue"),
}

CANDIDATE_ROOT = Path(__file__).resolve().parent
CANDIDATE_CONFIG_PATH = (
    CANDIDATE_ROOT / "configs" / "fer2013_ofix7_mid_seed42_ls005.yaml"
)
FROZEN_CONFIG_PATH = (
    FROZEN_PACKAGE_ROOT
    / "configs"
    / "fer2013_ofix7_mid_tensorflow_seed42.yaml"
)
FROZEN_TRAINER_PATH = (
    FROZEN_PACKAGE_SRC / "lap_gnn_tf" / "training" / "trainer.py"
)
FROZEN_WRAPPER_PATH = FROZEN_PACKAGE_ROOT / "tools" / "train_validation_only.py"
CANDIDATE_MARKER_NAME = "LS005_VALIDATION_ONLY_COMPLETE.json"

BASELINE_VAL_ACCURACY = 0.6319308999721371
BASELINE_VAL_MACRO_F1 = 0.5938407974340496
BASELINE_MACRO_GAP_PP = 22.69247637556647


class LS005RescueError(RuntimeError):
    """Raised when the isolated rescue contract cannot be proven."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def verify_candidate_config(config_path: str | Path = CANDIDATE_CONFIG_PATH) -> dict:
    path = Path(config_path).expanduser().resolve()
    if path != CANDIDATE_CONFIG_PATH.resolve():
        raise LS005RescueError(
            f"Only the registered candidate config is accepted: {CANDIDATE_CONFIG_PATH}"
        )
    if not path.is_file() or not FROZEN_CONFIG_PATH.is_file():
        raise LS005RescueError("Registered candidate or frozen config is missing")
    frozen = load_config(FROZEN_CONFIG_PATH)
    candidate = load_config(path)
    diff = _deep_diff(frozen, candidate)
    if diff != EXPECTED_CONFIG_DIFF:
        raise LS005RescueError(
            f"Candidate config substantive drift outside registered LS005 change: {diff}"
        )
    validate_locked_config(candidate)
    if candidate.get("loss", {}).get("label_smoothing") != LABEL_SMOOTHING:
        raise LS005RescueError("Registered label smoothing must equal 0.05")
    return candidate


def verify_frozen_guards() -> dict[str, Any]:
    actual = {
        "trainer_sha256": _sha256(FROZEN_TRAINER_PATH),
        "validation_only_wrapper_sha256": _sha256(FROZEN_WRAPPER_PATH),
        "scientific_payload_sha256": scientific_payload_checksum(
            FROZEN_PACKAGE_ROOT
        ),
    }
    expected = {
        "trainer_sha256": EXPECTED_TRAINER_SHA256,
        "validation_only_wrapper_sha256": EXPECTED_FROZEN_WRAPPER_SHA256,
        "scientific_payload_sha256": EXPECTED_SCIENTIFIC_PAYLOAD_SHA256,
    }
    if actual != expected:
        raise LS005RescueError(
            f"Frozen trainer/wrapper/scientific payload drift: {actual}"
        )
    if execution.sparse_cross_entropy is not frozen_hard_ce:
        raise LS005RescueError("Frozen execution binding is not original hard CE")
    if evaluator.sparse_cross_entropy is not frozen_hard_ce:
        raise LS005RescueError("Frozen evaluator binding is not original hard CE")
    return actual


def reject_explicit_test_path(path: str | Path, label: str) -> None:
    """Fail closed when a caller attempts to provide a test-specific input."""

    name = Path(path).name.casefold()
    if name in {"test", "test.csv"} or name.startswith("test_"):
        raise LS005RescueError(f"{label} must not identify a test path")


def _load_frozen_wrapper() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "lap_gnn_tf_frozen_validation_only_ls005", FROZEN_WRAPPER_PATH
    )
    if spec is None or spec.loader is None:
        raise LS005RescueError("Cannot load reviewed frozen validation-only wrapper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="",
    )
    os.replace(temporary, path)


def run_validation_only(
    config_path: str | Path,
    fer_csv: str | Path,
    prior_root: str | Path,
    output_root: str | Path,
    controls: ResourceControls,
    *,
    no_resume: bool = True,
    limit_epochs: int | None = None,
    limit_train_batches: int | None = None,
    limit_val_batches: int | None = None,
    limit_train_eval_batches: int | None = None,
) -> dict[str, Any]:
    """Run the frozen validation-only lifecycle under the temporary LS005 binding."""

    verify_candidate_config(config_path)
    reject_explicit_test_path(fer_csv, "fer_csv")
    reject_explicit_test_path(prior_root, "prior_root")
    frozen_guards = verify_frozen_guards()
    frozen_wrapper = _load_frozen_wrapper()
    original_execution_binding = execution.sparse_cross_entropy
    original_evaluator_binding = evaluator.sparse_cross_entropy
    frozen_marker: dict[str, Any]
    with training_loss_binding() as binding_state:
        if execution.sparse_cross_entropy is not smoothed_sparse_cross_entropy:
            raise LS005RescueError("Training loss binding replacement failed")
        if evaluator.sparse_cross_entropy is not original_evaluator_binding:
            raise LS005RescueError("Evaluator hard-CE binding changed")
        frozen_marker = frozen_wrapper.run_validation_only(
            config_path,
            fer_csv,
            prior_root,
            output_root,
            controls,
            no_resume=no_resume,
            limit_epochs=limit_epochs,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            limit_train_eval_batches=limit_train_eval_batches,
        )

    if execution.sparse_cross_entropy is not original_execution_binding:
        raise LS005RescueError("Original execution loss binding was not restored")
    if evaluator.sparse_cross_entropy is not original_evaluator_binding:
        raise LS005RescueError("Evaluator hard-CE binding changed during execution")
    marker = {
        "schema_version": 1,
        "status": "LS005_VALIDATION_ONLY_COMPLETE",
        "implementation_status": IMPLEMENTATION_STATUS,
        "implementation_base": IMPLEMENTATION_BASE,
        "training_label_smoothing": LABEL_SMOOTHING,
        "num_classes": NUM_CLASSES,
        "training_target_formula": "(1 - 0.05) * one_hot(label, 7) + 0.05 / 7",
        "training_execution_binding_replaced": binding_state[
            "execution_binding_replaced"
        ],
        "training_execution_binding_restored": True,
        "evaluator_hard_ce_binding_unchanged": binding_state[
            "evaluator_binding_unchanged"
        ],
        "frozen_guards": frozen_guards,
        "candidate_config_sha256": _sha256(CANDIDATE_CONFIG_PATH),
        "candidate_config_semantic_diff": {
            key: {"before": before, "after": after}
            for key, (before, after) in EXPECTED_CONFIG_DIFF.items()
        },
        "frozen_validation_only_marker": frozen_marker,
        "training_validation_completed": True,
        "test_accessed": False,
    }
    _atomic_json(Path(output_root).resolve() / CANDIDATE_MARKER_NAME, marker)
    return marker


def classify_outcome(
    *, val_accuracy: float, val_macro_f1: float, clean_train_macro_f1: float
) -> str:
    """Apply the preregistered future single-seed decision precedence."""

    macro_gap_pp = 100.0 * (clean_train_macro_f1 - val_macro_f1)
    gap_improvement_pp = BASELINE_MACRO_GAP_PP - macro_gap_pp
    val_accuracy_change_pp = 100.0 * (val_accuracy - BASELINE_VAL_ACCURACY)
    val_macro_change_pp = 100.0 * (val_macro_f1 - BASELINE_VAL_MACRO_F1)
    gap_rescue = gap_improvement_pp >= 7.5

    if val_accuracy >= 0.6500 and val_macro_f1 >= 0.6200 and macro_gap_pp <= 8.0:
        return "LAP_LS005_STRETCH_RESCUE"
    if (
        val_accuracy >= BASELINE_VAL_ACCURACY
        and val_macro_f1 >= BASELINE_VAL_MACRO_F1
        and macro_gap_pp <= 10.0
    ):
        return "LAP_LS005_RESCUE_PASS"
    if (
        val_accuracy >= BASELINE_VAL_ACCURACY - 0.01
        and gap_rescue
        and val_macro_f1 >= BASELINE_VAL_MACRO_F1 - 0.01
    ):
        return "LAP_LS005_GENERALIZATION_RESCUE_WITH_SMALL_ACCURACY_COST"
    if gap_rescue and (
        val_accuracy_change_pp < -1.0 or val_macro_change_pp < -1.0
    ):
        return "LAP_LS005_OVERREGULARIZED"
    if (
        abs(val_accuracy_change_pp) < 1.0
        and gap_improvement_pp < 7.5
        and abs(val_macro_change_pp) <= 1.0
    ):
        return "LAP_LS005_NO_CLEAR_RESCUE"
    if not gap_rescue and (
        val_accuracy <= BASELINE_VAL_ACCURACY - 0.01
        or val_macro_f1 <= BASELINE_VAL_MACRO_F1 - 0.01
    ):
        return "LAP_LS005_REGRESSION"
    return "LAP_LS005_INCONCLUSIVE"


def build_parser() -> argparse.ArgumentParser:
    frozen_wrapper = _load_frozen_wrapper()
    parser = frozen_wrapper.build_parser()
    parser.description = (
        "Run the frozen validation-only lifecycle with only registered training "
        "label smoothing epsilon=0.05. Test lifecycle remains unavailable."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    verify_candidate_config(config_path)
    if args.device.lower().startswith("gpu"):
        import tensorflow as tf

        if not tf.config.list_physical_devices("GPU") and not args.allow_cpu_training:
            parser.error(
                "GPU requested but unavailable; pass --allow-cpu-training explicitly "
                "to override"
            )
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
    marker = run_validation_only(
        config_path,
        args.fer_csv,
        args.prior_root,
        args.output_root,
        controls,
        no_resume=args.no_resume,
        limit_epochs=args.limit_epochs,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_train_eval_batches=args.limit_train_eval_batches,
    )
    print(json.dumps(marker, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
