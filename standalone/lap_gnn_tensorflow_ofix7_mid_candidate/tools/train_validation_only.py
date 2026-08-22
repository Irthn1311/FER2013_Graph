"""Run the frozen trainer through validation, then stop before final test.

This wrapper deliberately intercepts the first post-training boundary in the
reviewed trainer revision.  It does not duplicate the training loop and it
fails closed if the trainer source or frozen scientific payload has drifted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from lap_gnn_tf.config import load_config
from lap_gnn_tf.resources import ResourceControls
from lap_gnn_tf.training import trainer


WRAPPER_VERSION = "1.0.0"
REVIEWED_BASE_COMMIT = "ee3175f1c2556e1a6fe31cdb059fdca7a85cf688"
EXPECTED_TRAINER_SHA256 = (
    "4c3cb1aa311578038ff656cb7d119103ae5a651135f8ee1c76e37c2c04c1fc75"
)
EXPECTED_SCIENTIFIC_PAYLOAD_SHA256 = (
    "286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e"
)
MARKER_NAME = "VALIDATION_ONLY_COMPLETE.json"
REQUIRED_BOUNDARY_ARTIFACTS = (
    "history.json",
    "resolved_config.json",
    "telemetry.json",
)
FORBIDDEN_POST_TEST_ARTIFACTS = (
    "TRAINING_COMPLETE.json",
    "run_summary.json",
    "predictions.csv",
    "per_class_metrics.csv",
    "confusion_matrix.csv",
    "confusion_matrix.png",
)


class ValidationOnlyExecutionError(RuntimeError):
    """Raised when validation-only completion cannot be proven safely."""


class _ValidationOnlyBoundaryReached(BaseException):
    """Private control-flow sentinel; never interpreted as a training error."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationOnlyExecutionError(
            f"Malformed {description} JSON at validation-only boundary: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValidationOnlyExecutionError(
            f"{description} must be a JSON object at validation-only boundary: {path}"
        )
    return payload


def trainer_source_path() -> Path:
    source = getattr(trainer, "__file__", None)
    if not source:
        raise ValidationOnlyExecutionError(
            "Cannot locate lap_gnn_tf.training.trainer source for revision guard"
        )
    path = Path(source).resolve()
    if path.suffix in {".pyc", ".pyo"}:
        path = path.with_suffix(".py")
    if path.name != "trainer.py" or not path.is_file():
        raise ValidationOnlyExecutionError(
            f"Expected readable trainer.py source for revision guard, got: {path}"
        )
    return path


def verify_trainer_revision() -> tuple[Path, str]:
    path = trainer_source_path()
    actual = _sha256(path)
    if actual != EXPECTED_TRAINER_SHA256:
        raise ValidationOnlyExecutionError(
            "Frozen trainer revision does not match the reviewed validation/test "
            "boundary. Re-review trainer.py and update the wrapper guard in a "
            f"separate reviewed change; expected {EXPECTED_TRAINER_SHA256}, got {actual}."
        )
    if not callable(getattr(trainer, "run_training", None)):
        raise ValidationOnlyExecutionError("Reviewed trainer.run_training is unavailable")
    if not callable(getattr(trainer, "resolve_final_checkpoint", None)):
        raise ValidationOnlyExecutionError(
            "Reviewed trainer.resolve_final_checkpoint boundary is unavailable"
        )
    return path, actual


def _verify_input_config(config_path: str | Path) -> tuple[Path, str]:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Validation-only config does not exist: {path}")
    config = load_config(path)
    actual_payload = config.get("locked", {}).get("package_checksum")
    if actual_payload != EXPECTED_SCIENTIFIC_PAYLOAD_SHA256:
        raise ValidationOnlyExecutionError(
            "Config scientific payload does not match the frozen reviewed reference; "
            f"expected {EXPECTED_SCIENTIFIC_PAYLOAD_SHA256}, got {actual_payload!r}."
        )
    return path, _sha256(path)


def _verify_fresh_output_root(output_root: str | Path) -> Path:
    output = Path(output_root).expanduser().resolve()
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"Validation-only output root is not a directory: {output}")
        if any(output.iterdir()):
            raise FileExistsError(
                f"Fresh validation-only output must be absent or empty: {output}"
            )
    return output


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="",
    )
    os.replace(temporary, path)


def _completion_marker(
    *,
    output_root: Path,
    input_config_path: Path,
    input_config_sha256: str,
    trainer_path: Path,
    trainer_sha256: str,
    limits: dict[str, int | None],
) -> dict[str, Any]:
    missing = [
        name for name in REQUIRED_BOUNDARY_ARTIFACTS
        if not (output_root / name).is_file()
    ]
    if missing:
        raise ValidationOnlyExecutionError(
            f"Validation-only boundary reached without required artifacts: {missing}"
        )
    forbidden = [
        name for name in FORBIDDEN_POST_TEST_ARTIFACTS
        if (output_root / name).exists()
    ]
    forbidden.extend(
        path.name for path in sorted(output_root.glob("test_metrics_*.json"))
    )
    if forbidden:
        raise ValidationOnlyExecutionError(
            "Post-test artifacts exist at the validation-only boundary; refusing to "
            f"declare success: {sorted(set(forbidden))}"
        )
    marker_path = output_root / MARKER_NAME
    if marker_path.exists():
        raise ValidationOnlyExecutionError(
            f"Validation-only completion marker already exists unexpectedly: {marker_path}"
        )

    history_path = output_root / "history.json"
    resolved_config_path = output_root / "resolved_config.json"
    history = _json_object(history_path, "history")
    resolved_config = _json_object(resolved_config_path, "resolved config")
    epochs = history.get("epochs")
    if not isinstance(epochs, list) or not epochs:
        raise ValidationOnlyExecutionError(
            "history.json must contain a non-empty 'epochs' list before validation-only success"
        )
    final_row = epochs[-1]
    if not isinstance(final_row, dict):
        raise ValidationOnlyExecutionError(
            "Final history row must be an object before validation-only success"
        )
    final_epoch = final_row.get("epoch")
    if isinstance(final_epoch, bool) or not isinstance(final_epoch, int) or final_epoch < 1:
        raise ValidationOnlyExecutionError(
            "Final observed epoch must be a positive integer before validation-only success"
        )
    scientific_payload = resolved_config.get("locked", {}).get("package_checksum")
    if scientific_payload != EXPECTED_SCIENTIFIC_PAYLOAD_SHA256:
        raise ValidationOnlyExecutionError(
            "Resolved config scientific payload changed during execution; refusing "
            f"validation-only success: {scientific_payload!r}"
        )

    marker = {
        "schema_version": 1,
        "wrapper_version": WRAPPER_VERSION,
        "reviewed_base_commit": REVIEWED_BASE_COMMIT,
        "boundary": "before_resolve_final_checkpoint",
        "trainer_source_path": str(trainer_path),
        "trainer_source_sha256": trainer_sha256,
        "trainer_revision_guard_passed": True,
        "scientific_payload_sha256": scientific_payload,
        "input_config_path": str(input_config_path),
        "input_config_sha256": input_config_sha256,
        "resolved_config_path": str(resolved_config_path),
        "resolved_config_sha256": _sha256(resolved_config_path),
        "history_path": str(history_path),
        "history_sha256": _sha256(history_path),
        "telemetry_path": str(output_root / "telemetry.json"),
        "seed": resolved_config.get("seed"),
        "run_name": resolved_config.get("run_name"),
        "final_observed_epoch": final_epoch,
        "bounded_limits": limits,
        "training_validation_completed": True,
        "final_test_skipped": True,
        "test_accessed": False,
        "test_data_constructed": False,
        "test_checkpoint_loaded": False,
        "normal_full_training_completed": False,
        "intercepted_function_restored": True,
        "verified_artifacts": list(REQUIRED_BOUNDARY_ARTIFACTS),
    }
    _atomic_json(marker_path, marker)
    return marker


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
    """Execute the reviewed trainer until its first post-validation boundary."""

    trainer_path, trainer_sha256 = verify_trainer_revision()
    input_config_path, input_config_sha256 = _verify_input_config(config_path)
    output = _verify_fresh_output_root(output_root)
    limits = {
        "limit_epochs": limit_epochs,
        "limit_train_batches": limit_train_batches,
        "limit_val_batches": limit_val_batches,
        "limit_train_eval_batches": limit_train_eval_batches,
    }

    original_resolver = trainer.resolve_final_checkpoint
    boundary_signal = _ValidationOnlyBoundaryReached()

    def stop_before_final_test(_config, _policy):
        raise boundary_signal

    boundary_reached = False
    trainer.resolve_final_checkpoint = stop_before_final_test
    try:
        trainer.run_training(
            input_config_path,
            fer_csv,
            prior_root,
            output,
            controls,
            no_resume=no_resume,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            limit_train_eval_batches=limit_train_eval_batches,
            limit_epochs=limit_epochs,
        )
    except _ValidationOnlyBoundaryReached as exc:
        if exc is not boundary_signal:
            raise
        boundary_reached = True
    finally:
        trainer.resolve_final_checkpoint = original_resolver

    if not boundary_reached:
        raise ValidationOnlyExecutionError(
            "Frozen trainer returned without reaching the reviewed post-validation "
            "boundary; final-test isolation cannot be proven."
        )
    return _completion_marker(
        output_root=output,
        input_config_path=input_config_path,
        input_config_sha256=input_config_sha256,
        trainer_path=trainer_path,
        trainer_sha256=trainer_sha256,
        limits=limits,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen TensorFlow trainer through train/validation completion "
            "and stop before final-test checkpoint loading or test-data construction."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--fer-csv", required=True)
    parser.add_argument("--prior-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--graph-workers", type=int, default=2)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--intra-op-threads", type=int, default=0)
    parser.add_argument("--inter-op-threads", type=int, default=0)
    parser.add_argument("--tf-data-prefetch", type=int, default=2)
    parser.add_argument("--tf-data-parallel-calls", type=int, default=1)
    parser.add_argument("--graph-cache-size", type=int, default=64)
    parser.add_argument("--clean-graph-cache-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument(
        "--mixed-precision", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--xla", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--memory-growth", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--allow-cpu-training", action="store_true")
    parser.add_argument("--limit-epochs", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--limit-train-eval-batches", type=int, default=None)
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
        import tensorflow as tf

        if not tf.config.list_physical_devices("GPU") and not args.allow_cpu_training:
            parser.error(
                "GPU requested but unavailable; pass --allow-cpu-training explicitly "
                "to override"
            )
    marker = run_validation_only(
        args.config,
        args.fer_csv,
        args.prior_root,
        args.output_root,
        _resource_controls(args),
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
