"""Validation-only Gate-A forensic diagnostic for Issue #15.

This tool is deliberately separate from the Step 8 D0-D5 decomposition run.  It
executes only native inference and the reviewed manual D0 identity path, twice
each for every original validation batch.  Per-batch evidence is committed
atomically and Gate-A thresholds are recorded as diagnostic references only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import traceback
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import tensorflow as tf

from lap_gnn_tf.data.graph_generator import GraphBatchGenerator
from lap_gnn_tf.resources import environment_manifest
from lap_gnn_tf.signatures import sha256_file


TOOL_VERSION = "1.0.0"
ISSUE_NUMBER = 15
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
STEP7_TOOL_PATH = Path(__file__).with_name(
    "evaluate_fixed_checkpoint_direct_part_decomposition_probe.py"
)
STEP6_SUPPORT_PATH = Path(__file__).with_name(
    "evaluate_fixed_checkpoint_prior_probe.py"
)

EXPECTED_SCIENTIFIC_BASE_COMMIT = "d14e7e1e3eec2ffbd5339b5a3bd0d5db5ab3de8b"
EXPECTED_HOTFIX_ANCESTOR_COMMIT = "a1b1d279bb9ec388f1d93ad86196e423dc750ad1"
EXPECTED_STEP7_TOOL_SHA256 = (
    "fc60ece71caea14927c4840edfcd527d005737106f60d0bb475b9b1ba79eadd3"
)
EXPECTED_STEP6_SUPPORT_SHA256 = (
    "3a033c977a29e102cfed75282ae7c1062f41feac8bef1b955ae425ec7e4004b3"
)
EXPECTED_SCIENTIFIC_PAYLOAD_SHA256 = (
    "286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "9ec11bb819f97e4fbda432f68da76c1201b8a3f9e06fae9eb30489a528d6ac16"
)
EXPECTED_CHECKPOINT_METADATA_SHA256 = (
    "e62cf8c86f0d6a56c3041911de6397d18f47276f79f03c6a50ca71fa47300a37"
)
EXPECTED_RESOLVED_CONFIG_SHA256 = (
    "3c028dd2f32ebed3a252544e170220b150b5e29920cea865924dddce6aef5a32"
)
EXPECTED_FULL_VALIDATION_SAMPLES = 3589
GATE_A_REFERENCE_TOLERANCE = {
    "prediction_agreement": 1.0,
    "max_abs_logit_difference": 1e-5,
    "max_abs_probability_difference": 1e-6,
}
COMPARISON_ORDER = (
    "native_1_vs_native_2",
    "manual_1_vs_manual_2",
    "native_1_vs_manual_1",
    "native_2_vs_manual_2",
)


class GateAForensicError(RuntimeError):
    """Fail-closed technical diagnostic error."""


def _load_exact_step7_tool():
    actual_step7 = sha256_file(STEP7_TOOL_PATH)
    actual_step6 = sha256_file(STEP6_SUPPORT_PATH)
    if actual_step7 != EXPECTED_STEP7_TOOL_SHA256:
        raise GateAForensicError(
            f"Reviewed Step-7 tool drift: {actual_step7}"
        )
    if actual_step6 != EXPECTED_STEP6_SUPPORT_SHA256:
        raise GateAForensicError(
            f"Reviewed Step-6 support-tool drift: {actual_step6}"
        )
    spec = importlib.util.spec_from_file_location(
        "_issue15_reviewed_step7_probe", STEP7_TOOL_PATH
    )
    if spec is None or spec.loader is None:
        raise GateAForensicError(f"Unable to load Step-7 tool: {STEP7_TOOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


step7 = _load_exact_step7_tool()
step6 = step7.step6


def _sample_ids_sha256(sample_ids: tf.Tensor) -> str:
    values = np.asarray(sample_ids.numpy(), dtype=np.int64)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _comparison(
    left: Mapping[str, tf.Tensor], right: Mapping[str, tf.Tensor]
) -> dict:
    left_logits = np.asarray(left["logits"].numpy(), dtype=np.float64)
    right_logits = np.asarray(right["logits"].numpy(), dtype=np.float64)
    left_probabilities = np.asarray(
        left["probabilities"].numpy(), dtype=np.float64
    )
    right_probabilities = np.asarray(
        right["probabilities"].numpy(), dtype=np.float64
    )
    if left_logits.shape != right_logits.shape:
        raise GateAForensicError(
            f"Forensic logit shape mismatch: {left_logits.shape} != {right_logits.shape}"
        )
    if left_probabilities.shape != right_probabilities.shape:
        raise GateAForensicError(
            "Forensic probability shape mismatch: "
            f"{left_probabilities.shape} != {right_probabilities.shape}"
        )
    left_predictions = left_probabilities.argmax(axis=1)
    right_predictions = right_probabilities.argmax(axis=1)
    result = {
        "prediction_agreement": float(
            np.mean(left_predictions == right_predictions)
        ),
        "max_abs_logit_difference": float(
            np.max(np.abs(left_logits - right_logits), initial=0.0)
        ),
        "max_abs_probability_difference": float(
            np.max(
                np.abs(left_probabilities - right_probabilities), initial=0.0
            )
        ),
    }
    result["within_gate_a_reference"] = bool(
        result["prediction_agreement"]
        == GATE_A_REFERENCE_TOLERANCE["prediction_agreement"]
        and result["max_abs_logit_difference"]
        <= GATE_A_REFERENCE_TOLERANCE["max_abs_logit_difference"]
        and result["max_abs_probability_difference"]
        <= GATE_A_REFERENCE_TOLERANCE["max_abs_probability_difference"]
    )
    return result


def _layer_dtype_record(layer, *, path: str, role: str) -> dict:
    policy = getattr(layer, "dtype_policy", None)
    policy_name = getattr(policy, "name", None)
    if policy_name is None and policy is not None:
        policy_name = policy.__class__.__name__
    input_dtype = getattr(layer, "input_dtype", None)
    return {
        "path": path,
        "role": role,
        "class_name": layer.__class__.__name__,
        "layer_name": getattr(layer, "name", None),
        "dtype_policy": policy_name,
        "compute_dtype": str(getattr(layer, "compute_dtype", None)),
        "variable_dtype": str(getattr(layer, "variable_dtype", None)),
        "input_dtype": (
            tf.dtypes.as_dtype(input_dtype).name if input_dtype is not None else None
        ),
        "autocast": (
            bool(getattr(layer, "autocast"))
            if hasattr(layer, "autocast")
            else None
        ),
    }


def build_dtype_manifest(model) -> dict:
    """Record outer, GNN, context, readout, classifier, and nested policies."""
    records: list[dict] = []
    visited: set[tuple[int, str]] = set()

    def add(layer, *, path: str, role: str, recurse: bool = True) -> None:
        if layer is None or (id(layer), path) in visited:
            return
        visited.add((id(layer), path))
        records.append(_layer_dtype_record(layer, path=path, role=role))
        if not recurse:
            return
        children = list(getattr(layer, "_layers", ()))
        if not children:
            children = list(getattr(layer, "layers", ()))
        for index, child in enumerate(children):
            child_name = getattr(child, "name", f"layer_{index}")
            add(
                child,
                path=f"{path}.{child_name}",
                role=f"nested_{role}",
                recurse=True,
            )

    add(model, path="LapGNN", role="outer_lap_gnn", recurse=False)
    add(getattr(model, "encoder", None), path="LapGNN.encoder", role="encoder")
    gnn = getattr(model, "gnn", None)
    add(gnn, path="LapGNN.gnn", role="gnn_container", recurse=False)
    for index, layer in enumerate(getattr(gnn, "layers_", ())):
        add(
            layer,
            path=f"LapGNN.gnn.layers_[{index}]",
            role="gnn_layer",
        )
    add(
        getattr(gnn, "context", None),
        path="LapGNN.gnn.context",
        role="part_global_context",
    )
    add(getattr(model, "readout", None), path="LapGNN.readout", role="readout")
    add(
        getattr(model, "classifier", None),
        path="LapGNN.classifier",
        role="classifier",
    )
    return {
        "schema_version": 1,
        "global_policy": tf.keras.mixed_precision.global_policy().name,
        "op_determinism_enabled_by_tool": False,
        "layers": records,
    }


def _initial_progress() -> dict:
    return {
        "schema_version": 1,
        "status": "RUNNING",
        "completed_batch_count": 0,
        "completed_sample_count": 0,
        "completed_batch_indices": [],
        "gate_a_reference_tolerances": GATE_A_REFERENCE_TOLERANCE,
        "comparisons": {
            name: {
                "minimum_prediction_agreement": 1.0,
                "maximum_abs_logit_difference": 0.0,
                "maximum_abs_probability_difference": 0.0,
                "reference_exceedance_batch_count": 0,
            }
            for name in COMPARISON_ORDER
        },
    }


def _update_progress(progress: dict, batch_evidence: Mapping) -> None:
    progress["completed_batch_count"] += 1
    progress["completed_sample_count"] += int(batch_evidence["sample_count"])
    progress["completed_batch_indices"].append(int(batch_evidence["batch_index"]))
    for name in COMPARISON_ORDER:
        observed = batch_evidence["comparisons"][name]
        aggregate = progress["comparisons"][name]
        aggregate["minimum_prediction_agreement"] = min(
            aggregate["minimum_prediction_agreement"],
            observed["prediction_agreement"],
        )
        aggregate["maximum_abs_logit_difference"] = max(
            aggregate["maximum_abs_logit_difference"],
            observed["max_abs_logit_difference"],
        )
        aggregate["maximum_abs_probability_difference"] = max(
            aggregate["maximum_abs_probability_difference"],
            observed["max_abs_probability_difference"],
        )
        if not observed["within_gate_a_reference"]:
            aggregate["reference_exceedance_batch_count"] += 1


def evaluate_forensic_batches(
    model,
    batches: Iterable[Mapping[str, tf.Tensor]],
    output_root: Path,
    *,
    expected_model_weights_sha256: str,
) -> dict:
    """Run four ordered D0/native forwards and atomically persist every batch."""
    batch_root = output_root / "batches"
    batch_root.mkdir(parents=True, exist_ok=True)
    progress = _initial_progress()
    step6._write_json(output_root / "progress.json", progress)
    seen_sample_ids: set[int] = set()
    observed_boundaries: list[dict] = []

    for batch_index, source_batch in enumerate(batches):
        step6.validate_batch_schema(source_batch)
        snapshot = step7._snapshot_batch(source_batch)
        sample_ids = np.asarray(
            source_batch["sample_ids"].numpy(), dtype=np.int64
        )
        if any(int(value) in seen_sample_ids for value in sample_ids):
            raise GateAForensicError("Validation sample IDs are not unique")
        seen_sample_ids.update(int(value) for value in sample_ids)

        native_1 = model(source_batch, training=False)
        step7._assert_source_unchanged(source_batch, snapshot)
        native_2 = model(source_batch, training=False)
        step7._assert_source_unchanged(source_batch, snapshot)
        manual_1, trace_1 = step7.manual_forward(
            model, source_batch, step7.CONDITION_D0
        )
        step7._assert_source_unchanged(source_batch, snapshot)
        manual_2, trace_2 = step7.manual_forward(
            model, source_batch, step7.CONDITION_D0
        )
        step7._assert_source_unchanged(source_batch, snapshot)

        comparisons = {
            "native_1_vs_native_2": _comparison(native_1, native_2),
            "manual_1_vs_manual_2": _comparison(manual_1, manual_2),
            "native_1_vs_manual_1": _comparison(native_1, manual_1),
            "native_2_vs_manual_2": _comparison(native_2, manual_2),
        }
        boundary_evidence = {
            "manual_1": trace_1.get("model_boundary"),
            "manual_2": trace_2.get("model_boundary"),
        }
        if batch_index == 0:
            observed_boundaries.append(boundary_evidence)
            step6._write_json(
                output_root / "observed_model_boundary.json", boundary_evidence
            )
        elif boundary_evidence != observed_boundaries[0]:
            raise GateAForensicError(
                "Observed manual model-boundary semantics changed across batches"
            )

        batch_evidence = {
            "schema_version": 1,
            "batch_index": batch_index,
            "sample_count": int(sample_ids.size),
            "sample_ids_sha256": _sample_ids_sha256(
                source_batch["sample_ids"]
            ),
            "node_count": int(tf.shape(source_batch["node_features"])[0].numpy()),
            "edge_count": int(tf.shape(source_batch["edge_features"])[0].numpy()),
            "source_batch_dtypes": {
                name: tensor.dtype.name for name, tensor in source_batch.items()
            },
            "comparisons": comparisons,
            "model_boundary": boundary_evidence,
            "source_batch_unchanged_after_each_forward": True,
            "executed_paths": ["native", "manual_d0"],
            "intervention_conditions_executed": [],
        }
        step6._write_json(
            batch_root / f"batch_{batch_index:05d}.json", batch_evidence
        )
        _update_progress(progress, batch_evidence)
        step6._write_json(output_root / "progress.json", progress)

        current_weights = step6.model_weights_sha256(model)
        if current_weights != expected_model_weights_sha256:
            raise GateAForensicError(
                f"Model weights changed after forensic batch {batch_index}"
            )

    if progress["completed_batch_count"] == 0:
        raise GateAForensicError("No validation batches were produced")
    if progress["completed_sample_count"] != EXPECTED_FULL_VALIDATION_SAMPLES:
        raise GateAForensicError(
            "Full validation sample count drift: "
            f"{progress['completed_sample_count']} != {EXPECTED_FULL_VALIDATION_SAMPLES}"
        )
    progress["status"] = "COMPLETE"
    step6._write_json(output_root / "progress.json", progress)
    return progress


def _require_exact_artifact(path: Path, expected_sha256: str, name: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise GateAForensicError(
            f"Exact {name} SHA-256 mismatch: {actual} != {expected_sha256}"
        )
    return actual


def _artifact_inventory(output_root: Path) -> dict:
    inventory = {}
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "forensic_manifest.json":
            inventory[path.relative_to(output_root).as_posix()] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    return inventory


def build_immutability_evidence(
    checkpoint_path: Path,
    checkpoint_sha256_before: str | None,
    model,
    model_weights_sha256_before: str | None,
    *,
    source_batches_unchanged: bool | None,
) -> dict:
    """Re-hash the immutable checkpoint file and loaded model weights."""
    checkpoint_sha256_after = (
        sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
    )
    model_weights_sha256_after = (
        step6.model_weights_sha256(model) if model is not None else None
    )
    return {
        "schema_version": 1,
        "checkpoint_sha256_before": checkpoint_sha256_before,
        "checkpoint_sha256_after": checkpoint_sha256_after,
        "checkpoint_unchanged": bool(
            checkpoint_sha256_before is not None
            and checkpoint_sha256_before == checkpoint_sha256_after
        ),
        "model_weights_sha256_before": model_weights_sha256_before,
        "model_weights_sha256_after": model_weights_sha256_after,
        "model_weights_unchanged": bool(
            model_weights_sha256_before is not None
            and model_weights_sha256_before == model_weights_sha256_after
        ),
        "source_batches_unchanged": source_batches_unchanged,
    }


def run_diagnostic(args: argparse.Namespace) -> dict:
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Fresh forensic output must not exist: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    checkpoint_path = args.checkpoint.resolve()
    metadata_path = args.checkpoint_metadata.resolve()
    resolved_config_path = args.resolved_config.resolve()
    prior_root = args.prior_root.resolve()
    clean_graph_cache_dir = args.clean_graph_cache_dir.resolve()
    for path in (prior_root, clean_graph_cache_dir):
        if not path.is_dir():
            raise FileNotFoundError(path)

    model = None
    checkpoint_sha256_before = None
    model_weights_sha256_before = None
    progress = None
    resources = None
    failure = None
    checkpoint_cross_check = None
    contract = None
    artifact_hashes = {}

    try:
        artifact_hashes = {
            "checkpoint": _require_exact_artifact(
                checkpoint_path, EXPECTED_CHECKPOINT_SHA256, "checkpoint"
            ),
            "checkpoint_metadata": _require_exact_artifact(
                metadata_path,
                EXPECTED_CHECKPOINT_METADATA_SHA256,
                "checkpoint metadata",
            ),
            "resolved_config": _require_exact_artifact(
                resolved_config_path,
                EXPECTED_RESOLVED_CONFIG_SHA256,
                "resolved config",
            ),
        }
        checkpoint_sha256_before = artifact_hashes["checkpoint"]
        raw_config = step6.load_persisted_resolved_config(resolved_config_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise GateAForensicError("Checkpoint metadata must be a JSON object")
        checkpoint_cross_check = step6.validate_checkpoint_metadata(
            raw_config, metadata
        )
        config = step6.build_runtime_config(raw_config)
        contract = step6.validate_frozen_contract(config)
        if (
            contract["scientific_payload_sha256"]
            != EXPECTED_SCIENTIFIC_PAYLOAD_SHA256
        ):
            raise GateAForensicError("Frozen scientific payload drift")

        resources_config = dict(config.get("resources") or {})
        eval_batch_size = args.eval_batch_size or int(
            resources_config.get(
                "eval_batch_size", config["training"]["batch_size"]
            )
        )
        graph_workers = args.graph_workers or int(
            resources_config.get("graph_workers", 1)
        )
        graph_cache_size = args.graph_cache_size or int(
            resources_config.get("graph_cache_size", 64)
        )
        for name, value in (
            ("eval_batch_size", eval_batch_size),
            ("graph_workers", graph_workers),
            ("graph_cache_size", graph_cache_size),
        ):
            if isinstance(value, bool) or int(value) <= 0:
                raise GateAForensicError(f"{name} must be a positive integer")
        resources = {
            "eval_batch_size": int(eval_batch_size),
            "graph_workers": int(graph_workers),
            "graph_cache_size": int(graph_cache_size),
            "shuffle": False,
            "dataset_split": "val",
            "limit_val_batches": None,
            "op_determinism_enabled_by_tool": False,
            "clean_graph_cache_required": True,
            "graph_rebuild_allowed": False,
        }
        resources.update(
            step6.configure_gpu_memory_growth(
                bool(resources_config.get("memory_growth", True))
            )
        )

        model = step7.load_fixed_checkpoint(checkpoint_path)
        model_weights_sha256_before = step6.model_weights_sha256(model)
        step6._write_json(output_root / "dtype_manifest.json", build_dtype_manifest(model))
        step6._write_json(
            output_root / "provenance.json",
            {
                "schema_version": 1,
                "issue": ISSUE_NUMBER,
                "scientific_base_commit": EXPECTED_SCIENTIFIC_BASE_COMMIT,
                "required_hotfix_ancestor": EXPECTED_HOTFIX_ANCESTOR_COMMIT,
                "step7_tool_sha256": sha256_file(STEP7_TOOL_PATH),
                "step6_support_sha256": sha256_file(STEP6_SUPPORT_PATH),
                "scientific_payload_sha256": EXPECTED_SCIENTIFIC_PAYLOAD_SHA256,
                "artifact_hashes": artifact_hashes,
                "checkpoint_cross_check": checkpoint_cross_check,
                "frozen_contract": contract,
                "resources": resources,
                "diagnostic_only": True,
                "scientific_decomposition_run": False,
                "intervention_conditions_executed": [],
            },
        )

        validation_data = GraphBatchGenerator(
            prior_root=prior_root,
            split="val",
            config=config,
            batch_size=int(eval_batch_size),
            seed=int(config["seed"]),
            shuffle=False,
            graph_cache_size=int(graph_cache_size),
            graph_workers=int(graph_workers),
            clean_graph_cache_dir=clean_graph_cache_dir,
        )
        progress = evaluate_forensic_batches(
            model,
            validation_data.iter_epoch(0),
            output_root,
            expected_model_weights_sha256=model_weights_sha256_before,
        )
    except BaseException as exc:  # preserve diagnostic evidence before re-raising
        failure = {
            "schema_version": 1,
            "status": "TECHNICAL_FORENSIC_FAILURE",
            "exception_type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "scientific_interpretation": None,
        }
        step6._write_json(output_root / "diagnostic_failure.json", failure)

    progress_path = output_root / "progress.json"
    if progress is None and progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))

    immutability = build_immutability_evidence(
        checkpoint_path,
        checkpoint_sha256_before,
        model,
        model_weights_sha256_before,
        source_batches_unchanged=True if failure is None else None,
    )
    step6._write_json(output_root / "immutability.json", immutability)
    if failure is None and (
        not immutability["checkpoint_unchanged"]
        or not immutability["model_weights_unchanged"]
    ):
        failure = {
            "schema_version": 1,
            "status": "TECHNICAL_FORENSIC_FAILURE",
            "exception_type": "GateAForensicError",
            "message": "Checkpoint/model immutability failure",
            "scientific_interpretation": None,
        }
        step6._write_json(output_root / "diagnostic_failure.json", failure)

    manifest = {
        "schema_version": 1,
        "tool_version": TOOL_VERSION,
        "issue": ISSUE_NUMBER,
        "status": "COMPLETE" if failure is None else "TECHNICAL_FORENSIC_FAILURE",
        "split": "val",
        "validation_only": True,
        "diagnostic_only": True,
        "scientific_decomposition_run": False,
        "native_forward_count_per_batch": 2,
        "manual_d0_forward_count_per_batch": 2,
        "intervention_conditions_executed": [],
        "gate_a_reference_tolerances": GATE_A_REFERENCE_TOLERANCE,
        "gate_a_tolerances_are_diagnostic_only": True,
        "stop_on_reference_exceedance": False,
        "progress": progress,
        "immutability": immutability,
        "resources": resources,
        "environment": environment_manifest(),
        "test_access": {
            "test_split_constructed": False,
            "test_metrics_created": False,
            "test_predictions_created": False,
            "test_inference_run": False,
        },
        "training_access": {
            "optimizer_created": False,
            "training_step_created": False,
            "model_fit_called": False,
        },
        "failure": failure,
        "artifacts": _artifact_inventory(output_root),
        "scientific_interpretation": None,
    }
    step6._write_json(output_root / "forensic_manifest.json", manifest)
    if failure is not None:
        raise GateAForensicError(failure["message"])
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure Gate-A native/manual D0 repeatability on validation only; "
            "never execute Step-8 D1-D5 interventions."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-metadata", required=True, type=Path)
    parser.add_argument("--resolved-config", required=True, type=Path)
    parser.add_argument("--prior-root", required=True, type=Path)
    parser.add_argument("--clean-graph-cache-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--graph-workers", type=int)
    parser.add_argument("--graph-cache-size", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_diagnostic(args)
    print(json.dumps(step6._json_ready(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
