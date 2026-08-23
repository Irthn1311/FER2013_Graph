"""Validation-only fixed-topology prior-intervention probe for Issue #9.

This is an inference-only harness. It operates on already constructed graph
batches, evaluates all registered conditions from the same original batch, and
never constructs a train or test split.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import tensorflow as tf

from lap_gnn_tf.config import canonical_config_hash, load_config, validate_locked_config
from lap_gnn_tf.constants import (
    EDGE_FEATURE_NAMES,
    EXPECTED_PARAMETER_COUNT,
    NODE_FEATURE_NAMES,
)
from lap_gnn_tf.data.graph_generator import GraphBatchGenerator
from lap_gnn_tf.model import LapGNN
from lap_gnn_tf.resources import environment_manifest
from lap_gnn_tf.signatures import scientific_payload_checksum, sha256_file
from lap_gnn_tf.training.evaluator import build_compiled_evaluation_step
from lap_gnn_tf.training.metrics import classification_metrics


TOOL_VERSION = "1.0.0"
ISSUE_NUMBER = 9
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_SCIENTIFIC_PAYLOAD_SHA256 = (
    "286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e"
)
EXPECTED_EXECUTION_CONTRACT_SHA256 = (
    "14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22"
)
EXPECTED_GRAPH_SIGNATURE = (
    "1c7597b170fd8604056ab7787fd2880d6e84f3025962fc4b6c8fb3e3faf8e1e8"
)
EXPECTED_FEATURE_SIGNATURE = (
    "752538062fa2e40d9615c650c529e9f4117f33a030b74d281b5b21fa573731fc"
)
EXPECTED_PRIOR_SIGNATURE = (
    "ea888bab9c003af9b279719025da7c39f90537179411326c2c3119fc8c3f0824"
)
EXPECTED_DATASET_SPLIT_SIGNATURE = "fer2013_train28709_val3589_test3589"

EXPECTED_NODE_FEATURE_NAMES = (
    "intensity",
    "gx",
    "gy",
    "x_norm",
    "y_norm",
    "face_mask",
    *(f"part_soft_{index}" for index in range(13)),
    *(f"distance_map_{index}" for index in range(12)),
    "landmark_missing_flag",
    "grad_mag",
    "local_mean_3x3",
    "local_std_3x3",
    "laplacian_abs",
    "center_surround",
)
EXPECTED_EDGE_FEATURE_NAMES = (
    "dx",
    "dy",
    "spatial_dist",
    "abs_intensity_diff",
    "abs_grad_mag_diff",
    "abs_laplacian_diff",
    "part_similarity",
    "same_dominant_part",
)

CONDITION_OFFICIAL = "official"
CONDITION_DIRECT_ZERO = "direct_part_path_zero_fixed_graph"
CONDITION_SEMANTIC_ZERO = "semantic_prior_zero_fixed_graph"
CONDITIONS = (
    CONDITION_OFFICIAL,
    CONDITION_DIRECT_ZERO,
    CONDITION_SEMANTIC_ZERO,
)

DIRECT_FIELDS = ("part_soft", "valid_part_mask")
SEMANTIC_FIELDS = DIRECT_FIELDS + ("node_features", "edge_features")
NODE_SEMANTIC_SLICE = slice(5, 32)
NODE_VISUAL_BASE_SLICE = slice(0, 5)
NODE_VISUAL_DETAIL_SLICE = slice(32, 37)
EDGE_SEMANTIC_SLICE = slice(6, 8)
EDGE_VISUAL_SLICE = slice(0, 6)

INTERVENTION_SPECS = {
    CONDITION_OFFICIAL: {
        "changed_tensor_fields": [],
        "changed_node_feature_columns": [],
        "changed_edge_feature_columns": [],
        "topology_fixed": True,
        "description": "Identity condition; original validation batch values.",
    },
    CONDITION_DIRECT_ZERO: {
        "changed_tensor_fields": list(DIRECT_FIELDS),
        "changed_node_feature_columns": [],
        "changed_edge_feature_columns": [],
        "topology_fixed": True,
        "description": (
            "Zero direct part_soft and valid_part_mask paths after graph construction."
        ),
    },
    CONDITION_SEMANTIC_ZERO: {
        "changed_tensor_fields": list(SEMANTIC_FIELDS),
        "changed_node_feature_columns": list(range(5, 32)),
        "changed_edge_feature_columns": [6, 7],
        "topology_fixed": True,
        "description": (
            "C1 plus zero semantic-prior node columns 5..31 and edge columns 6..7; "
            "official MediaPipe-derived topology remains fixed."
        ),
    },
}


class PriorProbeError(RuntimeError):
    """Fail-closed Issue #9 harness error."""


ALREADY_INITIALIZED_MEMORY_GROWTH_ERROR = (
    "Physical devices cannot be modified after being initialized"
)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping) -> None:
    _write_text_atomic(
        path,
        json.dumps(_json_ready(dict(payload)), indent=2, sort_keys=True) + "\n",
    )


def _require_positive(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or int(value) <= 0:
        raise PriorProbeError(f"{name} must be a positive integer")
    return int(value)


def load_persisted_resolved_config(path: str | Path) -> dict:
    """Load the persisted JSON mapping without YAML scalar reinterpretation."""
    config_path = Path(path)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PriorProbeError(
            f"Persisted resolved config must be valid UTF-8 JSON: {config_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise PriorProbeError("Persisted resolved config must be a JSON object")
    return payload


def build_runtime_config(raw_resolved_config: Mapping) -> dict:
    """Isolate inference-only runtime use from persisted identity state."""
    return copy.deepcopy(dict(raw_resolved_config))


def configure_gpu_memory_growth(requested: bool) -> dict:
    """Apply the frozen runtime policy while recording its explicit outcome."""
    status = "not_requested"
    device_statuses = []
    if requested:
        gpus = list(tf.config.list_physical_devices("GPU"))
        status = "no_gpu_detected" if not gpus else "configured"
        for gpu in gpus:
            device_status = "configured"
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as exc:
                if ALREADY_INITIALIZED_MEMORY_GROWTH_ERROR not in str(exc):
                    raise PriorProbeError(
                        "Unable to configure GPU memory growth"
                    ) from exc
                device_status = "already_initialized"
                status = "already_initialized"
            device_statuses.append(
                {"device": getattr(gpu, "name", str(gpu)), "status": device_status}
            )
    return {
        "memory_growth_requested": bool(requested),
        "memory_growth_status": status,
        "memory_growth_devices": device_statuses,
    }


def validate_frozen_contract(config: Mapping, package_root: Path = PACKAGE_ROOT) -> dict:
    """Fail closed unless the current frozen schema/signatures are exact."""
    validate_locked_config(dict(config))
    if tuple(NODE_FEATURE_NAMES) != EXPECTED_NODE_FEATURE_NAMES:
        raise PriorProbeError("Frozen 37-channel node feature order drift")
    if tuple(EDGE_FEATURE_NAMES) != EXPECTED_EDGE_FEATURE_NAMES:
        raise PriorProbeError("Frozen 8-channel edge feature order drift")

    locked = dict(config.get("locked") or {})
    expected_locked = {
        "package_checksum": EXPECTED_SCIENTIFIC_PAYLOAD_SHA256,
        "execution_contract_sha256": EXPECTED_EXECUTION_CONTRACT_SHA256,
        "graph_signature": EXPECTED_GRAPH_SIGNATURE,
        "feature_signature": EXPECTED_FEATURE_SIGNATURE,
        "prior_signature": EXPECTED_PRIOR_SIGNATURE,
        "dataset_split_signature": EXPECTED_DATASET_SPLIT_SIGNATURE,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
    }
    mismatches = {
        key: {"expected": expected, "actual": locked.get(key)}
        for key, expected in expected_locked.items()
        if locked.get(key) != expected
    }
    if mismatches:
        raise PriorProbeError(f"Frozen config/signature drift: {mismatches}")

    payload_sha256 = scientific_payload_checksum(package_root)
    if payload_sha256 != EXPECTED_SCIENTIFIC_PAYLOAD_SHA256:
        raise PriorProbeError(
            "Frozen scientific payload drift: "
            f"{payload_sha256} != {EXPECTED_SCIENTIFIC_PAYLOAD_SHA256}"
        )
    return {
        "node_width": len(EXPECTED_NODE_FEATURE_NAMES),
        "edge_width": len(EXPECTED_EDGE_FEATURE_NAMES),
        "part_width": 13,
        "scientific_payload_sha256": payload_sha256,
        "locked": expected_locked,
    }


def infer_checkpoint_metadata_path(checkpoint: str | Path) -> Path:
    checkpoint_path = Path(checkpoint)
    return checkpoint_path.with_name(f"{checkpoint_path.stem}.metadata.json")


def validate_checkpoint_metadata(config: Mapping, metadata: Mapping) -> dict:
    """Cross-check persisted checkpoint metadata against the resolved config."""
    locked = dict(config.get("locked") or {})
    expected = {
        "config_hash": canonical_config_hash(dict(config)),
        "seed": int(config.get("seed")),
        "package_checksum": EXPECTED_SCIENTIFIC_PAYLOAD_SHA256,
        "execution_contract_sha256": EXPECTED_EXECUTION_CONTRACT_SHA256,
        "graph_signature": locked.get("graph_signature"),
        "feature_signature": locked.get("feature_signature"),
        "prior_signature": locked.get("prior_signature"),
        "dataset_split_signature": locked.get("dataset_split_signature"),
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise PriorProbeError(f"Checkpoint/config metadata mismatch: {mismatches}")
    epoch = metadata.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise PriorProbeError(f"Invalid checkpoint epoch: {epoch!r}")
    return {"checkpoint_epoch": epoch, "cross_checked_fields": sorted(expected)}


def validate_batch_schema(batch: Mapping[str, tf.Tensor]) -> dict:
    """Validate the exact GraphBatchGenerator tensor contract."""
    expected = GraphBatchGenerator.output_signature()
    if set(batch) != set(expected):
        missing = sorted(set(expected) - set(batch))
        extra = sorted(set(batch) - set(expected))
        raise PriorProbeError(f"Graph batch fields drift: missing={missing}, extra={extra}")

    for name, spec in expected.items():
        tensor = batch[name]
        if not tf.is_tensor(tensor):
            raise PriorProbeError(f"Batch field {name!r} is not a TensorFlow tensor")
        if tensor.dtype != spec.dtype:
            raise PriorProbeError(
                f"Batch field {name!r} dtype drift: {tensor.dtype} != {spec.dtype}"
            )
        if not spec.shape.is_compatible_with(tensor.shape):
            raise PriorProbeError(
                f"Batch field {name!r} shape drift: {tensor.shape} vs {spec.shape}"
            )

    node_count = int(tf.shape(batch["node_features"])[0].numpy())
    edge_count = int(tf.shape(batch["edge_features"])[0].numpy())
    graph_count = int(tf.shape(batch["labels"])[0].numpy())
    if int(tf.shape(batch["edge_index"])[1].numpy()) != edge_count:
        raise PriorProbeError("edge_index and edge_features counts differ")
    if int(tf.reduce_sum(batch["graph_node_counts"]).numpy()) != node_count:
        raise PriorProbeError("graph_node_counts do not sum to the node count")
    if int(tf.reduce_sum(batch["graph_edge_counts"]).numpy()) != edge_count:
        raise PriorProbeError("graph_edge_counts do not sum to the edge count")
    for name in ("sample_ids", "valid_part_mask", "valid_anchor_mask", "detected", "landmark_missing_flag", "image_48"):
        if int(tf.shape(batch[name])[0].numpy()) != graph_count:
            raise PriorProbeError(f"Graph-level field {name!r} count drift")
    for name in ("node_types", "node_graph_index", "coordinates", "anchor_mask", "part_soft", "face_mask"):
        if int(tf.shape(batch[name])[0].numpy()) != node_count:
            raise PriorProbeError(f"Node-level field {name!r} count drift")
    if int(tf.shape(batch["edge_graph_index"])[0].numpy()) != edge_count:
        raise PriorProbeError("edge_graph_index count drift")
    return {
        "graphs": graph_count,
        "nodes": node_count,
        "edges": edge_count,
        "node_width": int(batch["node_features"].shape[1]),
        "edge_width": int(batch["edge_features"].shape[1]),
        "part_width": int(batch["part_soft"].shape[1]),
    }


def apply_intervention(
    batch: Mapping[str, tf.Tensor], condition: str
) -> dict[str, tf.Tensor]:
    """Return a non-mutating post-graph transformation for one registered condition."""
    if condition not in CONDITIONS:
        raise PriorProbeError(f"Unknown prior-probe condition: {condition!r}")
    validate_batch_schema(batch)
    transformed = dict(batch)
    if condition in (CONDITION_DIRECT_ZERO, CONDITION_SEMANTIC_ZERO):
        transformed["part_soft"] = tf.zeros_like(batch["part_soft"])
        transformed["valid_part_mask"] = tf.zeros_like(batch["valid_part_mask"])
    if condition == CONDITION_SEMANTIC_ZERO:
        node_features = batch["node_features"]
        transformed["node_features"] = tf.concat(
            (
                node_features[:, NODE_VISUAL_BASE_SLICE],
                tf.zeros_like(node_features[:, NODE_SEMANTIC_SLICE]),
                node_features[:, NODE_VISUAL_DETAIL_SLICE],
            ),
            axis=1,
        )
        edge_features = batch["edge_features"]
        transformed["edge_features"] = tf.concat(
            (
                edge_features[:, EDGE_VISUAL_SLICE],
                tf.zeros_like(edge_features[:, EDGE_SEMANTIC_SLICE]),
            ),
            axis=1,
        )
    return transformed


def _assert_equal(actual: tf.Tensor, expected: tf.Tensor, message: str) -> None:
    try:
        tf.debugging.assert_equal(actual, expected, message=message)
    except (tf.errors.InvalidArgumentError, ValueError) as exc:
        raise PriorProbeError(message) from exc


def validate_intervention_integrity(
    source: Mapping[str, tf.Tensor],
    transformed: Mapping[str, tf.Tensor],
    condition: str,
) -> dict:
    """Prove exact field/slice invariants for a transformed batch."""
    source_shape = validate_batch_schema(source)
    transformed_shape = validate_batch_schema(transformed)
    if source_shape != transformed_shape:
        raise PriorProbeError(f"Condition {condition} changed batch dimensions")

    changed_fields = set(INTERVENTION_SPECS[condition]["changed_tensor_fields"])
    for name in source:
        if source[name].shape != transformed[name].shape:
            raise PriorProbeError(f"Condition {condition} changed shape for {name}")
        if source[name].dtype != transformed[name].dtype:
            raise PriorProbeError(f"Condition {condition} changed dtype for {name}")
        if name not in changed_fields:
            _assert_equal(
                transformed[name], source[name],
                f"Condition {condition} changed forbidden field {name}",
            )

    if condition == CONDITION_OFFICIAL:
        for name in source:
            _assert_equal(
                transformed[name], source[name],
                f"Official condition is not identity for {name}",
            )
    else:
        _assert_equal(
            transformed["part_soft"], tf.zeros_like(source["part_soft"]),
            f"Condition {condition} did not zero part_soft",
        )
        _assert_equal(
            transformed["valid_part_mask"],
            tf.zeros_like(source["valid_part_mask"]),
            f"Condition {condition} did not zero valid_part_mask",
        )
    if condition == CONDITION_SEMANTIC_ZERO:
        _assert_equal(
            transformed["node_features"][:, NODE_SEMANTIC_SLICE],
            tf.zeros_like(source["node_features"][:, NODE_SEMANTIC_SLICE]),
            "C2 did not zero node columns 5..31",
        )
        _assert_equal(
            transformed["node_features"][:, NODE_VISUAL_BASE_SLICE],
            source["node_features"][:, NODE_VISUAL_BASE_SLICE],
            "C2 changed node columns 0..4",
        )
        _assert_equal(
            transformed["node_features"][:, NODE_VISUAL_DETAIL_SLICE],
            source["node_features"][:, NODE_VISUAL_DETAIL_SLICE],
            "C2 changed node columns 32..36",
        )
        _assert_equal(
            transformed["edge_features"][:, EDGE_SEMANTIC_SLICE],
            tf.zeros_like(source["edge_features"][:, EDGE_SEMANTIC_SLICE]),
            "C2 did not zero edge columns 6..7",
        )
        _assert_equal(
            transformed["edge_features"][:, EDGE_VISUAL_SLICE],
            source["edge_features"][:, EDGE_VISUAL_SLICE],
            "C2 changed edge columns 0..5",
        )
    return {
        "condition": condition,
        "changed_tensor_fields": sorted(changed_fields),
        "dimensions": source_shape,
        "shapes_and_dtypes_preserved": True,
        "topology_and_sample_identity_preserved": True,
    }


def model_weights_sha256(model) -> str:
    digest = hashlib.sha256()
    for variable in model.weights:
        value = np.asarray(variable.numpy())
        digest.update(str(variable.name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def load_fixed_checkpoint(checkpoint: str | Path):
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.suffix.lower() != ".keras":
        raise PriorProbeError("Issue #9 requires a .keras checkpoint")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    model = tf.keras.models.load_model(
        checkpoint_path,
        custom_objects={"LapGNN": LapGNN, "lap_gnn_tf>LapGNN": LapGNN},
        compile=False,
    )
    if getattr(model, "optimizer", None) is not None:
        raise PriorProbeError("compile=False checkpoint load unexpectedly restored an optimizer")
    if int(model.count_params()) != EXPECTED_PARAMETER_COUNT:
        raise PriorProbeError(
            f"Loaded model parameter count drift: {model.count_params()}"
        )
    return model


def evaluate_conditions(model, batches: Iterable[Mapping[str, tf.Tensor]]) -> dict:
    """Evaluate C0/C1/C2 as a paired set for every original validation batch."""
    evaluate_step = build_compiled_evaluation_step(model)
    labels_parts: list[np.ndarray] = []
    sample_id_parts: list[np.ndarray] = []
    probability_parts = {condition: [] for condition in CONDITIONS}
    losses = {condition: [] for condition in CONDITIONS}
    integrity_counts = {condition: 0 for condition in CONDITIONS}
    batch_count = 0

    for source_batch in batches:
        validate_batch_schema(source_batch)
        labels = np.asarray(source_batch["labels"].numpy(), dtype=np.int64)
        sample_ids = np.asarray(source_batch["sample_ids"].numpy(), dtype=np.int64)
        labels_parts.append(labels)
        sample_id_parts.append(sample_ids)
        for condition in CONDITIONS:
            transformed = apply_intervention(source_batch, condition)
            validate_intervention_integrity(source_batch, transformed, condition)
            loss, probabilities = evaluate_step(transformed)
            probabilities_array = np.asarray(probabilities.numpy(), dtype=np.float64)
            if probabilities_array.shape != (labels.size, 7):
                raise PriorProbeError(
                    f"Condition {condition} probability shape drift: {probabilities_array.shape}"
                )
            losses[condition].append(float(loss.numpy()))
            probability_parts[condition].append(probabilities_array)
            integrity_counts[condition] += 1
        batch_count += 1

    if not batch_count:
        raise PriorProbeError("No validation batches were produced")
    labels = np.concatenate(labels_parts)
    sample_ids = np.concatenate(sample_id_parts)
    if len(np.unique(sample_ids)) != len(sample_ids):
        raise PriorProbeError("Validation sample IDs are not unique")

    probabilities_by_condition = {
        condition: np.concatenate(parts, axis=0)
        for condition, parts in probability_parts.items()
    }
    metrics = {}
    for condition in CONDITIONS:
        condition_metrics = classification_metrics(
            labels, probabilities_by_condition[condition]
        )
        condition_metrics["loss"] = float(np.mean(losses[condition]))
        metrics[condition] = condition_metrics
    return {
        "batch_count": batch_count,
        "sample_count": int(labels.size),
        "labels": labels,
        "sample_ids": sample_ids,
        "probabilities": probabilities_by_condition,
        "metrics": metrics,
        "integrity_counts": integrity_counts,
    }


def _paired_predictions_csv(result: Mapping) -> str:
    fieldnames = ["sample_id", "label"]
    for condition in CONDITIONS:
        fieldnames.append(f"{condition}_prediction")
        fieldnames.extend(f"{condition}_probability_{index}" for index in range(7))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    labels = result["labels"]
    sample_ids = result["sample_ids"]
    probabilities = result["probabilities"]
    for row_index in range(len(labels)):
        row = {
            "sample_id": int(sample_ids[row_index]),
            "label": int(labels[row_index]),
        }
        for condition in CONDITIONS:
            row_probabilities = probabilities[condition][row_index]
            row[f"{condition}_prediction"] = int(np.argmax(row_probabilities))
            for class_index, value in enumerate(row_probabilities):
                row[f"{condition}_probability_{class_index}"] = format(
                    float(value), ".17g"
                )
        writer.writerow(row)
    return stream.getvalue()


def write_probe_outputs(
    output_root: Path,
    *,
    result: Mapping,
    checkpoint_path: Path,
    checkpoint_metadata_path: Path,
    checkpoint_sha256_before: str,
    checkpoint_sha256_after: str,
    model_weights_sha256_before: str,
    model_weights_sha256_after: str,
    resolved_config_path: Path,
    resolved_config_sha256: str,
    contract: Mapping,
    checkpoint_cross_check: Mapping,
    resources: Mapping,
    bounded_limit: int | None,
) -> dict:
    if output_root.exists():
        raise FileExistsError(f"Fresh probe output must not already exist: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    condition_paths = {}
    for condition in CONDITIONS:
        path = output_root / f"validation_metrics_{condition}.json"
        _write_json(
            path,
            {
                "schema_version": 1,
                "issue": ISSUE_NUMBER,
                "condition": condition,
                "split": "val",
                "topology_fixed": True,
                "checkpoint_sha256": checkpoint_sha256_before,
                "metrics": result["metrics"][condition],
            },
        )
        condition_paths[condition] = path

    predictions_path = output_root / "paired_validation_predictions.csv"
    _write_text_atomic(predictions_path, _paired_predictions_csv(result))

    integrity_path = output_root / "intervention_integrity.json"
    integrity_payload = {
        "schema_version": 1,
        "issue": ISSUE_NUMBER,
        "split": "val",
        "batch_count": result["batch_count"],
        "sample_count": result["sample_count"],
        "paired_original_batch_evaluation": True,
        "source_batches_mutated": False,
        "conditions": INTERVENTION_SPECS,
        "runtime_invariance_checks_per_condition": result["integrity_counts"],
        "preserved_identity": [
            "edge_index",
            "node_graph_index",
            "edge_graph_index",
            "graph_node_counts",
            "graph_edge_counts",
            "node_types",
            "coordinates",
            "anchor_mask",
            "labels",
            "sample_ids",
            "image_48",
        ],
        "checkpoint_unchanged": checkpoint_sha256_before == checkpoint_sha256_after,
        "model_weights_unchanged": model_weights_sha256_before == model_weights_sha256_after,
        "test_split_constructed": False,
        "training_or_optimizer_step_created": False,
    }
    _write_json(integrity_path, integrity_payload)

    artifacts = [*condition_paths.values(), predictions_path, integrity_path]
    artifact_hashes = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in artifacts
    }
    manifest = {
        "schema_version": 1,
        "tool_version": TOOL_VERSION,
        "issue": ISSUE_NUMBER,
        "split": "val",
        "validation_only": True,
        "topology_fixed": True,
        "condition_order": list(CONDITIONS),
        "conditions": INTERVENTION_SPECS,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "metadata_path": str(checkpoint_metadata_path.resolve()),
            "sha256_before": checkpoint_sha256_before,
            "sha256_after": checkpoint_sha256_after,
            "unchanged": checkpoint_sha256_before == checkpoint_sha256_after,
            "model_weights_sha256_before": model_weights_sha256_before,
            "model_weights_sha256_after": model_weights_sha256_after,
            "model_weights_unchanged": model_weights_sha256_before == model_weights_sha256_after,
            **dict(checkpoint_cross_check),
        },
        "resolved_config": {
            "path": str(resolved_config_path.resolve()),
            "sha256": resolved_config_sha256,
        },
        "frozen_contract": dict(contract),
        "resources": dict(resources),
        "limit_val_batches": bounded_limit,
        "bounded_smoke_only": bounded_limit is not None,
        "batch_count": result["batch_count"],
        "sample_count": result["sample_count"],
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
        "artifacts": artifact_hashes,
    }
    manifest_path = output_root / "probe_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate registered C0/C1/C2 post-graph interventions on validation "
            "only, using one immutable .keras checkpoint."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-metadata")
    parser.add_argument("--resolved-config", required=True)
    parser.add_argument("--prior-root", required=True)
    parser.add_argument("--clean-graph-cache-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--graph-workers", type=int)
    parser.add_argument("--graph-cache-size", type=int)
    parser.add_argument(
        "--limit-val-batches",
        type=int,
        help="Bounded implementation smoke only; forbidden for the registered Step 6 run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint_path = Path(args.checkpoint)
    metadata_path = (
        Path(args.checkpoint_metadata)
        if args.checkpoint_metadata
        else infer_checkpoint_metadata_path(checkpoint_path)
    )
    resolved_config_path = Path(args.resolved_config)
    prior_root = Path(args.prior_root)
    clean_graph_cache_dir = Path(args.clean_graph_cache_dir)
    output_root = Path(args.output_root)

    if output_root.exists():
        raise FileExistsError(f"Fresh probe output must not already exist: {output_root}")
    for required in (checkpoint_path, metadata_path, resolved_config_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    for required in (prior_root, clean_graph_cache_dir):
        if not required.is_dir():
            raise FileNotFoundError(required)

    eval_batch_size = _require_positive("eval_batch_size", args.eval_batch_size)
    graph_workers = _require_positive("graph_workers", args.graph_workers)
    graph_cache_size = _require_positive("graph_cache_size", args.graph_cache_size)
    limit_val_batches = _require_positive(
        "limit_val_batches", args.limit_val_batches
    )

    raw_resolved_config = load_persisted_resolved_config(resolved_config_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise PriorProbeError("Checkpoint metadata must be a JSON object")
    checkpoint_cross_check = validate_checkpoint_metadata(
        raw_resolved_config, metadata
    )

    config = build_runtime_config(raw_resolved_config)
    contract = validate_frozen_contract(config)

    resources_config = dict(config.get("resources") or {})
    eval_batch_size = eval_batch_size or int(
        resources_config.get("eval_batch_size", config["training"]["batch_size"])
    )
    graph_workers = graph_workers or int(resources_config.get("graph_workers", 1))
    graph_cache_size = graph_cache_size or int(
        resources_config.get("graph_cache_size", 64)
    )
    resources = {
        "eval_batch_size": eval_batch_size,
        "graph_workers": graph_workers,
        "graph_cache_size": graph_cache_size,
        "shuffle": False,
        "dataset_split": "val",
    }
    resources.update(
        configure_gpu_memory_growth(
            bool(resources_config.get("memory_growth", True))
        )
    )

    checkpoint_sha256_before = sha256_file(checkpoint_path)
    model = load_fixed_checkpoint(checkpoint_path)
    model_weights_sha256_before = model_weights_sha256(model)
    validation_data = GraphBatchGenerator(
        prior_root=prior_root,
        split="val",
        config=config,
        batch_size=eval_batch_size,
        seed=int(config["seed"]),
        shuffle=False,
        graph_cache_size=graph_cache_size,
        graph_workers=graph_workers,
        clean_graph_cache_dir=clean_graph_cache_dir,
    )
    result = evaluate_conditions(
        model,
        validation_data.iter_epoch(0, limit_batches=limit_val_batches),
    )
    model_weights_sha256_after = model_weights_sha256(model)
    checkpoint_sha256_after = sha256_file(checkpoint_path)
    if checkpoint_sha256_after != checkpoint_sha256_before:
        raise PriorProbeError("Checkpoint file changed during validation-only evaluation")
    if model_weights_sha256_after != model_weights_sha256_before:
        raise PriorProbeError("Loaded model weights changed during inference")

    manifest = write_probe_outputs(
        output_root,
        result=result,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata_path=metadata_path,
        checkpoint_sha256_before=checkpoint_sha256_before,
        checkpoint_sha256_after=checkpoint_sha256_after,
        model_weights_sha256_before=model_weights_sha256_before,
        model_weights_sha256_after=model_weights_sha256_after,
        resolved_config_path=resolved_config_path,
        resolved_config_sha256=sha256_file(resolved_config_path),
        contract=contract,
        checkpoint_cross_check=checkpoint_cross_check,
        resources=resources,
        bounded_limit=limit_val_batches,
    )
    print(json.dumps(_json_ready(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
