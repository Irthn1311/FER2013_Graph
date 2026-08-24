"""Issue #13 validation-only fixed-checkpoint direct-part decomposition harness.

The harness manually reproduces the frozen LapGNN inference call graph so the
four registered direct-part interfaces can be varied independently.  It never
constructs a train or test split and never mutates the source graph batch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import tensorflow as tf

from lap_gnn_tf.constants import EXPECTED_PARAMETER_COUNT
from lap_gnn_tf.data.graph_generator import GraphBatchGenerator
from lap_gnn_tf.model import LapGNN
from lap_gnn_tf.model.motif_layers import PART_ORDER, part_pool
from lap_gnn_tf.resources import environment_manifest
from lap_gnn_tf.signatures import sha256_file
from lap_gnn_tf.training.losses import sparse_cross_entropy
from lap_gnn_tf.training.metrics import classification_metrics


TOOL_VERSION = "1.0.1"
ISSUE_NUMBER = 13
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SUPPORT_TOOL_PATH = Path(__file__).with_name(
    "evaluate_fixed_checkpoint_prior_probe.py"
)


def _load_step6_support():
    spec = importlib.util.spec_from_file_location(
        "_issue13_step6_probe_support", SUPPORT_TOOL_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Step 6 support: {SUPPORT_TOOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


step6 = _load_step6_support()

EXPECTED_BASE_COMMIT = "8675c839004c18322da28c95770ee6e126e0e22f"
EXPECTED_SCIENTIFIC_PAYLOAD_SHA256 = (
    "286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e"
)
EXPECTED_EXECUTION_CONTRACT_SHA256 = step6.EXPECTED_EXECUTION_CONTRACT_SHA256

CONDITION_D0 = "official_manual_forward"
CONDITION_D1 = "context_local_prior_neutralized"
CONDITION_D2 = "readout_local_prior_neutralized"
CONDITION_D3 = "local_part_residual_zero"
CONDITION_D4 = "local_motif_validity_off"
CONDITION_D5 = "full_direct_part_zero_anchor"
CONDITIONS = (
    CONDITION_D0,
    CONDITION_D1,
    CONDITION_D2,
    CONDITION_D3,
    CONDITION_D4,
    CONDITION_D5,
)
LOCAL_PARTS = tuple(PART_ORDER[:4])

INTERVENTION_SPECS = {
    CONDITION_D0: {
        "changed_pathway_arguments": [],
        "description": "Manual identity forward; every pathway is official.",
    },
    CONDITION_D1: {
        "changed_pathway_arguments": ["context.part_soft"],
        "description": (
            "Only PartGlobalContext receives zero part_soft; its global token/path "
            "and every other pathway remain official."
        ),
    },
    CONDITION_D2: {
        "changed_pathway_arguments": ["readout.part_soft"],
        "description": (
            "Only readout part_soft is zero, neutralizing local spatial log-prior "
            "bias without deleting attention or changing global-prior queries."
        ),
    },
    CONDITION_D3: {
        "changed_pathway_arguments": [
            *(f"readout.part_embeddings.{name}" for name in LOCAL_PARTS)
        ],
        "description": (
            "Only four local pooled residual embeddings are zero; global pooled "
            "embedding, validity flags, and readout priors remain official."
        ),
    },
    CONDITION_D4: {
        "changed_pathway_arguments": [
            *(f"readout.valid_groups.{name}" for name in LOCAL_PARTS)
        ],
        "description": (
            "Only four local motif-validity flags are false; all pooled embeddings "
            "and readout priors remain official, with global validity true."
        ),
    },
    CONDITION_D5: {
        "changed_pathway_arguments": [
            "context.part_soft",
            "part_pool.part_soft",
            "part_pool.valid_part_mask",
            "readout.part_soft",
        ],
        "description": (
            "Exact Step 6 C1 anchor: zero direct part_soft at context, pooling, and "
            "readout plus zero valid_part_mask at pooling; graph tensors unchanged."
        ),
    },
}

NATIVE_MANUAL_TOLERANCE = {
    "prediction_agreement": 1.0,
    "max_abs_logit_difference": 1e-5,
    "max_abs_probability_difference": 1e-6,
}
D0_REFERENCE = {
    "accuracy": 0.63137364168292,
    "macro_f1": 0.5932591901893336,
    "loss": 1.1537981724317095,
}
D5_REFERENCE = {
    "accuracy": 0.27751462803009197,
    "macro_f1": 0.19745892656222366,
    "loss": 1.757720434560185,
}
REFERENCE_TOLERANCE = {"accuracy": 0.001, "macro_f1": 0.001, "loss": 0.005}
EXPECTED_FULL_VALIDATION_SAMPLES = 3589


class DirectPartProbeError(RuntimeError):
    """Fail-closed Issue #13 harness error."""


class ManualForwardEquivalenceError(DirectPartProbeError):
    """Native/manual D0 identity failure with preservation evidence."""

    def __init__(self, evidence: Mapping):
        self.evidence = dict(evidence)
        super().__init__("INVALID_MANUAL_FORWARD_EQUIVALENCE")


def _require_condition(condition: str) -> None:
    if condition not in CONDITIONS:
        raise DirectPartProbeError(
            f"Unknown direct-part decomposition condition: {condition!r}"
        )


def _snapshot_batch(batch: Mapping[str, tf.Tensor]) -> dict[str, np.ndarray]:
    step6.validate_batch_schema(batch)
    return {
        name: np.array(tensor.numpy(), copy=True) for name, tensor in batch.items()
    }


def _assert_source_unchanged(
    batch: Mapping[str, tf.Tensor], snapshot: Mapping[str, np.ndarray]
) -> None:
    if set(batch) != set(snapshot):
        raise DirectPartProbeError("Source batch field set changed")
    for name, tensor in batch.items():
        actual = np.asarray(tensor.numpy())
        if not np.array_equal(actual, snapshot[name]):
            raise DirectPartProbeError(
                f"Source batch tensor changed during decomposition: {name}"
            )


def _assert_equal(actual, expected, message: str) -> None:
    try:
        tf.debugging.assert_equal(actual, expected, message=message)
    except (tf.errors.InvalidArgumentError, ValueError) as exc:
        raise DirectPartProbeError(message) from exc


def _copy_tensor_dict(values: Mapping[str, tf.Tensor]) -> dict[str, tf.Tensor]:
    return {name: value for name, value in values.items()}


def _normalize_model_boundary_inputs(
    model, batch: Mapping[str, tf.Tensor]
) -> dict[str, tf.Tensor]:
    """Mirror Keras Model.__call__ autocast before manually entering call()."""
    input_dtype = tf.dtypes.as_dtype(model.input_dtype)
    autocast = bool(model.autocast)
    return {
        name: (
            tf.cast(value, input_dtype)
            if autocast and value.dtype.is_floating and value.dtype != input_dtype
            else value
        )
        for name, value in batch.items()
    }


def manual_forward(
    model,
    batch: Mapping[str, tf.Tensor],
    condition: str,
) -> tuple[dict[str, tf.Tensor], dict]:
    """Run one registered manual forward without changing model or source batch."""
    _require_condition(condition)
    step6.validate_batch_schema(batch)
    boundary_batch = _normalize_model_boundary_inputs(model, batch)

    node_features = tf.cast(boundary_batch["node_features"], tf.float32)
    edge_features = tf.cast(boundary_batch["edge_features"], tf.float32)
    edge_index = tf.cast(boundary_batch["edge_index"], tf.int64)
    node_graph_index = tf.cast(boundary_batch["node_graph_index"], tf.int32)
    official_part_soft = tf.cast(boundary_batch["part_soft"], tf.float32)
    official_valid_part_mask = tf.cast(
        boundary_batch["valid_part_mask"], tf.float32
    )
    num_graphs = tf.shape(boundary_batch["labels"])[0]

    h = model.encoder(node_features, training=False)
    for layer in model.gnn.layers_:
        h = layer(h, edge_index, edge_features, training=False)
    pre_context_h = h

    context_part_soft = (
        tf.zeros_like(official_part_soft)
        if condition in (CONDITION_D1, CONDITION_D5)
        else official_part_soft
    )
    h = model.gnn.context(
        h,
        part_soft=context_part_soft,
        node_graph_index=node_graph_index,
        num_graphs=num_graphs,
        training=False,
    )

    pool_part_soft = (
        tf.zeros_like(official_part_soft)
        if condition == CONDITION_D5
        else official_part_soft
    )
    pool_valid_part_mask = (
        tf.zeros_like(official_valid_part_mask)
        if condition == CONDITION_D5
        else official_valid_part_mask
    )
    pooled_official, valid_official = part_pool(
        h,
        pool_part_soft,
        node_graph_index,
        pool_valid_part_mask,
        num_graphs,
    )

    readout_part_soft = (
        tf.zeros_like(official_part_soft)
        if condition in (CONDITION_D2, CONDITION_D5)
        else official_part_soft
    )
    readout_embeddings = _copy_tensor_dict(pooled_official)
    readout_valid = _copy_tensor_dict(valid_official)
    if condition == CONDITION_D3:
        for name in LOCAL_PARTS:
            readout_embeddings[name] = tf.zeros_like(pooled_official[name])
    if condition == CONDITION_D4:
        for name in LOCAL_PARTS:
            readout_valid[name] = tf.zeros_like(valid_official[name], dtype=tf.bool)
        readout_valid["global"] = tf.ones_like(
            valid_official["global"], dtype=tf.bool
        )

    readout = model.readout(
        h,
        node_features,
        readout_part_soft,
        node_graph_index,
        num_graphs,
        readout_embeddings,
        readout_valid,
        training=False,
    )
    logits = model.classifier(readout["z_image"], training=False)
    probabilities = tf.nn.softmax(logits, axis=-1)
    output = {
        "logits": logits,
        "probabilities": probabilities,
        "predictions": tf.argmax(logits, axis=1, output_type=tf.int64),
        "z_image": readout["z_image"],
        "node_embeddings": h,
        "part_embeddings": readout_embeddings,
    }
    trace = {
        "condition": condition,
        "model_boundary": {
            "autocast": bool(model.autocast),
            "input_dtype": tf.dtypes.as_dtype(model.input_dtype).name,
            "source_dtypes": {
                name: value.dtype.name for name, value in batch.items()
            },
            "effective_dtypes": {
                name: value.dtype.name for name, value in boundary_batch.items()
            },
        },
        "message_passing": {
            "node_features": node_features,
            "edge_features": edge_features,
            "edge_index": edge_index,
            "node_graph_index": node_graph_index,
            "pre_context_h": pre_context_h,
        },
        "official_part_soft": official_part_soft,
        "official_valid_part_mask": official_valid_part_mask,
        "context_part_soft": context_part_soft,
        "pool_part_soft": pool_part_soft,
        "pool_valid_part_mask": pool_valid_part_mask,
        "pooled_before_readout_intervention": pooled_official,
        "valid_before_readout_intervention": valid_official,
        "readout_part_soft": readout_part_soft,
        "readout_part_embeddings": readout_embeddings,
        "readout_valid_groups": readout_valid,
    }
    return output, trace


def validate_pathway_integrity(
    model,
    batch: Mapping[str, tf.Tensor],
    snapshot: Mapping[str, np.ndarray],
    condition: str,
    trace: Mapping,
) -> dict:
    """Prove that only registered intermediate pathway arguments changed."""
    _require_condition(condition)
    _assert_source_unchanged(batch, snapshot)
    boundary_batch = _normalize_model_boundary_inputs(model, batch)
    official_part_soft = tf.cast(boundary_batch["part_soft"], tf.float32)
    official_valid_part_mask = tf.cast(
        boundary_batch["valid_part_mask"], tf.float32
    )
    zero_part_soft = tf.zeros_like(official_part_soft)
    zero_valid_part_mask = tf.zeros_like(official_valid_part_mask)

    expected_context = (
        zero_part_soft
        if condition in (CONDITION_D1, CONDITION_D5)
        else official_part_soft
    )
    expected_pool_part = zero_part_soft if condition == CONDITION_D5 else official_part_soft
    expected_pool_valid = (
        zero_valid_part_mask if condition == CONDITION_D5 else official_valid_part_mask
    )
    expected_readout = (
        zero_part_soft
        if condition in (CONDITION_D2, CONDITION_D5)
        else official_part_soft
    )
    _assert_equal(trace["context_part_soft"], expected_context, "Context prior drift")
    _assert_equal(trace["pool_part_soft"], expected_pool_part, "Pool prior drift")
    _assert_equal(
        trace["pool_valid_part_mask"], expected_pool_valid, "Pool validity drift"
    )
    _assert_equal(trace["readout_part_soft"], expected_readout, "Readout prior drift")

    message = trace["message_passing"]
    for name, expected in (
        ("node_features", tf.cast(boundary_batch["node_features"], tf.float32)),
        ("edge_features", tf.cast(boundary_batch["edge_features"], tf.float32)),
        ("edge_index", tf.cast(boundary_batch["edge_index"], tf.int64)),
        (
            "node_graph_index",
            tf.cast(boundary_batch["node_graph_index"], tf.int32),
        ),
    ):
        _assert_equal(message[name], expected, f"Message-passing input drift: {name}")
    expected_boundary = {
        "autocast": bool(model.autocast),
        "input_dtype": tf.dtypes.as_dtype(model.input_dtype).name,
        "source_dtypes": {name: value.dtype.name for name, value in batch.items()},
        "effective_dtypes": {
            name: value.dtype.name for name, value in boundary_batch.items()
        },
    }
    if trace.get("model_boundary") != expected_boundary:
        raise DirectPartProbeError("Keras model-boundary input semantics drift")

    pooled = trace["pooled_before_readout_intervention"]
    readout_pooled = trace["readout_part_embeddings"]
    valid = trace["valid_before_readout_intervention"]
    readout_valid = trace["readout_valid_groups"]
    for name in PART_ORDER:
        expected_embedding = (
            tf.zeros_like(pooled[name])
            if condition == CONDITION_D3 and name in LOCAL_PARTS
            else pooled[name]
        )
        _assert_equal(
            readout_pooled[name], expected_embedding,
            f"Readout pooled embedding drift: {condition}/{name}",
        )
        expected_valid = (
            tf.zeros_like(valid[name], dtype=tf.bool)
            if condition == CONDITION_D4 and name in LOCAL_PARTS
            else (
                tf.ones_like(valid[name], dtype=tf.bool)
                if condition == CONDITION_D4 and name == "global"
                else valid[name]
            )
        )
        _assert_equal(
            readout_valid[name], expected_valid,
            f"Readout validity drift: {condition}/{name}",
        )

    if condition == CONDITION_D5:
        for name in LOCAL_PARTS:
            _assert_equal(
                pooled[name], tf.zeros_like(pooled[name]),
                f"D5 local pooled embedding is not zero: {name}",
            )
            _assert_equal(
                valid[name], tf.zeros_like(valid[name], dtype=tf.bool),
                f"D5 local pooled validity is not false: {name}",
            )

    return {
        "condition": condition,
        "changed_pathway_arguments": list(
            INTERVENTION_SPECS[condition]["changed_pathway_arguments"]
        ),
        "source_batch_unchanged": True,
        "message_passing_inputs_unchanged": True,
        "node_edge_topology_unchanged": True,
        "registered_pathway_arguments_exact": True,
        "model_boundary_input_semantics": expected_boundary,
    }


def native_manual_equivalence(
    native_output: Mapping[str, tf.Tensor],
    manual_output: Mapping[str, tf.Tensor],
    sample_ids: tf.Tensor,
) -> dict:
    native_logits = np.asarray(native_output["logits"].numpy(), dtype=np.float64)
    manual_logits = np.asarray(manual_output["logits"].numpy(), dtype=np.float64)
    native_probabilities = np.asarray(
        native_output["probabilities"].numpy(), dtype=np.float64
    )
    manual_probabilities = np.asarray(
        manual_output["probabilities"].numpy(), dtype=np.float64
    )
    if native_logits.shape != manual_logits.shape:
        raise DirectPartProbeError("Native/manual D0 logit shape drift")
    if native_probabilities.shape != manual_probabilities.shape:
        raise DirectPartProbeError("Native/manual D0 probability shape drift")
    native_predictions = native_probabilities.argmax(axis=1)
    manual_predictions = manual_probabilities.argmax(axis=1)
    agreement = float(np.mean(native_predictions == manual_predictions))
    evidence = {
        "sample_count": int(native_logits.shape[0]),
        "sample_ids_sha256": hashlib.sha256(
            np.asarray(sample_ids.numpy(), dtype=np.int64).tobytes(order="C")
        ).hexdigest(),
        "sample_order_equal": True,
        "prediction_agreement": agreement,
        "max_abs_logit_difference": float(
            np.max(np.abs(native_logits - manual_logits), initial=0.0)
        ),
        "max_abs_probability_difference": float(
            np.max(
                np.abs(native_probabilities - manual_probabilities), initial=0.0
            )
        ),
    }
    evidence["gate_pass"] = bool(
        evidence["prediction_agreement"] == 1.0
        and evidence["max_abs_logit_difference"]
        <= NATIVE_MANUAL_TOLERANCE["max_abs_logit_difference"]
        and evidence["max_abs_probability_difference"]
        <= NATIVE_MANUAL_TOLERANCE["max_abs_probability_difference"]
    )
    return evidence


def load_fixed_checkpoint(checkpoint: str | Path):
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.suffix.lower() != ".keras":
        raise DirectPartProbeError("Issue #13 requires a .keras checkpoint")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    model = tf.keras.models.load_model(
        checkpoint_path,
        custom_objects={"LapGNN": LapGNN, "lap_gnn_tf>LapGNN": LapGNN},
        compile=False,
    )
    if getattr(model, "optimizer", None) is not None:
        raise DirectPartProbeError(
            "compile=False checkpoint load unexpectedly restored an optimizer"
        )
    if int(model.count_params()) != EXPECTED_PARAMETER_COUNT:
        raise DirectPartProbeError(
            f"Loaded model parameter count drift: {model.count_params()}"
        )
    return model


def _paired_diagnostics(
    labels: np.ndarray, probabilities: Mapping[str, np.ndarray]
) -> dict:
    d0_predictions = probabilities[CONDITION_D0].argmax(axis=1)
    result = {}
    for condition in CONDITIONS[1:]:
        predictions = probabilities[condition].argmax(axis=1)
        d0_correct = d0_predictions == labels
        condition_correct = predictions == labels
        result[condition] = {
            "prediction_disagreement_count": int(
                np.count_nonzero(predictions != d0_predictions)
            ),
            "prediction_disagreement_rate": float(
                np.mean(predictions != d0_predictions)
            ),
            "correctness_transitions": {
                "d0_correct_to_intervention_incorrect": int(
                    np.count_nonzero(d0_correct & ~condition_correct)
                ),
                "d0_incorrect_to_intervention_correct": int(
                    np.count_nonzero(~d0_correct & condition_correct)
                ),
                "unchanged_correct": int(
                    np.count_nonzero(d0_correct & condition_correct)
                ),
                "unchanged_incorrect": int(
                    np.count_nonzero(~d0_correct & ~condition_correct)
                ),
            },
        }
    return result


def evaluate_conditions(
    model, batches: Iterable[Mapping[str, tf.Tensor]]
) -> dict:
    """Evaluate D0-D5 as one paired set before advancing each source batch."""
    labels_parts: list[np.ndarray] = []
    sample_id_parts: list[np.ndarray] = []
    probability_parts = {condition: [] for condition in CONDITIONS}
    losses = {condition: [] for condition in CONDITIONS}
    integrity_counts = {condition: 0 for condition in CONDITIONS}
    equivalence_batches = []
    model_boundary_input_semantics = None
    batch_count = 0

    for source_batch in batches:
        step6.validate_batch_schema(source_batch)
        snapshot = _snapshot_batch(source_batch)
        labels = np.asarray(source_batch["labels"].numpy(), dtype=np.int64)
        sample_ids = np.asarray(source_batch["sample_ids"].numpy(), dtype=np.int64)
        native_output = model(source_batch, training=False)
        d0_output, d0_trace = manual_forward(model, source_batch, CONDITION_D0)
        d0_integrity = validate_pathway_integrity(
            model, source_batch, snapshot, CONDITION_D0, d0_trace
        )
        current_boundary_semantics = d0_integrity[
            "model_boundary_input_semantics"
        ]
        if model_boundary_input_semantics is None:
            model_boundary_input_semantics = current_boundary_semantics
        elif current_boundary_semantics != model_boundary_input_semantics:
            raise DirectPartProbeError(
                "Model-boundary input semantics changed across validation batches"
            )
        equivalence = native_manual_equivalence(
            native_output, d0_output, source_batch["sample_ids"]
        )
        equivalence["batch_index"] = batch_count
        equivalence_batches.append(equivalence)
        if not equivalence["gate_pass"]:
            raise ManualForwardEquivalenceError(
                {
                    "status": "INVALID_MANUAL_FORWARD_EQUIVALENCE",
                    "tolerances": NATIVE_MANUAL_TOLERANCE,
                    "model_boundary_input_semantics": current_boundary_semantics,
                    "batches": equivalence_batches,
                }
            )

        labels_parts.append(labels)
        sample_id_parts.append(sample_ids)
        for condition in CONDITIONS:
            if condition == CONDITION_D0:
                output, trace = d0_output, d0_trace
            else:
                output, trace = manual_forward(model, source_batch, condition)
            condition_integrity = validate_pathway_integrity(
                model, source_batch, snapshot, condition, trace
            )
            if (
                condition_integrity["model_boundary_input_semantics"]
                != d0_integrity["model_boundary_input_semantics"]
            ):
                raise DirectPartProbeError(
                    "Model-boundary input semantics changed across conditions"
                )
            probabilities = np.asarray(
                output["probabilities"].numpy(), dtype=np.float64
            )
            if probabilities.shape != (labels.size, 7):
                raise DirectPartProbeError(
                    f"Condition {condition} probability shape drift: "
                    f"{probabilities.shape}"
                )
            loss = sparse_cross_entropy(source_batch["labels"], output["logits"])
            probability_parts[condition].append(probabilities)
            losses[condition].append(float(loss.numpy()))
            integrity_counts[condition] += 1
            _assert_source_unchanged(source_batch, snapshot)
        batch_count += 1

    if not batch_count:
        raise DirectPartProbeError("No validation batches were produced")
    labels = np.concatenate(labels_parts)
    sample_ids = np.concatenate(sample_id_parts)
    if len(np.unique(sample_ids)) != len(sample_ids):
        raise DirectPartProbeError("Validation sample IDs are not unique")
    probabilities = {
        condition: np.concatenate(parts, axis=0)
        for condition, parts in probability_parts.items()
    }
    metrics = {}
    for condition in CONDITIONS:
        condition_metrics = classification_metrics(labels, probabilities[condition])
        condition_metrics["loss"] = float(np.mean(losses[condition]))
        metrics[condition] = condition_metrics
    equivalence_summary = {
        "status": "PASS",
        "tolerances": NATIVE_MANUAL_TOLERANCE,
        "prediction_agreement": float(
            sum(
                item["prediction_agreement"] * item["sample_count"]
                for item in equivalence_batches
            )
            / labels.size
        ),
        "max_abs_logit_difference": max(
            item["max_abs_logit_difference"] for item in equivalence_batches
        ),
        "max_abs_probability_difference": max(
            item["max_abs_probability_difference"] for item in equivalence_batches
        ),
        "batches": equivalence_batches,
    }
    return {
        "batch_count": batch_count,
        "sample_count": int(labels.size),
        "labels": labels,
        "sample_ids": sample_ids,
        "probabilities": probabilities,
        "metrics": metrics,
        "paired_diagnostics": _paired_diagnostics(labels, probabilities),
        "integrity_counts": integrity_counts,
        "native_manual_equivalence": equivalence_summary,
        "model_boundary_input_semantics": model_boundary_input_semantics,
    }


def _reference_checks(observed: Mapping, reference: Mapping) -> dict:
    return {
        name: {
            "observed": float(observed[name]),
            "reference": float(reference[name]),
            "absolute_difference": abs(float(observed[name]) - float(reference[name])),
            "tolerance": REFERENCE_TOLERANCE[name],
            "pass": abs(float(observed[name]) - float(reference[name]))
            <= REFERENCE_TOLERANCE[name],
        }
        for name in ("accuracy", "macro_f1", "loss")
    }


def evaluate_registered_gates(result: Mapping, bounded_limit: int | None) -> dict:
    gate_a = dict(result["native_manual_equivalence"])
    gate_a["pass"] = gate_a.get("status") == "PASS"
    if bounded_limit is not None:
        return {
            "status": "BOUNDED_SMOKE_NO_SCIENTIFIC_INTERPRETATION",
            "gate_a_native_manual_equivalence": gate_a,
            "gate_b_d0_reference": {"status": "NOT_EVALUATED_BOUNDED_SMOKE"},
            "gate_c_d5_anchor": {"status": "NOT_EVALUATED_BOUNDED_SMOKE"},
            "per_path_diagnostics": None,
            "overall_decision": None,
        }

    d0_checks = _reference_checks(result["metrics"][CONDITION_D0], D0_REFERENCE)
    d0_sample_pass = result["sample_count"] == EXPECTED_FULL_VALIDATION_SAMPLES
    gate_b_pass = d0_sample_pass and all(item["pass"] for item in d0_checks.values())
    d5_checks = _reference_checks(result["metrics"][CONDITION_D5], D5_REFERENCE)
    gate_c_pass = all(item["pass"] for item in d5_checks.values())
    gate_b = {
        "status": "PASS" if gate_b_pass else "FAIL",
        "sample_count": result["sample_count"],
        "required_sample_count": EXPECTED_FULL_VALIDATION_SAMPLES,
        "sample_count_exact": d0_sample_pass,
        "checks": d0_checks,
    }
    gate_c = {
        "status": "PASS" if gate_c_pass else "FAIL",
        "checks": d5_checks,
    }
    if not gate_b_pass:
        status = "INVALID_D0_REFERENCE_REPRODUCTION"
    elif not gate_c_pass:
        status = "INVALID_C1_ANCHOR_REPRODUCTION"
    else:
        status = "VALID_REGISTERED_DECOMPOSITION"
    if status != "VALID_REGISTERED_DECOMPOSITION":
        return {
            "status": status,
            "gate_a_native_manual_equivalence": gate_a,
            "gate_b_d0_reference": gate_b,
            "gate_c_d5_anchor": gate_c,
            "per_path_diagnostics": None,
            "overall_decision": None,
        }

    d0_f1 = float(result["metrics"][CONDITION_D0]["macro_f1"])
    per_path = {}
    high_count = 0
    for condition in CONDITIONS[1:5]:
        delta = 100.0 * (
            d0_f1 - float(result["metrics"][condition]["macro_f1"])
        )
        if delta >= 10.0:
            label = "HIGH_PATH_SENSITIVITY"
            high_count += 1
        elif delta >= 5.0:
            label = "MODERATE_PATH_SENSITIVITY"
        else:
            label = "LOW_PATH_SENSITIVITY"
        per_path[condition] = {
            "delta_f1_pp": delta,
            "label": label,
            "negative_effect_note": (
                "Intervention improved macro-F1; negative delta retained exactly."
                if delta < 0.0
                else None
            ),
        }
    if high_count == 1:
        decision = "SINGLE_HIGH_DIRECT_PATH"
    elif high_count >= 2:
        decision = "MULTIPLE_HIGH_DIRECT_PATHS"
    else:
        decision = "INTERACTION_DOMINATED_DIRECT_DEPENDENCY"
    return {
        "status": status,
        "gate_a_native_manual_equivalence": gate_a,
        "gate_b_d0_reference": gate_b,
        "gate_c_d5_anchor": gate_c,
        "per_path_diagnostics": per_path,
        "overall_decision": decision,
        "non_additivity_warning": (
            "D1-D4 effects are nonlinear diagnostic sensitivities and must not be "
            "summed or treated as additive causal contributions."
        ),
    }


def _paired_predictions_csv(result: Mapping) -> str:
    fieldnames = ["sample_id", "label"]
    for condition in CONDITIONS:
        fieldnames.append(f"{condition}_prediction")
        fieldnames.extend(
            f"{condition}_probability_{index}" for index in range(7)
        )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row_index, (sample_id, label) in enumerate(
        zip(result["sample_ids"], result["labels"])
    ):
        row = {"sample_id": int(sample_id), "label": int(label)}
        for condition in CONDITIONS:
            probabilities = result["probabilities"][condition][row_index]
            row[f"{condition}_prediction"] = int(np.argmax(probabilities))
            for class_index, value in enumerate(probabilities):
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
    gates = evaluate_registered_gates(result, bounded_limit)

    artifacts = []
    for condition in CONDITIONS:
        path = output_root / f"validation_metrics_{condition}.json"
        step6._write_json(
            path,
            {
                "schema_version": 1,
                "issue": ISSUE_NUMBER,
                "condition": condition,
                "split": "val",
                "metrics": result["metrics"][condition],
            },
        )
        artifacts.append(path)
    predictions_path = output_root / "paired_validation_predictions.csv"
    step6._write_text_atomic(predictions_path, _paired_predictions_csv(result))
    artifacts.append(predictions_path)

    equivalence_path = output_root / "native_manual_d0_equivalence.json"
    step6._write_json(equivalence_path, result["native_manual_equivalence"])
    artifacts.append(equivalence_path)
    integrity_path = output_root / "intervention_integrity.json"
    integrity = {
        "schema_version": 1,
        "issue": ISSUE_NUMBER,
        "split": "val",
        "condition_order": list(CONDITIONS),
        "conditions": INTERVENTION_SPECS,
        "checks_per_condition": result["integrity_counts"],
        "source_batches_mutated": False,
        "message_passing_inputs_changed": False,
        "node_edge_features_changed": False,
        "topology_changed": False,
        "paired_original_batch_evaluation": True,
        "model_boundary_input_semantics": result[
            "model_boundary_input_semantics"
        ],
        "checkpoint_unchanged": checkpoint_sha256_before == checkpoint_sha256_after,
        "model_weights_unchanged": model_weights_sha256_before
        == model_weights_sha256_after,
        "training_or_optimizer_state_created": False,
        "test_split_constructed": False,
    }
    step6._write_json(integrity_path, integrity)
    artifacts.append(integrity_path)

    artifact_hashes = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in artifacts
    }
    manifest = {
        "schema_version": 1,
        "tool_version": TOOL_VERSION,
        "issue": ISSUE_NUMBER,
        "base_commit": EXPECTED_BASE_COMMIT,
        "split": "val",
        "validation_only": True,
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
            "model_weights_unchanged": model_weights_sha256_before
            == model_weights_sha256_after,
            **dict(checkpoint_cross_check),
        },
        "resolved_config": {
            "path": str(resolved_config_path.resolve()),
            "sha256": resolved_config_sha256,
        },
        "frozen_contract": dict(contract),
        "scientific_payload_sha256": EXPECTED_SCIENTIFIC_PAYLOAD_SHA256,
        "resources": dict(resources),
        "limit_val_batches": bounded_limit,
        "bounded_smoke_only": bounded_limit is not None,
        "batch_count": result["batch_count"],
        "sample_count": result["sample_count"],
        "paired_diagnostics": result["paired_diagnostics"],
        "registered_gates_and_diagnostics": gates,
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
        "interpretation_boundary": (
            "Functional pathway sensitivity under the fixed official MediaPipe-derived "
            "scaffold; not causal proof, not model selection, and D1-D4 are non-additive."
        ),
        "artifacts": artifact_hashes,
    }
    manifest_path = output_root / "probe_manifest.json"
    step6._write_json(manifest_path, manifest)
    return manifest


def write_equivalence_failure(
    output_root: Path, evidence: Mapping, *, checkpoint_path: Path
) -> None:
    if output_root.exists():
        raise FileExistsError(f"Fresh probe output must not already exist: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    step6._write_json(
        output_root / "native_manual_d0_equivalence.json", dict(evidence)
    )
    step6._write_json(
        output_root / "probe_manifest.json",
        {
            "schema_version": 1,
            "issue": ISSUE_NUMBER,
            "status": "INVALID_MANUAL_FORWARD_EQUIVALENCE",
            "checkpoint_path": str(checkpoint_path.resolve()),
            "scientific_interpretation": None,
            "training_performed": False,
            "test_access": False,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate registered D0-D5 direct-part pathway decomposition on "
            "validation only using one immutable .keras checkpoint."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-metadata", type=Path)
    parser.add_argument("--resolved-config", required=True, type=Path)
    parser.add_argument("--prior-root", required=True, type=Path)
    parser.add_argument("--clean-graph-cache-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--graph-workers", type=int)
    parser.add_argument("--graph-cache-size", type=int)
    parser.add_argument(
        "--limit-val-batches",
        type=int,
        help="Bounded implementation smoke only; forbidden for the registered run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint_path = args.checkpoint.resolve()
    metadata_path = (
        args.checkpoint_metadata.resolve()
        if args.checkpoint_metadata
        else step6.infer_checkpoint_metadata_path(checkpoint_path).resolve()
    )
    resolved_config_path = args.resolved_config.resolve()
    prior_root = args.prior_root.resolve()
    clean_graph_cache_dir = args.clean_graph_cache_dir.resolve()
    output_root = args.output_root.resolve()
    eval_batch_size = step6._require_positive("eval_batch_size", args.eval_batch_size)
    graph_workers = step6._require_positive("graph_workers", args.graph_workers)
    graph_cache_size = step6._require_positive(
        "graph_cache_size", args.graph_cache_size
    )
    limit_val_batches = step6._require_positive(
        "limit_val_batches", args.limit_val_batches
    )

    for path in (checkpoint_path, metadata_path, resolved_config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (prior_root, clean_graph_cache_dir):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if output_root.exists():
        raise FileExistsError(f"Fresh probe output must not already exist: {output_root}")

    raw_config = step6.load_persisted_resolved_config(resolved_config_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise DirectPartProbeError("Checkpoint metadata must be a JSON object")
    checkpoint_cross_check = step6.validate_checkpoint_metadata(raw_config, metadata)
    config = step6.build_runtime_config(raw_config)
    contract = step6.validate_frozen_contract(config)
    if contract["scientific_payload_sha256"] != EXPECTED_SCIENTIFIC_PAYLOAD_SHA256:
        raise DirectPartProbeError("Frozen scientific payload drift")

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
        step6.configure_gpu_memory_growth(
            bool(resources_config.get("memory_growth", True))
        )
    )

    checkpoint_sha256_before = sha256_file(checkpoint_path)
    model = load_fixed_checkpoint(checkpoint_path)
    model_weights_sha256_before = step6.model_weights_sha256(model)
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
    try:
        result = evaluate_conditions(
            model,
            validation_data.iter_epoch(0, limit_batches=limit_val_batches),
        )
    except ManualForwardEquivalenceError as exc:
        write_equivalence_failure(
            output_root, exc.evidence, checkpoint_path=checkpoint_path
        )
        raise

    model_weights_sha256_after = step6.model_weights_sha256(model)
    checkpoint_sha256_after = sha256_file(checkpoint_path)
    if checkpoint_sha256_after != checkpoint_sha256_before:
        raise DirectPartProbeError("Checkpoint changed during validation-only evaluation")
    if model_weights_sha256_after != model_weights_sha256_before:
        raise DirectPartProbeError("Model weights changed during inference")

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
    print(json.dumps(step6._json_ready(manifest), indent=2, sort_keys=True))
    status = manifest["registered_gates_and_diagnostics"]["status"]
    if status.startswith("INVALID_"):
        raise DirectPartProbeError(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
