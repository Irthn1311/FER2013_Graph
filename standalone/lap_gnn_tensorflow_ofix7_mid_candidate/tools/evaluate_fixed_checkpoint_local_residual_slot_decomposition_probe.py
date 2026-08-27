"""Issue #21 validation-only fixed-checkpoint local residual slot harness.

The harness reuses the SHA-locked reviewed Step-7 manual D0 forward to obtain
the official post-part_pool state.  It then varies only the four registered
local pooled residual embeddings immediately before readout.  It never builds
train/test data, changes graph tensors, or mutates the source batch/model.
"""

from __future__ import annotations

import argparse
import csv
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
from lap_gnn_tf.model.motif_layers import PART_ORDER
from lap_gnn_tf.resources import environment_manifest
from lap_gnn_tf.signatures import sha256_file
from lap_gnn_tf.training.losses import sparse_cross_entropy
from lap_gnn_tf.training.metrics import classification_metrics


TOOL_VERSION = "1.0.0"
ISSUE_NUMBER = 21
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
STEP7_TOOL_PATH = Path(__file__).with_name(
    "evaluate_fixed_checkpoint_direct_part_decomposition_probe.py"
)
STEP6_SUPPORT_PATH = Path(__file__).with_name(
    "evaluate_fixed_checkpoint_prior_probe.py"
)

EXPECTED_IMPLEMENTATION_BASE = "cd6a6b751d52729f7330adad58d94fbe7d1a7ac4"
EXPECTED_SCIENTIFIC_PAYLOAD_SHA256 = (
    "286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e"
)
EXPECTED_EXECUTION_CONTRACT_SHA256 = (
    "14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22"
)
EXPECTED_STEP7_TOOL_SHA256 = (
    "c0b1df778e469665dd6437c58831d29dcc34fbde44231db75894c5469a1ade78"
)
EXPECTED_STEP6_SUPPORT_SHA256 = (
    "3a033c977a29e102cfed75282ae7c1062f41feac8bef1b955ae425ec7e4004b3"
)


def _load_reviewed_tool(path: Path, expected_sha256: str, module_name: str):
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Reviewed tool identity drift: {path.name} "
            f"{actual_sha256} != {expected_sha256}"
        )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load reviewed tool: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Verify both identities independently before Step 7 dynamically imports Step 6.
if sha256_file(STEP6_SUPPORT_PATH) != EXPECTED_STEP6_SUPPORT_SHA256:
    raise RuntimeError("Reviewed Step-6 support-tool identity drift")
step7 = _load_reviewed_tool(
    STEP7_TOOL_PATH,
    EXPECTED_STEP7_TOOL_SHA256,
    "_issue21_reviewed_step7_direct_part_probe",
)
step6 = step7.step6
if step6.EXPECTED_EXECUTION_CONTRACT_SHA256 != EXPECTED_EXECUTION_CONTRACT_SHA256:
    raise RuntimeError("Reviewed Step-6 execution-contract identity drift")


CONDITION_S0 = "official_manual_forward"
CONDITION_S1 = "mouth_local_residual_zero"
CONDITION_S2 = "eye_local_residual_zero"
CONDITION_S3 = "brow_local_residual_zero"
CONDITION_S4 = "nose_cheek_local_residual_zero"
CONDITION_S5 = "all_local_residuals_zero_anchor"
CONDITIONS = (
    CONDITION_S0,
    CONDITION_S1,
    CONDITION_S2,
    CONDITION_S3,
    CONDITION_S4,
    CONDITION_S5,
)
LOCAL_PARTS = ("mouth", "eye", "brow", "nose_cheek")
if tuple(PART_ORDER[:4]) != LOCAL_PARTS or tuple(PART_ORDER[4:]) != ("global",):
    raise RuntimeError(f"Official pooled-part order drift: {PART_ORDER}")

SLOT_BY_CONDITION = {
    CONDITION_S1: "mouth",
    CONDITION_S2: "eye",
    CONDITION_S3: "brow",
    CONDITION_S4: "nose_cheek",
}
INTERVENTION_SPECS = {
    CONDITION_S0: {
        "zeroed_local_slots": [],
        "changed_pathway_arguments": [],
        "description": "Exact reviewed Step-7 D0 official manual forward.",
    },
    CONDITION_S1: {
        "zeroed_local_slots": ["mouth"],
        "changed_pathway_arguments": ["readout.part_embeddings.mouth"],
        "description": "Zero only the pooled mouth local residual before readout.",
    },
    CONDITION_S2: {
        "zeroed_local_slots": ["eye"],
        "changed_pathway_arguments": ["readout.part_embeddings.eye"],
        "description": "Zero only the pooled eye local residual before readout.",
    },
    CONDITION_S3: {
        "zeroed_local_slots": ["brow"],
        "changed_pathway_arguments": ["readout.part_embeddings.brow"],
        "description": "Zero only the pooled brow local residual before readout.",
    },
    CONDITION_S4: {
        "zeroed_local_slots": ["nose_cheek"],
        "changed_pathway_arguments": ["readout.part_embeddings.nose_cheek"],
        "description": (
            "Zero only the pooled nose_cheek local residual before readout."
        ),
    },
    CONDITION_S5: {
        "zeroed_local_slots": list(LOCAL_PARTS),
        "changed_pathway_arguments": [
            *(f"readout.part_embeddings.{name}" for name in LOCAL_PARTS)
        ],
        "description": (
            "Zero all four local pooled residuals exactly as reviewed Step-8 D3; "
            "global/validity/readout prior and upstream state remain official."
        ),
    },
}

NATIVE_MANUAL_TOLERANCE = {
    "prediction_agreement": 1.0,
    "max_abs_logit_difference": 1e-5,
    "max_abs_probability_difference": 3e-6,
}
S0_REFERENCE = {
    "accuracy": 0.63137364168292,
    "macro_f1": 0.5932591901893336,
    "loss": 1.1537981840361535,
}
S5_REFERENCE = {
    "accuracy": 0.22596823627751464,
    "macro_f1": 0.1958426679087715,
    "loss": 1.883221954371022,
}
REFERENCE_TOLERANCE = {"accuracy": 0.001, "macro_f1": 0.001, "loss": 0.005}
EXPECTED_FULL_VALIDATION_SAMPLES = 3589
SLOT_HIGH_THRESHOLD_PP = 10.0
SLOT_MODERATE_THRESHOLD_PP = 5.0


class LocalResidualSlotProbeError(RuntimeError):
    """Fail-closed Issue #21 harness error."""


class ManualForwardEquivalenceError(LocalResidualSlotProbeError):
    """Native/manual S0 identity failure with preservation evidence."""

    def __init__(self, evidence: Mapping):
        self.evidence = dict(evidence)
        super().__init__("INVALID_MANUAL_FORWARD_EQUIVALENCE")


def _require_condition(condition: str) -> None:
    if condition not in CONDITIONS:
        raise LocalResidualSlotProbeError(
            f"Unknown local residual slot condition: {condition!r}"
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
        raise LocalResidualSlotProbeError("Source batch field set changed")
    for name, tensor in batch.items():
        if not np.array_equal(np.asarray(tensor.numpy()), snapshot[name]):
            raise LocalResidualSlotProbeError(
                f"Source batch tensor changed during slot decomposition: {name}"
            )


def _assert_equal(actual, expected, message: str) -> None:
    try:
        tf.debugging.assert_equal(actual, expected, message=message)
    except (tf.errors.InvalidArgumentError, ValueError) as exc:
        raise LocalResidualSlotProbeError(message) from exc


def _official_manual_state(model, batch: Mapping[str, tf.Tensor]):
    output, trace = step7.manual_forward(model, batch, step7.CONDITION_D0)
    trace = dict(trace)
    trace["condition"] = CONDITION_S0
    trace["reviewed_step7_condition"] = step7.CONDITION_D0
    return output, trace


def _zeroed_slots(condition: str) -> tuple[str, ...]:
    if condition == CONDITION_S5:
        return LOCAL_PARTS
    slot = SLOT_BY_CONDITION.get(condition)
    return (slot,) if slot is not None else ()


def _forward_from_official_state(
    model,
    batch: Mapping[str, tf.Tensor],
    condition: str,
    official_output: Mapping[str, tf.Tensor],
    official_trace: Mapping,
) -> tuple[dict[str, tf.Tensor], dict]:
    _require_condition(condition)
    if condition == CONDITION_S0:
        return dict(official_output), dict(official_trace)

    pooled = official_trace["pooled_before_readout_intervention"]
    valid = official_trace["valid_before_readout_intervention"]
    readout_embeddings = {name: value for name, value in pooled.items()}
    for name in _zeroed_slots(condition):
        readout_embeddings[name] = tf.zeros_like(pooled[name])
    readout_valid = {name: value for name, value in valid.items()}

    node_embeddings = official_output["node_embeddings"]
    node_features = official_trace["message_passing"]["node_features"]
    node_graph_index = official_trace["message_passing"]["node_graph_index"]
    num_graphs = tf.shape(batch["labels"])[0]
    readout = model.readout(
        node_embeddings,
        node_features,
        official_trace["readout_part_soft"],
        node_graph_index,
        num_graphs,
        readout_embeddings,
        readout_valid,
        training=False,
    )
    logits = model.classifier(readout["z_image"], training=False)
    output = {
        "logits": logits,
        "probabilities": tf.nn.softmax(logits, axis=-1),
        "predictions": tf.argmax(logits, axis=1, output_type=tf.int64),
        "z_image": readout["z_image"],
        "node_embeddings": node_embeddings,
        "part_embeddings": readout_embeddings,
    }
    trace = dict(official_trace)
    trace.update(
        {
            "condition": condition,
            "reviewed_step7_condition": (
                step7.CONDITION_D3 if condition == CONDITION_S5 else None
            ),
            "zeroed_local_slots": list(_zeroed_slots(condition)),
            "readout_part_embeddings": readout_embeddings,
            "readout_valid_groups": readout_valid,
        }
    )
    return output, trace


def manual_forward(
    model, batch: Mapping[str, tf.Tensor], condition: str
) -> tuple[dict[str, tf.Tensor], dict]:
    """Run one fixed registered condition from the reviewed official state."""
    _require_condition(condition)
    step6.validate_batch_schema(batch)
    official_output, official_trace = _official_manual_state(model, batch)
    return _forward_from_official_state(
        model, batch, condition, official_output, official_trace
    )


def validate_slot_integrity(
    model,
    batch: Mapping[str, tf.Tensor],
    snapshot: Mapping[str, np.ndarray],
    condition: str,
    trace: Mapping,
) -> dict:
    """Prove that only the registered pooled local residual slot(s) changed."""
    _require_condition(condition)
    _assert_source_unchanged(batch, snapshot)
    boundary_batch = step7._normalize_model_boundary_inputs(model, batch)
    official_part_soft = tf.cast(boundary_batch["part_soft"], tf.float32)
    official_valid_part_mask = tf.cast(
        boundary_batch["valid_part_mask"], tf.float32
    )
    _assert_equal(trace["context_part_soft"], official_part_soft, "Context drift")
    _assert_equal(trace["pool_part_soft"], official_part_soft, "Pool prior drift")
    _assert_equal(
        trace["pool_valid_part_mask"],
        official_valid_part_mask,
        "Pool validity drift",
    )
    _assert_equal(trace["readout_part_soft"], official_part_soft, "Readout prior drift")

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
        _assert_equal(message[name], expected, f"Upstream graph input drift: {name}")

    expected_boundary = {
        "autocast": bool(model.autocast),
        "input_dtype": tf.dtypes.as_dtype(model.input_dtype).name,
        "source_dtypes": {name: value.dtype.name for name, value in batch.items()},
        "effective_dtypes": {
            name: value.dtype.name for name, value in boundary_batch.items()
        },
    }
    if trace.get("model_boundary") != expected_boundary:
        raise LocalResidualSlotProbeError("Keras model-boundary semantics drift")

    pooled = trace["pooled_before_readout_intervention"]
    readout_pooled = trace["readout_part_embeddings"]
    valid = trace["valid_before_readout_intervention"]
    readout_valid = trace["readout_valid_groups"]
    zeroed = set(_zeroed_slots(condition))
    for name in PART_ORDER:
        expected_embedding = (
            tf.zeros_like(pooled[name]) if name in zeroed else pooled[name]
        )
        _assert_equal(
            readout_pooled[name],
            expected_embedding,
            f"Readout pooled embedding drift: {condition}/{name}",
        )
        _assert_equal(
            readout_valid[name], valid[name],
            f"Readout validity drift: {condition}/{name}",
        )
    if trace.get("zeroed_local_slots", list(zeroed)) != list(
        _zeroed_slots(condition)
    ):
        raise LocalResidualSlotProbeError("Recorded zeroed-slot identity drift")

    return {
        "condition": condition,
        "zeroed_local_slots": list(_zeroed_slots(condition)),
        "changed_pathway_arguments": list(
            INTERVENTION_SPECS[condition]["changed_pathway_arguments"]
        ),
        "source_batch_unchanged": True,
        "labels_and_sample_ids_unchanged": True,
        "message_passing_inputs_unchanged": True,
        "context_output_unchanged": True,
        "node_edge_coordinates_topology_unchanged": True,
        "global_embedding_unchanged": True,
        "validity_flags_unchanged": True,
        "readout_part_soft_unchanged": True,
        "registered_slot_arguments_exact": True,
        "model_boundary_input_semantics": expected_boundary,
    }


def native_manual_equivalence(
    native_output: Mapping[str, tf.Tensor],
    manual_output: Mapping[str, tf.Tensor],
    sample_ids: tf.Tensor,
) -> dict:
    evidence = step7.native_manual_equivalence(
        native_output, manual_output, sample_ids
    )
    evidence["tolerances"] = dict(NATIVE_MANUAL_TOLERANCE)
    return evidence


def load_fixed_checkpoint(checkpoint: str | Path):
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.suffix.lower() != ".keras":
        raise LocalResidualSlotProbeError("Issue #21 requires a .keras checkpoint")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    model = tf.keras.models.load_model(
        checkpoint_path,
        custom_objects={"LapGNN": LapGNN, "lap_gnn_tf>LapGNN": LapGNN},
        compile=False,
    )
    if getattr(model, "optimizer", None) is not None:
        raise LocalResidualSlotProbeError(
            "compile=False checkpoint load unexpectedly restored an optimizer"
        )
    if int(model.count_params()) != EXPECTED_PARAMETER_COUNT:
        raise LocalResidualSlotProbeError(
            f"Loaded model parameter count drift: {model.count_params()}"
        )
    return model


def _paired_diagnostics(
    labels: np.ndarray, probabilities: Mapping[str, np.ndarray]
) -> dict:
    s0_predictions = probabilities[CONDITION_S0].argmax(axis=1)
    s0_correct = s0_predictions == labels
    result = {}
    for condition in CONDITIONS[1:]:
        predictions = probabilities[condition].argmax(axis=1)
        condition_correct = predictions == labels
        result[condition] = {
            "prediction_disagreement_count": int(
                np.count_nonzero(predictions != s0_predictions)
            ),
            "prediction_disagreement_rate": float(
                np.mean(predictions != s0_predictions)
            ),
            "correctness_transitions": {
                "s0_correct_to_intervention_incorrect": {
                    "count": int(np.count_nonzero(s0_correct & ~condition_correct)),
                    "rate": float(np.mean(s0_correct & ~condition_correct)),
                },
                "s0_incorrect_to_intervention_correct": {
                    "count": int(np.count_nonzero(~s0_correct & condition_correct)),
                    "rate": float(np.mean(~s0_correct & condition_correct)),
                },
                "unchanged_correct": {
                    "count": int(np.count_nonzero(s0_correct & condition_correct)),
                    "rate": float(np.mean(s0_correct & condition_correct)),
                },
                "unchanged_incorrect": {
                    "count": int(np.count_nonzero(~s0_correct & ~condition_correct)),
                    "rate": float(np.mean(~s0_correct & ~condition_correct)),
                },
            },
        }
    return result


def evaluate_conditions(
    model, batches: Iterable[Mapping[str, tf.Tensor]]
) -> dict:
    """Evaluate fixed S0-S5 paired on each source batch before advancing."""
    labels_parts = []
    sample_id_parts = []
    probability_parts = {condition: [] for condition in CONDITIONS}
    losses = {condition: [] for condition in CONDITIONS}
    integrity_counts = {condition: 0 for condition in CONDITIONS}
    equivalence_batches = []
    model_boundary_input_semantics = None
    batch_count = 0

    for source_batch in batches:
        step6.validate_batch_schema(source_batch)
        snapshot = _snapshot_batch(source_batch)
        native_output = model(source_batch, training=False)
        official_output, official_trace = _official_manual_state(model, source_batch)
        equivalence = native_manual_equivalence(
            native_output, official_output, source_batch["sample_ids"]
        )
        equivalence["batch_index"] = batch_count
        equivalence_batches.append(equivalence)
        if not equivalence["gate_pass"]:
            raise ManualForwardEquivalenceError(
                {
                    "status": "INVALID_MANUAL_FORWARD_EQUIVALENCE",
                    "failed_batch_index": batch_count,
                    "tolerances": NATIVE_MANUAL_TOLERANCE,
                    "batch_evidence": equivalence,
                    "completed_batches": equivalence_batches,
                }
            )

        outputs = {}
        traces = {}
        for condition in CONDITIONS:
            output, trace = _forward_from_official_state(
                model,
                source_batch,
                condition,
                official_output,
                official_trace,
            )
            outputs[condition] = output
            traces[condition] = trace
            integrity = validate_slot_integrity(
                model, source_batch, snapshot, condition, trace
            )
            boundary = integrity["model_boundary_input_semantics"]
            if model_boundary_input_semantics is None:
                model_boundary_input_semantics = boundary
            elif model_boundary_input_semantics != boundary:
                raise LocalResidualSlotProbeError(
                    "Model-boundary semantics changed between conditions/batches"
                )
            integrity_counts[condition] += 1
            probabilities = np.asarray(
                output["probabilities"].numpy(), dtype=np.float64
            )
            labels = np.asarray(source_batch["labels"].numpy(), dtype=np.int64)
            if probabilities.shape != (labels.size, 7):
                raise LocalResidualSlotProbeError(
                    f"Condition {condition} probability shape drift: "
                    f"{probabilities.shape}"
                )
            probability_parts[condition].append(probabilities)
            losses[condition].append(
                float(
                    sparse_cross_entropy(
                        source_batch["labels"], output["logits"]
                    ).numpy()
                )
            )
            _assert_source_unchanged(source_batch, snapshot)
        labels_parts.append(labels)
        sample_id_parts.append(
            np.asarray(source_batch["sample_ids"].numpy(), dtype=np.int64)
        )
        batch_count += 1

    if not batch_count:
        raise LocalResidualSlotProbeError("No validation batches were produced")
    labels = np.concatenate(labels_parts)
    sample_ids = np.concatenate(sample_id_parts)
    if len(np.unique(sample_ids)) != len(sample_ids):
        raise LocalResidualSlotProbeError("Validation sample IDs are not unique")
    probabilities = {
        condition: np.concatenate(parts, axis=0)
        for condition, parts in probability_parts.items()
    }
    metrics = {}
    for condition in CONDITIONS:
        item = classification_metrics(labels, probabilities[condition])
        item["loss"] = float(np.mean(losses[condition]))
        metrics[condition] = item
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
            "gate_a_native_manual_s0_equivalence": gate_a,
            "gate_b_s0_reference": {"status": "NOT_EVALUATED_BOUNDED_SMOKE"},
            "gate_c_s5_d3_anchor": {"status": "NOT_EVALUATED_BOUNDED_SMOKE"},
            "per_slot_diagnostics": None,
            "overall_decision": None,
        }

    sample_count_exact = result["sample_count"] == EXPECTED_FULL_VALIDATION_SAMPLES
    s0_checks = _reference_checks(result["metrics"][CONDITION_S0], S0_REFERENCE)
    s5_checks = _reference_checks(result["metrics"][CONDITION_S5], S5_REFERENCE)
    gate_b_pass = sample_count_exact and all(item["pass"] for item in s0_checks.values())
    gate_c_pass = sample_count_exact and all(item["pass"] for item in s5_checks.values())
    gate_b = {
        "status": "PASS" if gate_b_pass else "FAIL",
        "sample_count": result["sample_count"],
        "required_sample_count": EXPECTED_FULL_VALIDATION_SAMPLES,
        "sample_count_exact": sample_count_exact,
        "checks": s0_checks,
    }
    gate_c = {
        "status": "PASS" if gate_c_pass else "FAIL",
        "sample_count": result["sample_count"],
        "required_sample_count": EXPECTED_FULL_VALIDATION_SAMPLES,
        "sample_count_exact": sample_count_exact,
        "checks": s5_checks,
    }
    if not gate_b_pass:
        status = "INVALID_S0_REFERENCE_REPRODUCTION"
    elif not gate_c_pass:
        status = "INVALID_D3_ANCHOR_REPRODUCTION"
    else:
        status = "VALID_REGISTERED_SLOT_DECOMPOSITION"
    if status != "VALID_REGISTERED_SLOT_DECOMPOSITION":
        return {
            "status": status,
            "gate_a_native_manual_s0_equivalence": gate_a,
            "gate_b_s0_reference": gate_b,
            "gate_c_s5_d3_anchor": gate_c,
            "per_slot_diagnostics": None,
            "overall_decision": None,
        }

    s0_f1 = float(result["metrics"][CONDITION_S0]["macro_f1"])
    per_slot = {}
    high_count = 0
    for condition in CONDITIONS[1:5]:
        delta = 100.0 * (
            s0_f1 - float(result["metrics"][condition]["macro_f1"])
        )
        threshold_delta = round(delta, 12)
        if threshold_delta >= SLOT_HIGH_THRESHOLD_PP:
            label = "HIGH_SLOT_SENSITIVITY"
            high_count += 1
        elif threshold_delta >= SLOT_MODERATE_THRESHOLD_PP:
            label = "MODERATE_SLOT_SENSITIVITY"
        else:
            label = "LOW_SLOT_SENSITIVITY"
        per_slot[condition] = {
            "delta_f1_pp": delta,
            "label": label,
            "negative_effect_note": (
                "Intervention improved macro-F1; negative delta retained exactly."
                if delta < 0.0
                else None
            ),
        }
    if high_count == 1:
        decision = "SINGLE_HIGH_LOCAL_SLOT"
    elif high_count >= 2:
        decision = "MULTIPLE_HIGH_LOCAL_SLOTS"
    else:
        decision = "NO_SINGLE_HIGH_LOCAL_SLOT_WITH_JOINT_DEPENDENCY"
    return {
        "status": status,
        "gate_a_native_manual_s0_equivalence": gate_a,
        "gate_b_s0_reference": gate_b,
        "gate_c_s5_d3_anchor": gate_c,
        "per_slot_diagnostics": per_slot,
        "overall_decision": decision,
        "non_additivity_warning": (
            "S1-S4 are nonlinear diagnostic sensitivities; do not sum them, "
            "divide by S5, or treat them as additive causal contributions."
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
    equivalence_path = output_root / "native_manual_s0_equivalence.json"
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
        "coordinates_changed": False,
        "topology_changed": False,
        "labels_or_sample_ids_changed": False,
        "global_embedding_changed": False,
        "validity_flags_changed": False,
        "readout_part_soft_changed": False,
        "context_or_upstream_state_changed": False,
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
        "implementation_base": EXPECTED_IMPLEMENTATION_BASE,
        "split": "val",
        "validation_only": True,
        "condition_order": list(CONDITIONS),
        "conditions": INTERVENTION_SPECS,
        "reviewed_step7_tool_sha256": EXPECTED_STEP7_TOOL_SHA256,
        "reviewed_step6_support_sha256": EXPECTED_STEP6_SUPPORT_SHA256,
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
            "Functional learned local pooled residual-slot sensitivity under the "
            "fixed official MediaPipe-derived scaffold; not MediaPipe-prior "
            "isolation, not causal attribution, and S1-S4 are non-additive."
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
        output_root / "native_manual_s0_equivalence.json", dict(evidence)
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
            "Evaluate fixed S0-S5 local residual slot decomposition on validation "
            "only using one immutable .keras checkpoint."
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
        help=(
            "Bounded implementation smoke only with NO_SCIENTIFIC_INTERPRETATION; "
            "forbidden for a future registered run."
        ),
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
        raise LocalResidualSlotProbeError(
            "Checkpoint metadata must be a JSON object"
        )
    checkpoint_cross_check = step6.validate_checkpoint_metadata(raw_config, metadata)
    config = step6.build_runtime_config(raw_config)
    contract = step6.validate_frozen_contract(config)
    if contract["scientific_payload_sha256"] != EXPECTED_SCIENTIFIC_PAYLOAD_SHA256:
        raise LocalResidualSlotProbeError("Frozen scientific payload drift")
    if contract.get("execution_contract_sha256") != EXPECTED_EXECUTION_CONTRACT_SHA256:
        raise LocalResidualSlotProbeError("Frozen execution contract drift")

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
        raise LocalResidualSlotProbeError(
            "Checkpoint changed during validation-only evaluation"
        )
    if model_weights_sha256_after != model_weights_sha256_before:
        raise LocalResidualSlotProbeError("Model weights changed during inference")

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
        raise LocalResidualSlotProbeError(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
