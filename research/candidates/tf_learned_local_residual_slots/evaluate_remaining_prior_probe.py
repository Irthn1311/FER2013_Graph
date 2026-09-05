"""Issue #38 validation-only learned-slot remaining-prior probe.

The tool evaluates P0-P9 from each already-constructed official graph batch.
It performs inference only, never rebuilds topology, and never constructs a
train or test split.  Scientific execution is intentionally deferred.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
FROZEN_PACKAGE_ROOT = (
    REPOSITORY_ROOT / "standalone" / "lap_gnn_tensorflow_ofix7_mid_candidate"
)
FROZEN_PACKAGE_SRC = FROZEN_PACKAGE_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(FROZEN_PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(FROZEN_PACKAGE_SRC))

import numpy as np
import tensorflow as tf

from lap_gnn_tf.data.graph_generator import GraphBatchGenerator  # noqa: E402
from lap_gnn_tf.model.motif_layers import PART_ORDER, part_pool  # noqa: E402
from lap_gnn_tf.signatures import scientific_payload_checksum, sha256_file  # noqa: E402
from lap_gnn_tf.training.losses import sparse_cross_entropy  # noqa: E402
from lap_gnn_tf.training.metrics import classification_metrics  # noqa: E402

from research.candidates.tf_learned_local_residual_slots.model import (  # noqa: E402
    HIDDEN_DIM,
    NUM_LOCAL_SLOTS,
    LearnedLocalResidualSlotLapGNN,
    LearnedLocalResidualSlotPool,
)


TOOL_VERSION = "1.0.1"
ISSUE_NUMBER = 38
CANDIDATE_MODEL_PATH = Path(__file__).with_name("model.py")
STEP6_SUPPORT_PATH = FROZEN_PACKAGE_ROOT / "tools/evaluate_fixed_checkpoint_prior_probe.py"


def _load_step6_support():
    spec = importlib.util.spec_from_file_location(
        "_issue38_step6_probe_support", STEP6_SUPPORT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load frozen Step-6 support: {STEP6_SUPPORT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


step6 = _load_step6_support()

EXPECTED_BASE_COMMIT = "e9b4deec2d4986b4a94fce32f3c1586cdb301047"
EXPECTED_CANDIDATE_MODEL_SHA256 = (
    "0a7cfa315baf0d6666b7ab86139b328ab6d138985cfddd71180410675602fcca"
)
EXPECTED_STEP12E_ARCHIVE_SHA256 = (
    "f436b0a7a20c751b2fd2f47738469fb409ecf9a1a40628e05d20974639927451"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "e0d633cb6200e963f31a28750e28c7febdaae40344c90ba9d94b826a09e4b78c"
)
EXPECTED_WEIGHTS_SHA256 = (
    "a18a372f70ce56868ae43257e9b7fa5e20517499c2c1e35c48dba4d65eaaaa74"
)
EXPECTED_METADATA_SHA256 = (
    "a5ee759bc6fbef587e025199d0dcfe6ebd3a1764cffa567f793c53e972eb47cf"
)
EXPECTED_RESOLVED_CONFIG_SHA256 = (
    "3c028dd2f32ebed3a252544e170220b150b5e29920cea865924dddce6aef5a32"
)
EXPECTED_CHECKPOINT_EPOCH = 42
EXPECTED_MODEL_CLASS = "LearnedLocalResidualSlotLapGNN"
EXPECTED_PARAMETER_COUNT = 1_061_576
EXPECTED_TRAINABLE_VARIABLE_COUNT = 128
EXPECTED_Q_INDEX = 127
EXPECTED_Q_SHAPE = (4, 96)
EXPECTED_Q_DTYPE = "float32"
EXPECTED_Q_SHA256 = (
    "54b368aa183c65d5843d8b8e340d3020412d1a2dfeaabbe8b2c0166684ab3ff9"
)
EXPECTED_SCIENTIFIC_PAYLOAD_SHA256 = (
    "286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e"
)
EXPECTED_EXECUTION_CONTRACT_SHA256 = (
    "14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22"
)

EXPECTED_FULL_VALIDATION_SAMPLES = 3_589
P0_REFERENCE = {
    "accuracy": 0.6232933964892727,
    "macro_f1": 0.596090717851928,
    "loss": 1.1486882999934982,
}
REFERENCE_TOLERANCE = {"accuracy": 0.001, "macro_f1": 0.001, "loss": 0.005}
NATIVE_MANUAL_TOLERANCE = {
    "prediction_agreement": 1.0,
    "max_abs_logit_difference": 1e-5,
    "max_abs_probability_difference": 3e-6,
}

P0 = "official_candidate_manual_forward"
P1 = "node_face_mask_zero_fixed_graph"
P2 = "node_part_soft_channels_zero_fixed_graph"
P3 = "node_distance_map_channels_zero_fixed_graph"
P4 = "node_landmark_missing_flag_zero_fixed_graph"
P5 = "edge_semantic_channels_zero_fixed_graph"
P6 = "context_direct_part_soft_neutralized"
P7 = "readout_direct_part_soft_neutralized"
P8 = "readout_validity_off"
P9 = "all_explicit_semantic_prior_zero_fixed_topology_anchor"
CONDITIONS = (P0, P1, P2, P3, P4, P5, P6, P7, P8, P9)
INDIVIDUAL_CONDITIONS = CONDITIONS[1:9]
LOCAL_PARTS = tuple(PART_ORDER[:NUM_LOCAL_SLOTS])

INTERVENTION_SPECS = {
    P0: {"changed_paths": []},
    P1: {"changed_paths": ["node_features[:,5]"]},
    P2: {"changed_paths": ["node_features[:,6:19]"]},
    P3: {"changed_paths": ["node_features[:,19:31]"]},
    P4: {"changed_paths": ["node_features[:,31]"]},
    P5: {"changed_paths": ["edge_features[:,6:8]"]},
    P6: {"changed_paths": ["gnn.encode.part_soft"]},
    P7: {"changed_paths": ["readout.part_soft"]},
    P8: {
        "changed_paths": [
            *(f"readout.valid_groups.{name}" for name in LOCAL_PARTS)
        ]
    },
    P9: {
        "changed_paths": [
            "node_features[:,5:32]",
            "edge_features[:,6:8]",
            "gnn.encode.part_soft",
            "readout.part_soft",
            *(f"readout.valid_groups.{name}" for name in LOCAL_PARTS),
        ]
    },
}

TOPOLOGY_FIELDS = (
    "edge_index",
    "node_graph_index",
    "edge_graph_index",
    "graph_node_counts",
    "graph_edge_counts",
    "coordinates",
    "labels",
    "sample_ids",
)


class RemainingPriorProbeError(RuntimeError):
    """Fail-closed Issue #38 probe error."""


class ManualForwardEquivalenceError(RemainingPriorProbeError):
    """P0 failed the registered native/manual equivalence gate."""

    def __init__(self, evidence: Mapping[str, Any]):
        self.evidence = dict(evidence)
        super().__init__("INVALID_MANUAL_FORWARD_EQUIVALENCE")


def _require_condition(condition: str) -> None:
    if condition not in CONDITIONS:
        raise RemainingPriorProbeError(f"Unknown registered condition: {condition!r}")


def _normalize_model_boundary_inputs(
    model: tf.keras.Model, batch: Mapping[str, tf.Tensor]
) -> dict[str, tf.Tensor]:
    """Mirror Keras Model.__call__ boundary autocasting before manual call entry."""

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


def snapshot_batch(batch: Mapping[str, tf.Tensor]) -> dict[str, np.ndarray]:
    return {name: np.array(value.numpy(), copy=True) for name, value in batch.items()}


def assert_source_batch_unchanged(
    batch: Mapping[str, tf.Tensor], snapshot: Mapping[str, np.ndarray]
) -> None:
    if set(batch) != set(snapshot):
        raise RemainingPriorProbeError("Source batch field set changed")
    for name, value in batch.items():
        if not np.array_equal(np.asarray(value.numpy()), snapshot[name]):
            raise RemainingPriorProbeError(f"Source batch mutated: {name}")


def _assert_equal(actual: Any, expected: Any, message: str) -> None:
    try:
        tf.debugging.assert_equal(actual, expected, message=message)
    except (tf.errors.InvalidArgumentError, ValueError) as exc:
        raise RemainingPriorProbeError(message) from exc


def _zero_feature_slice(
    values: tf.Tensor, start: int, stop: int
) -> tf.Tensor:
    return tf.concat(
        (values[:, :start], tf.zeros_like(values[:, start:stop]), values[:, stop:]),
        axis=1,
    )


def _effective_features(
    node_features: tf.Tensor, edge_features: tf.Tensor, condition: str
) -> tuple[tf.Tensor, tf.Tensor]:
    if condition == P1:
        node_features = _zero_feature_slice(node_features, 5, 6)
    elif condition == P2:
        node_features = _zero_feature_slice(node_features, 6, 19)
    elif condition == P3:
        node_features = _zero_feature_slice(node_features, 19, 31)
    elif condition == P4:
        node_features = _zero_feature_slice(node_features, 31, 32)
    elif condition == P9:
        node_features = _zero_feature_slice(node_features, 5, 32)
    if condition in (P5, P9):
        edge_features = _zero_feature_slice(edge_features, 6, 8)
    return node_features, edge_features


def manual_forward(
    model: LearnedLocalResidualSlotLapGNN,
    batch: Mapping[str, tf.Tensor],
    condition: str,
) -> tuple[dict[str, tf.Tensor], dict[str, Any]]:
    """Mechanically mirror candidate inference with registered path overrides."""

    _require_condition(condition)
    boundary = _normalize_model_boundary_inputs(model, batch)
    official_node_features = tf.cast(boundary["node_features"], tf.float32)
    official_edge_features = tf.cast(boundary["edge_features"], tf.float32)
    node_features, edge_features = _effective_features(
        official_node_features, official_edge_features, condition
    )
    edge_index = tf.cast(boundary["edge_index"], tf.int64)
    node_graph_index = tf.cast(boundary["node_graph_index"], tf.int32)
    official_part_soft = tf.cast(boundary["part_soft"], tf.float32)
    official_valid_part_mask = tf.cast(boundary["valid_part_mask"], tf.float32)
    num_graphs = tf.shape(boundary["labels"])[0]

    context_part_soft = (
        tf.zeros_like(official_part_soft) if condition in (P6, P9) else official_part_soft
    )
    h = model.encoder(node_features, training=False)
    h = model.gnn.encode(
        h,
        edge_index,
        edge_features,
        node_graph_index,
        num_graphs,
        context_part_soft,
        training=False,
        collect=False,
    )

    official_pooled, official_valid_groups = part_pool(
        h,
        official_part_soft,
        node_graph_index,
        official_valid_part_mask,
        num_graphs,
    )
    slot_diagnostics = model.learned_local_residual_slots(
        h, node_graph_index, num_graphs
    )
    raw_slots = slot_diagnostics["slot_embeddings"]
    residual_slots = tf.cast(raw_slots, official_pooled["global"].dtype)
    unpacked_slots = tf.unstack(residual_slots, num=NUM_LOCAL_SLOTS, axis=1)
    residual_embeddings = {
        name: unpacked_slots[index] for index, name in enumerate(LOCAL_PARTS)
    }
    residual_embeddings["global"] = official_pooled["global"]

    readout_part_soft = (
        tf.zeros_like(official_part_soft) if condition in (P7, P9) else official_part_soft
    )
    readout_valid_groups = dict(official_valid_groups)
    if condition in (P8, P9):
        for name in LOCAL_PARTS:
            readout_valid_groups[name] = tf.zeros_like(
                official_valid_groups[name], dtype=tf.bool
            )
        readout_valid_groups["global"] = tf.ones_like(
            official_valid_groups["global"], dtype=tf.bool
        )

    readout = model.readout(
        h,
        node_features,
        readout_part_soft,
        node_graph_index,
        num_graphs,
        residual_embeddings,
        readout_valid_groups,
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
        "learned_local_residual_slots": raw_slots,
        "learned_local_attention_entropy": slot_diagnostics["attention_entropy"],
        "learned_local_attention_peak": slot_diagnostics["attention_peak"],
        "official_global_residual": official_pooled["global"],
    }
    trace = {
        "condition": condition,
        "official_node_features": official_node_features,
        "effective_node_features": node_features,
        "official_edge_features": official_edge_features,
        "effective_edge_features": edge_features,
        "official_part_soft": official_part_soft,
        "context_part_soft": context_part_soft,
        "part_pool_part_soft": official_part_soft,
        "readout_part_soft": readout_part_soft,
        "official_valid_part_mask": official_valid_part_mask,
        "part_pool_valid_part_mask": official_valid_part_mask,
        "official_readout_valid_groups": official_valid_groups,
        "readout_valid_groups": readout_valid_groups,
        "residual_part_embeddings": residual_embeddings,
        "topology": {name: boundary[name] for name in TOPOLOGY_FIELDS},
        "model_boundary": {
            "autocast": bool(model.autocast),
            "input_dtype": tf.dtypes.as_dtype(model.input_dtype).name,
        },
    }
    return output, trace


def validate_pathway_integrity(
    model: LearnedLocalResidualSlotLapGNN,
    batch: Mapping[str, tf.Tensor],
    snapshot: Mapping[str, np.ndarray],
    condition: str,
    trace: Mapping[str, Any],
) -> dict[str, Any]:
    _require_condition(condition)
    assert_source_batch_unchanged(batch, snapshot)
    boundary = _normalize_model_boundary_inputs(model, batch)
    official_node = tf.cast(boundary["node_features"], tf.float32)
    official_edge = tf.cast(boundary["edge_features"], tf.float32)
    expected_node, expected_edge = _effective_features(
        official_node, official_edge, condition
    )
    official_part_soft = tf.cast(boundary["part_soft"], tf.float32)
    expected_context = (
        tf.zeros_like(official_part_soft) if condition in (P6, P9) else official_part_soft
    )
    expected_readout = (
        tf.zeros_like(official_part_soft) if condition in (P7, P9) else official_part_soft
    )
    for actual, expected, label in (
        (trace["official_node_features"], official_node, "official node features"),
        (trace["effective_node_features"], expected_node, "effective node features"),
        (trace["official_edge_features"], official_edge, "official edge features"),
        (trace["effective_edge_features"], expected_edge, "effective edge features"),
        (trace["context_part_soft"], expected_context, "context part_soft"),
        (trace["part_pool_part_soft"], official_part_soft, "part_pool part_soft"),
        (trace["readout_part_soft"], expected_readout, "readout part_soft"),
        (
            trace["part_pool_valid_part_mask"],
            tf.cast(boundary["valid_part_mask"], tf.float32),
            "part_pool validity",
        ),
    ):
        _assert_equal(actual, expected, f"Registered pathway drift: {label}")
    for name in TOPOLOGY_FIELDS:
        _assert_equal(trace["topology"][name], boundary[name], f"Topology drift: {name}")
    official_valid = trace["official_readout_valid_groups"]
    readout_valid = trace["readout_valid_groups"]
    for name in PART_ORDER:
        expected = (
            tf.zeros_like(official_valid[name], dtype=tf.bool)
            if condition in (P8, P9) and name in LOCAL_PARTS
            else tf.ones_like(official_valid[name], dtype=tf.bool)
            if condition in (P8, P9) and name == "global"
            else official_valid[name]
        )
        _assert_equal(readout_valid[name], expected, f"Readout validity drift: {name}")
    return {
        "condition": condition,
        "changed_paths": list(INTERVENTION_SPECS[condition]["changed_paths"]),
        "source_batch_unchanged": True,
        "topology_unchanged": True,
        "part_pool_prior_official": True,
        "part_pool_validity_official": True,
        "registered_paths_exact": True,
    }


def native_manual_equivalence(
    native_output: Mapping[str, tf.Tensor],
    manual_output: Mapping[str, tf.Tensor],
    sample_ids: tf.Tensor,
) -> dict[str, Any]:
    native_logits = np.asarray(native_output["logits"].numpy(), dtype=np.float64)
    manual_logits = np.asarray(manual_output["logits"].numpy(), dtype=np.float64)
    native_probabilities = np.asarray(
        native_output["probabilities"].numpy(), dtype=np.float64
    )
    manual_probabilities = np.asarray(
        manual_output["probabilities"].numpy(), dtype=np.float64
    )
    if native_logits.shape != manual_logits.shape:
        raise RemainingPriorProbeError("Native/P0 logit shape drift")
    if native_probabilities.shape != manual_probabilities.shape:
        raise RemainingPriorProbeError("Native/P0 probability shape drift")
    agreement = float(
        np.mean(native_probabilities.argmax(axis=1) == manual_probabilities.argmax(axis=1))
    )
    evidence = {
        "sample_count": int(native_logits.shape[0]),
        "sample_ids_sha256": hashlib.sha256(
            np.asarray(sample_ids.numpy(), dtype=np.int64).tobytes(order="C")
        ).hexdigest(),
        "prediction_agreement": agreement,
        "max_abs_logit_difference": float(
            np.max(np.abs(native_logits - manual_logits), initial=0.0)
        ),
        "max_abs_probability_difference": float(
            np.max(np.abs(native_probabilities - manual_probabilities), initial=0.0)
        ),
    }
    evidence["gate_pass"] = bool(
        agreement == NATIVE_MANUAL_TOLERANCE["prediction_agreement"]
        and evidence["max_abs_logit_difference"]
        <= NATIVE_MANUAL_TOLERANCE["max_abs_logit_difference"]
        and evidence["max_abs_probability_difference"]
        <= NATIVE_MANUAL_TOLERANCE["max_abs_probability_difference"]
    )
    return evidence


def model_weights_sha256(model: tf.keras.Model) -> str:
    digest = hashlib.sha256()
    for variable in model.weights:
        value = np.asarray(variable.numpy())
        digest.update(str(getattr(variable, "path", variable.name)).encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def q_sha256(model: LearnedLocalResidualSlotLapGNN) -> str:
    q = np.asarray(model.learned_local_residual_slots.Q.numpy(), dtype=np.float32)
    return hashlib.sha256(q.reshape(-1).tobytes(order="C")).hexdigest()


def model_identity(model: LearnedLocalResidualSlotLapGNN) -> dict[str, Any]:
    q = model.learned_local_residual_slots.Q
    return {
        "class": type(model).__name__,
        "parameter_count": int(model.count_params()),
        "trainable_variable_count": len(model.trainable_variables),
        "q_index": EXPECTED_Q_INDEX,
        "q_is_index_127": model.trainable_variables[EXPECTED_Q_INDEX] is q,
        "q_shape": list(q.shape),
        "q_dtype": str(q.dtype),
        "q_flat_float32_sha256": q_sha256(model),
    }


def validate_model_identity(
    model: LearnedLocalResidualSlotLapGNN,
) -> dict[str, Any]:
    identity = model_identity(model)
    expected = {
        "class": EXPECTED_MODEL_CLASS,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "trainable_variable_count": EXPECTED_TRAINABLE_VARIABLE_COUNT,
        "q_index": EXPECTED_Q_INDEX,
        "q_is_index_127": True,
        "q_shape": list(EXPECTED_Q_SHAPE),
        "q_dtype": EXPECTED_Q_DTYPE,
        "q_flat_float32_sha256": EXPECTED_Q_SHA256,
    }
    drift = {
        key: {"actual": identity.get(key), "expected": value}
        for key, value in expected.items()
        if identity.get(key) != value
    }
    if drift:
        raise RemainingPriorProbeError(f"Candidate checkpoint identity drift: {drift}")
    if getattr(model, "optimizer", None) is not None:
        raise RemainingPriorProbeError("compile=False unexpectedly restored optimizer")
    return identity


def validate_source_artifacts(
    *,
    step12e_archive: str | Path,
    checkpoint: str | Path,
    checkpoint_weights: str | Path,
    checkpoint_metadata: str | Path,
    resolved_config: str | Path,
) -> dict[str, Any]:
    paths = {
        "step12e_archive": Path(step12e_archive),
        "checkpoint": Path(checkpoint),
        "checkpoint_weights": Path(checkpoint_weights),
        "checkpoint_metadata": Path(checkpoint_metadata),
        "resolved_config": Path(resolved_config),
        "candidate_model": CANDIDATE_MODEL_PATH,
    }
    expected = {
        "step12e_archive": EXPECTED_STEP12E_ARCHIVE_SHA256,
        "checkpoint": EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_weights": EXPECTED_WEIGHTS_SHA256,
        "checkpoint_metadata": EXPECTED_METADATA_SHA256,
        "resolved_config": EXPECTED_RESOLVED_CONFIG_SHA256,
        "candidate_model": EXPECTED_CANDIDATE_MODEL_SHA256,
    }
    actual = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual[name] = sha256_file(path)
        if actual[name] != expected[name]:
            raise RemainingPriorProbeError(f"Locked artifact SHA drift: {name}")
    return actual


def validate_checkpoint_metadata(metadata_path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RemainingPriorProbeError("Checkpoint metadata unreadable") from exc
    if not isinstance(payload, dict):
        raise RemainingPriorProbeError("Checkpoint metadata must be a JSON object")
    metrics = payload.get("validation_metrics")
    if payload.get("epoch") != EXPECTED_CHECKPOINT_EPOCH or not isinstance(metrics, dict):
        raise RemainingPriorProbeError("Checkpoint metadata epoch/metrics drift")
    for name, expected in P0_REFERENCE.items():
        actual = metrics.get(name)
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            raise RemainingPriorProbeError(f"Checkpoint reference metric drift: {name}")
        if float(actual) != expected:
            raise RemainingPriorProbeError(f"Checkpoint reference metric drift: {name}")
    return payload


def validate_frozen_runtime_config(
    raw_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use the reviewed Step-6 validator and assert its real nested identity."""

    config = step6.build_runtime_config(raw_config)
    contract = step6.validate_frozen_contract(config, package_root=FROZEN_PACKAGE_ROOT)
    if contract.get("scientific_payload_sha256") != EXPECTED_SCIENTIFIC_PAYLOAD_SHA256:
        raise RemainingPriorProbeError("Frozen scientific payload drift")
    if (
        contract.get("locked", {}).get("execution_contract_sha256")
        != EXPECTED_EXECUTION_CONTRACT_SHA256
    ):
        raise RemainingPriorProbeError("Frozen execution contract drift")
    return config, dict(contract)


def validate_metadata_config_provenance(
    raw_config: Mapping[str, Any], metadata: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind the exact raw config to checkpoint metadata via reviewed Step 6."""

    try:
        result = step6.validate_checkpoint_metadata(raw_config, metadata)
    except (KeyError, TypeError, ValueError, step6.PriorProbeError) as exc:
        raise RemainingPriorProbeError(
            "Checkpoint/resolved-config provenance mismatch"
        ) from exc
    return dict(result)


def load_fixed_checkpoint(checkpoint: str | Path) -> LearnedLocalResidualSlotLapGNN:
    model = tf.keras.models.load_model(
        Path(checkpoint),
        custom_objects={
            "LearnedLocalResidualSlotLapGNN": LearnedLocalResidualSlotLapGNN,
            "LearnedLocalResidualSlotPool": LearnedLocalResidualSlotPool,
            "fer2013_graph_research>LearnedLocalResidualSlotLapGNN": (
                LearnedLocalResidualSlotLapGNN
            ),
            "fer2013_graph_research>LearnedLocalResidualSlotPool": (
                LearnedLocalResidualSlotPool
            ),
        },
        compile=False,
    )
    validate_model_identity(model)
    return model


def paired_diagnostics(
    labels: np.ndarray, probabilities: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    p0_predictions = probabilities[P0].argmax(axis=1)
    p0_metrics = classification_metrics(labels, probabilities[P0])
    result = {}
    for condition in CONDITIONS[1:]:
        predictions = probabilities[condition].argmax(axis=1)
        metrics = classification_metrics(labels, probabilities[condition])
        p0_correct = p0_predictions == labels
        condition_correct = predictions == labels
        disagreement = predictions != p0_predictions
        result[condition] = {
            "prediction_disagreement_count": int(np.count_nonzero(disagreement)),
            "prediction_disagreement_rate": float(np.mean(disagreement)),
            "correctness_transitions": {
                "p0_correct_to_intervention_wrong": int(
                    np.count_nonzero(p0_correct & ~condition_correct)
                ),
                "p0_wrong_to_intervention_correct": int(
                    np.count_nonzero(~p0_correct & condition_correct)
                ),
                "unchanged_correct": int(np.count_nonzero(p0_correct & condition_correct)),
                "unchanged_wrong": int(np.count_nonzero(~p0_correct & ~condition_correct)),
            },
            "per_class_f1": list(metrics["per_class_f1"]),
            "p0_minus_condition_per_class_f1_delta": [
                float(left - right)
                for left, right in zip(
                    p0_metrics["per_class_f1"], metrics["per_class_f1"]
                )
            ],
        }
    return result


def _required_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    metrics = classification_metrics(labels, probabilities)
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "per_class_f1": list(metrics["per_class_f1"]),
    }


def evaluate_conditions(
    model: LearnedLocalResidualSlotLapGNN,
    batches: Iterable[Mapping[str, tf.Tensor]],
) -> dict[str, Any]:
    labels_parts: list[np.ndarray] = []
    sample_id_parts: list[np.ndarray] = []
    probability_parts = {condition: [] for condition in CONDITIONS}
    loss_sums = {condition: 0.0 for condition in CONDITIONS}
    integrity_counts = {condition: 0 for condition in CONDITIONS}
    equivalence_batches = []
    batch_count = 0

    for batch in batches:
        step6.validate_batch_schema(batch)
        snapshot = snapshot_batch(batch)
        labels = np.asarray(batch["labels"].numpy(), dtype=np.int64)
        sample_ids = np.asarray(batch["sample_ids"].numpy(), dtype=np.int64)
        native = model(batch, training=False)
        p0_output, p0_trace = manual_forward(model, batch, P0)
        validate_pathway_integrity(model, batch, snapshot, P0, p0_trace)
        equivalence = native_manual_equivalence(native, p0_output, batch["sample_ids"])
        equivalence["batch_index"] = batch_count
        equivalence_batches.append(equivalence)
        if not equivalence["gate_pass"]:
            raise ManualForwardEquivalenceError(
                {
                    "status": "INVALID_MANUAL_FORWARD_EQUIVALENCE",
                    "tolerances": NATIVE_MANUAL_TOLERANCE,
                    "batches": equivalence_batches,
                }
            )
        labels_parts.append(labels)
        sample_id_parts.append(sample_ids)
        for condition in CONDITIONS:
            if condition == P0:
                output, trace = p0_output, p0_trace
            else:
                output, trace = manual_forward(model, batch, condition)
            validate_pathway_integrity(model, batch, snapshot, condition, trace)
            probabilities = np.asarray(output["probabilities"].numpy(), dtype=np.float64)
            if probabilities.shape != (labels.size, 7):
                raise RemainingPriorProbeError(
                    f"Condition probability shape drift: {condition}/{probabilities.shape}"
                )
            probability_parts[condition].append(probabilities)
            batch_loss = float(sparse_cross_entropy(batch["labels"], output["logits"]).numpy())
            loss_sums[condition] += batch_loss * labels.size
            integrity_counts[condition] += 1
            assert_source_batch_unchanged(batch, snapshot)
        batch_count += 1

    if batch_count == 0:
        raise RemainingPriorProbeError("No validation batches produced")
    labels = np.concatenate(labels_parts)
    sample_ids = np.concatenate(sample_id_parts)
    if len(np.unique(sample_ids)) != len(sample_ids):
        raise RemainingPriorProbeError("Validation sample IDs are not unique")
    probabilities = {
        condition: np.concatenate(parts, axis=0)
        for condition, parts in probability_parts.items()
    }
    metrics = {}
    for condition in CONDITIONS:
        metrics[condition] = _required_metrics(labels, probabilities[condition])
        metrics[condition]["loss"] = loss_sums[condition] / labels.size
    equivalence_summary = {
        "status": "PASS",
        "tolerances": dict(NATIVE_MANUAL_TOLERANCE),
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
        "paired_diagnostics": paired_diagnostics(labels, probabilities),
        "integrity_counts": integrity_counts,
        "native_manual_equivalence": equivalence_summary,
    }


def classify_sensitivity(delta_macro_pp: float) -> str:
    if delta_macro_pp >= 10.0:
        return "HIGH_REMAINING_PRIOR_PATH_SENSITIVITY"
    if delta_macro_pp >= 5.0:
        return "MODERATE_REMAINING_PRIOR_PATH_SENSITIVITY"
    return "LOW_REMAINING_PRIOR_PATH_SENSITIVITY"


def overall_decision(individual_labels: Mapping[str, str], p9_delta: float) -> str:
    if set(individual_labels) != set(INDIVIDUAL_CONDITIONS):
        raise RemainingPriorProbeError("Overall decision requires exact P1-P8 inventory")
    high_count = sum(
        label == "HIGH_REMAINING_PRIOR_PATH_SENSITIVITY"
        for label in individual_labels.values()
    )
    if high_count >= 2:
        return "MULTIPLE_HIGH_REMAINING_PRIOR_PATHS"
    if high_count == 1:
        return "SINGLE_HIGH_REMAINING_PRIOR_PATH"
    if p9_delta >= 10.0:
        return "DISTRIBUTED_OR_INTERACTION_REMAINING_PRIOR_DEPENDENCY"
    return "LOW_REMAINING_EXPLICIT_PRIOR_SENSITIVITY"


def evaluate_registered_gates(
    result: Mapping[str, Any], *, bounded_limit: int | None
) -> dict[str, Any]:
    gate_a = dict(result["native_manual_equivalence"])
    gate_a["pass"] = gate_a.get("status") == "PASS"
    if bounded_limit is not None:
        return {
            "status": "BOUNDED_SMOKE_NO_SCIENTIFIC_INTERPRETATION",
            "gate_a_native_vs_p0": gate_a,
            "gate_b_checkpoint_metrics": {"status": "NOT_EVALUATED_BOUNDED_SMOKE"},
            "gate_c_checkpoint_identity": {"status": "PRE_EVALUATION_ONLY"},
            "per_condition_sensitivity": None,
            "overall_decision": None,
        }
    checks = {
        name: {
            "observed": float(result["metrics"][P0][name]),
            "reference": reference,
            "absolute_difference": abs(float(result["metrics"][P0][name]) - reference),
            "tolerance": REFERENCE_TOLERANCE[name],
        }
        for name, reference in P0_REFERENCE.items()
    }
    gate_b_pass = result["sample_count"] == EXPECTED_FULL_VALIDATION_SAMPLES and all(
        item["absolute_difference"] <= item["tolerance"] for item in checks.values()
    )
    gate_b = {
        "status": "PASS" if gate_b_pass else "FAIL",
        "sample_count": result["sample_count"],
        "required_sample_count": EXPECTED_FULL_VALIDATION_SAMPLES,
        "checks": checks,
    }
    if not gate_a["pass"] or not gate_b_pass:
        return {
            "status": (
                "INVALID_MANUAL_FORWARD_EQUIVALENCE"
                if not gate_a["pass"]
                else "INVALID_P0_REFERENCE_REPRODUCTION"
            ),
            "gate_a_native_vs_p0": gate_a,
            "gate_b_checkpoint_metrics": gate_b,
            "per_condition_sensitivity": None,
            "overall_decision": None,
        }
    p0_macro = float(result["metrics"][P0]["macro_f1"])
    sensitivity = {}
    labels = {}
    for condition in INDIVIDUAL_CONDITIONS:
        delta = 100.0 * (p0_macro - float(result["metrics"][condition]["macro_f1"]))
        label = classify_sensitivity(delta)
        labels[condition] = label
        sensitivity[condition] = {
            "delta_macro_pp": delta,
            "label": label,
            "negative_delta_retained_as_low": delta < 0.0,
        }
    p9_delta = 100.0 * (p0_macro - float(result["metrics"][P9]["macro_f1"]))
    return {
        "status": "VALID_REGISTERED_REMAINING_PRIOR_DECOMPOSITION",
        "gate_a_native_vs_p0": gate_a,
        "gate_b_checkpoint_metrics": gate_b,
        "per_condition_sensitivity": sensitivity,
        "p9_joint_anchor": {
            "delta_macro_pp": p9_delta,
            "at_least_10_pp": p9_delta >= 10.0,
        },
        "overall_decision": overall_decision(labels, p9_delta),
        "non_additivity_warning": (
            "P1-P8 sensitivities are nonlinear and must not be summed, divided by P9, "
            "or reported as percentage contributions."
        ),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_ready(dict(payload)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_probe_outputs(
    output_root: Path,
    *,
    result: Mapping[str, Any],
    gates: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    identity_before: Mapping[str, Any],
    identity_after: Mapping[str, Any],
    model_weights_before: str,
    model_weights_after: str,
    bounded_limit: int | None,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    metrics_path = output_root / "condition_metrics.json"
    paired_path = output_root / "paired_diagnostics.json"
    integrity_path = output_root / "intervention_integrity.json"
    _write_json(metrics_path, {"condition_order": CONDITIONS, "metrics": result["metrics"]})
    _write_json(paired_path, result["paired_diagnostics"])
    _write_json(
        integrity_path,
        {
            "condition_order": CONDITIONS,
            "specs": INTERVENTION_SPECS,
            "checks_per_condition": result["integrity_counts"],
            "source_batch_mutated": False,
            "topology_rebuilt": False,
            "training_or_optimizer_update": False,
            "test_access": False,
            "model_identity_before": identity_before,
            "model_identity_after": identity_after,
            "model_weights_sha256_before": model_weights_before,
            "model_weights_sha256_after": model_weights_after,
        },
    )
    manifest = {
        "schema_version": 1,
        "tool_version": TOOL_VERSION,
        "issue": ISSUE_NUMBER,
        "base_commit": EXPECTED_BASE_COMMIT,
        "status": (
            "BOUNDED_SMOKE_NO_SCIENTIFIC_INTERPRETATION"
            if bounded_limit is not None
            else gates["status"]
        ),
        "validation_only": True,
        "inference_only": True,
        "condition_order": CONDITIONS,
        "source_hashes": dict(source_hashes),
        "scientific_payload_sha256": EXPECTED_SCIENTIFIC_PAYLOAD_SHA256,
        "execution_contract_sha256": EXPECTED_EXECUTION_CONTRACT_SHA256,
        "sample_count": result["sample_count"],
        "batch_count": result["batch_count"],
        "limit_val_batches": bounded_limit,
        "registered_gates_and_decision": gates,
        "scientific_interpretation": None if bounded_limit is not None else gates.get("overall_decision"),
        "training": False,
        "optimizer_updates": False,
        "test_access": False,
        "topology_rebuilt": False,
        "artifacts": {
            path.name: sha256_file(path)
            for path in (metrics_path, paired_path, integrity_path)
        },
    }
    _write_json(output_root / "probe_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed-checkpoint P0-P9 remaining-prior paths on validation only."
    )
    parser.add_argument("--step12e-archive", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-weights", required=True, type=Path)
    parser.add_argument("--checkpoint-metadata", required=True, type=Path)
    parser.add_argument("--resolved-config", required=True, type=Path)
    parser.add_argument("--prior-root", required=True, type=Path)
    parser.add_argument("--clean-graph-cache-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--graph-workers", type=int, default=2)
    parser.add_argument("--graph-cache-size", type=int, default=64)
    parser.add_argument("--limit-val-batches", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (
            args.eval_batch_size,
            args.graph_workers,
            args.graph_cache_size,
        )
    ):
        raise RemainingPriorProbeError("Resource arguments must be positive integers")
    if args.limit_val_batches is not None and args.limit_val_batches <= 0:
        raise RemainingPriorProbeError("limit_val_batches must be positive")
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    source_hashes = validate_source_artifacts(
        step12e_archive=args.step12e_archive,
        checkpoint=args.checkpoint,
        checkpoint_weights=args.checkpoint_weights,
        checkpoint_metadata=args.checkpoint_metadata,
        resolved_config=args.resolved_config,
    )
    metadata = validate_checkpoint_metadata(args.checkpoint_metadata)
    if scientific_payload_checksum(FROZEN_PACKAGE_ROOT) != EXPECTED_SCIENTIFIC_PAYLOAD_SHA256:
        raise RemainingPriorProbeError("Frozen scientific payload drift")
    raw_config = step6.load_persisted_resolved_config(args.resolved_config)
    config, frozen_contract = validate_frozen_runtime_config(raw_config)
    metadata_config_cross_check = validate_metadata_config_provenance(
        raw_config, metadata
    )
    model = load_fixed_checkpoint(args.checkpoint)
    identity_before = validate_model_identity(model)
    model_weights_before = model_weights_sha256(model)
    validation_data = GraphBatchGenerator(
        prior_root=args.prior_root,
        split="val",
        config=config,
        batch_size=args.eval_batch_size,
        seed=int(config["seed"]),
        shuffle=False,
        graph_cache_size=args.graph_cache_size,
        graph_workers=args.graph_workers,
        clean_graph_cache_dir=args.clean_graph_cache_dir,
    )
    result = evaluate_conditions(
        model,
        validation_data.iter_epoch(0, limit_batches=args.limit_val_batches),
    )
    identity_after = validate_model_identity(model)
    model_weights_after = model_weights_sha256(model)
    if identity_after != identity_before or model_weights_after != model_weights_before:
        raise RemainingPriorProbeError("Model/Q/weights changed during inference")
    if sha256_file(args.checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise RemainingPriorProbeError("Checkpoint changed during inference")
    source_hashes_after = validate_source_artifacts(
        step12e_archive=args.step12e_archive,
        checkpoint=args.checkpoint,
        checkpoint_weights=args.checkpoint_weights,
        checkpoint_metadata=args.checkpoint_metadata,
        resolved_config=args.resolved_config,
    )
    if source_hashes_after != source_hashes:
        raise RemainingPriorProbeError("Locked source artifacts changed during inference")
    gates = evaluate_registered_gates(result, bounded_limit=args.limit_val_batches)
    gates["gate_c_checkpoint_identity"] = {
        "status": "PASS",
        "source_hashes_before": source_hashes,
        "source_hashes_after": source_hashes_after,
        "frozen_contract": frozen_contract,
        "metadata_config_cross_check": metadata_config_cross_check,
        "resolved_config_sha256": source_hashes["resolved_config"],
        "model_identity_before": identity_before,
        "model_identity_after": identity_after,
        "model_weights_sha256_before": model_weights_before,
        "model_weights_sha256_after": model_weights_after,
        "checkpoint_unchanged": True,
        "model_q_weights_unchanged": True,
    }
    write_probe_outputs(
        args.output_root,
        result=result,
        gates=gates,
        source_hashes=source_hashes,
        identity_before=identity_before,
        identity_after=identity_after,
        model_weights_before=model_weights_before,
        model_weights_after=model_weights_after,
        bounded_limit=args.limit_val_batches,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
