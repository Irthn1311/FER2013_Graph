from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "standalone/lap_gnn_tensorflow_ofix7_mid_candidate"
PACKAGE_SRC = PACKAGE_ROOT / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from lap_gnn_tf.conversion import load_pytorch_npz  # noqa: E402
from lap_gnn_tf.graph.batch import load_golden_batch  # noqa: E402
from research.candidates.tf_learned_local_residual_slots.model import (  # noqa: E402
    build_candidate_model,
)


PROBE_PATH = (
    ROOT
    / "research/candidates/tf_learned_local_residual_slots/evaluate_remaining_prior_probe.py"
)
GOLDEN = PACKAGE_ROOT / "validation_assets/golden"
BASE = "e9b4deec2d4986b4a94fce32f3c1586cdb301047"


def _load_probe():
    spec = importlib.util.spec_from_file_location("issue38_remaining_prior_probe", PROBE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_probe()


@pytest.fixture(scope="module")
def candidate_and_batch():
    batch = load_golden_batch(str(GOLDEN / "graph_batch.npz"))
    candidate = build_candidate_model(batch)
    load_pytorch_npz(candidate, GOLDEN / "model_state.npz")
    q = tf.reshape(tf.linspace(-0.04, 0.04, 4 * 96), (4, 96))
    candidate.learned_local_residual_slots.Q.assign(q)
    return candidate, batch


def _arrays_equal(left, right):
    np.testing.assert_array_equal(np.asarray(left.numpy()), np.asarray(right.numpy()))


def _changed_columns(actual, original):
    changed = np.any(np.asarray(actual.numpy()) != np.asarray(original.numpy()), axis=0)
    return set(np.flatnonzero(changed).tolist())


def test_exact_base_source_checkpoint_q_and_protocol_locks():
    assert probe.EXPECTED_BASE_COMMIT == BASE
    assert probe.EXPECTED_CANDIDATE_MODEL_SHA256 == (
        "0a7cfa315baf0d6666b7ab86139b328ab6d138985cfddd71180410675602fcca"
    )
    assert probe.EXPECTED_STEP12E_ARCHIVE_SHA256 == (
        "f436b0a7a20c751b2fd2f47738469fb409ecf9a1a40628e05d20974639927451"
    )
    assert probe.EXPECTED_CHECKPOINT_SHA256 == (
        "e0d633cb6200e963f31a28750e28c7febdaae40344c90ba9d94b826a09e4b78c"
    )
    assert probe.EXPECTED_WEIGHTS_SHA256 == (
        "a18a372f70ce56868ae43257e9b7fa5e20517499c2c1e35c48dba4d65eaaaa74"
    )
    assert probe.EXPECTED_METADATA_SHA256 == (
        "a5ee759bc6fbef587e025199d0dcfe6ebd3a1764cffa567f793c53e972eb47cf"
    )
    assert probe.EXPECTED_Q_SHA256 == (
        "54b368aa183c65d5843d8b8e340d3020412d1a2dfeaabbe8b2c0166684ab3ff9"
    )
    assert probe.EXPECTED_CHECKPOINT_EPOCH == 42
    assert probe.EXPECTED_SCIENTIFIC_PAYLOAD_SHA256 == (
        "286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e"
    )
    assert probe.EXPECTED_EXECUTION_CONTRACT_SHA256 == (
        "14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22"
    )
    assert probe.sha256_file(probe.CANDIDATE_MODEL_PATH) == (
        probe.EXPECTED_CANDIDATE_MODEL_SHA256
    )


@pytest.mark.parametrize(
    "tampered_name",
    ["step12e_archive", "checkpoint", "checkpoint_weights", "checkpoint_metadata"],
)
def test_artifact_locks_pass_and_each_wrong_sha_fails_closed(
    monkeypatch, tmp_path, tampered_name
):
    paths = {}
    mapping = {
        "step12e_archive": "EXPECTED_STEP12E_ARCHIVE_SHA256",
        "checkpoint": "EXPECTED_CHECKPOINT_SHA256",
        "checkpoint_weights": "EXPECTED_WEIGHTS_SHA256",
        "checkpoint_metadata": "EXPECTED_METADATA_SHA256",
    }
    for name, constant in mapping.items():
        path = tmp_path / name
        path.write_bytes(name.encode())
        paths[name] = path
        monkeypatch.setattr(probe, constant, probe.sha256_file(path))
    candidate = tmp_path / "model.py"
    candidate.write_bytes(b"candidate")
    monkeypatch.setattr(probe, "CANDIDATE_MODEL_PATH", candidate)
    monkeypatch.setattr(probe, "EXPECTED_CANDIDATE_MODEL_SHA256", probe.sha256_file(candidate))
    assert set(probe.validate_source_artifacts(**paths)) == {
        *mapping,
        "candidate_model",
    }
    paths[tampered_name].write_bytes(b"tampered")
    with pytest.raises(probe.RemainingPriorProbeError, match=tampered_name):
        probe.validate_source_artifacts(**paths)


@pytest.mark.parametrize("nested_value", ["wrong", None])
def test_execution_contract_uses_real_nested_step6_shape_and_fails_closed(
    monkeypatch, nested_value
):
    returned = {
        "scientific_payload_sha256": probe.EXPECTED_SCIENTIFIC_PAYLOAD_SHA256,
        "locked": {"execution_contract_sha256": nested_value},
    }
    monkeypatch.setattr(probe.step6, "build_runtime_config", lambda raw: dict(raw))
    monkeypatch.setattr(
        probe.step6,
        "validate_frozen_contract",
        lambda config, package_root: returned,
    )
    with pytest.raises(probe.RemainingPriorProbeError, match="execution contract"):
        probe.validate_frozen_runtime_config({"seed": 42})


def test_execution_contract_valid_nested_step6_shape_passes(monkeypatch):
    returned = {
        "scientific_payload_sha256": probe.EXPECTED_SCIENTIFIC_PAYLOAD_SHA256,
        "locked": {
            "execution_contract_sha256": probe.EXPECTED_EXECUTION_CONTRACT_SHA256
        },
    }
    monkeypatch.setattr(probe.step6, "build_runtime_config", lambda raw: dict(raw))
    monkeypatch.setattr(
        probe.step6,
        "validate_frozen_contract",
        lambda config, package_root: returned,
    )
    config, contract = probe.validate_frozen_runtime_config({"seed": 42})
    assert config == {"seed": 42}
    assert contract == returned


def test_metadata_lock_requires_epoch42_and_exact_reference(tmp_path):
    metadata = {
        "epoch": 42,
        "validation_metrics": dict(probe.P0_REFERENCE),
    }
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    assert probe.validate_checkpoint_metadata(path)["epoch"] == 42
    metadata["validation_metrics"]["macro_f1"] += 1e-12
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(probe.RemainingPriorProbeError, match="macro_f1"):
        probe.validate_checkpoint_metadata(path)


def test_candidate_class_params_128_variables_and_q_identity(candidate_and_batch, monkeypatch):
    candidate, _batch = candidate_and_batch
    expected_q = probe.q_sha256(candidate)
    monkeypatch.setattr(probe, "EXPECTED_Q_SHA256", expected_q)
    identity = probe.validate_model_identity(candidate)
    assert identity == {
        "class": "LearnedLocalResidualSlotLapGNN",
        "parameter_count": 1_061_576,
        "trainable_variable_count": 128,
        "q_index": 127,
        "q_is_index_127": True,
        "q_shape": [4, 96],
        "q_dtype": "float32",
        "q_flat_float32_sha256": expected_q,
    }
    monkeypatch.setattr(probe, "EXPECTED_Q_SHA256", "0" * 64)
    with pytest.raises(probe.RemainingPriorProbeError, match="identity drift"):
        probe.validate_model_identity(candidate)


def test_candidate_deserializes_compile_false_with_exact_identity(
    candidate_and_batch, monkeypatch, tmp_path
):
    candidate, _batch = candidate_and_batch
    checkpoint = tmp_path / "candidate.keras"
    candidate.save(checkpoint)
    monkeypatch.setattr(probe, "EXPECTED_Q_SHA256", probe.q_sha256(candidate))
    restored = probe.load_fixed_checkpoint(checkpoint)
    identity = probe.validate_model_identity(restored)
    assert identity["class"] == "LearnedLocalResidualSlotLapGNN"
    assert identity["parameter_count"] == 1_061_576
    assert identity["trainable_variable_count"] == 128
    assert getattr(restored, "optimizer", None) is None


def test_p0_manual_native_equivalence_gate(candidate_and_batch):
    candidate, batch = candidate_and_batch
    native = candidate(batch, training=False)
    manual, trace = probe.manual_forward(candidate, batch, probe.P0)
    integrity = probe.validate_pathway_integrity(
        candidate, batch, probe.snapshot_batch(batch), probe.P0, trace
    )
    gate = probe.native_manual_equivalence(native, manual, batch["sample_ids"])
    assert integrity["changed_paths"] == []
    assert gate["gate_pass"] is True
    assert gate["prediction_agreement"] == 1.0
    assert gate["max_abs_logit_difference"] <= 1e-5
    assert gate["max_abs_probability_difference"] <= 3e-6


@pytest.mark.parametrize(
    "condition,expected_node_columns,expected_edge_columns",
    [
        (probe.P1, {5}, set()),
        (probe.P2, set(range(6, 19)), set()),
        (probe.P3, set(range(19, 31)), set()),
        (probe.P4, {31}, set()),
        (probe.P5, set(), {6, 7}),
    ],
)
def test_p1_p5_change_only_registered_feature_columns_and_keep_direct_prior(
    candidate_and_batch, condition, expected_node_columns, expected_edge_columns
):
    candidate, batch = candidate_and_batch
    snapshot = probe.snapshot_batch(batch)
    _output, trace = probe.manual_forward(candidate, batch, condition)
    probe.validate_pathway_integrity(candidate, batch, snapshot, condition, trace)
    assert _changed_columns(
        trace["effective_node_features"], trace["official_node_features"]
    ) == expected_node_columns
    assert _changed_columns(
        trace["effective_edge_features"], trace["official_edge_features"]
    ) == expected_edge_columns
    _arrays_equal(trace["context_part_soft"], trace["official_part_soft"])
    _arrays_equal(trace["part_pool_part_soft"], trace["official_part_soft"])
    _arrays_equal(trace["readout_part_soft"], trace["official_part_soft"])


def test_p6_context_only_and_downstream_official_prior_restored(candidate_and_batch):
    candidate, batch = candidate_and_batch
    _output, trace = probe.manual_forward(candidate, batch, probe.P6)
    probe.validate_pathway_integrity(
        candidate, batch, probe.snapshot_batch(batch), probe.P6, trace
    )
    _arrays_equal(trace["context_part_soft"], tf.zeros_like(trace["official_part_soft"]))
    _arrays_equal(trace["part_pool_part_soft"], trace["official_part_soft"])
    _arrays_equal(trace["readout_part_soft"], trace["official_part_soft"])


def test_p7_changes_only_readout_part_soft_after_official_upstream(candidate_and_batch):
    candidate, batch = candidate_and_batch
    p0, p0_trace = probe.manual_forward(candidate, batch, probe.P0)
    p7, p7_trace = probe.manual_forward(candidate, batch, probe.P7)
    _arrays_equal(p7_trace["context_part_soft"], p0_trace["context_part_soft"])
    _arrays_equal(p7_trace["effective_node_features"], p0_trace["effective_node_features"])
    _arrays_equal(p7_trace["effective_edge_features"], p0_trace["effective_edge_features"])
    _arrays_equal(p7["node_embeddings"], p0["node_embeddings"])
    _arrays_equal(p7["learned_local_residual_slots"], p0["learned_local_residual_slots"])
    _arrays_equal(p7_trace["readout_part_soft"], tf.zeros_like(p0_trace["readout_part_soft"]))


def test_p8_changes_only_readout_validity(candidate_and_batch):
    candidate, batch = candidate_and_batch
    p0, p0_trace = probe.manual_forward(candidate, batch, probe.P0)
    p8, p8_trace = probe.manual_forward(candidate, batch, probe.P8)
    _arrays_equal(p8_trace["readout_part_soft"], p0_trace["readout_part_soft"])
    _arrays_equal(p8["node_embeddings"], p0["node_embeddings"])
    _arrays_equal(p8["learned_local_residual_slots"], p0["learned_local_residual_slots"])
    for name in probe.LOCAL_PARTS:
        _arrays_equal(
            p8_trace["readout_valid_groups"][name],
            tf.zeros_like(p0_trace["official_readout_valid_groups"][name], dtype=tf.bool),
        )
    _arrays_equal(
        p8_trace["readout_valid_groups"]["global"],
        tf.ones_like(p0_trace["official_readout_valid_groups"]["global"], dtype=tf.bool),
    )


def test_p9_changes_exact_joint_paths_and_preserves_visual_topology(candidate_and_batch):
    candidate, batch = candidate_and_batch
    snapshot = probe.snapshot_batch(batch)
    _output, trace = probe.manual_forward(candidate, batch, probe.P9)
    probe.validate_pathway_integrity(candidate, batch, snapshot, probe.P9, trace)
    assert _changed_columns(
        trace["effective_node_features"], trace["official_node_features"]
    ) == set(range(5, 32))
    assert _changed_columns(
        trace["effective_edge_features"], trace["official_edge_features"]
    ) == {6, 7}
    _arrays_equal(trace["effective_node_features"][:, 0:5], batch["node_features"][:, 0:5])
    _arrays_equal(trace["effective_node_features"][:, 32:37], batch["node_features"][:, 32:37])
    _arrays_equal(trace["effective_edge_features"][:, 0:6], batch["edge_features"][:, 0:6])
    _arrays_equal(trace["context_part_soft"], tf.zeros_like(trace["official_part_soft"]))
    _arrays_equal(trace["readout_part_soft"], tf.zeros_like(trace["official_part_soft"]))
    _arrays_equal(trace["part_pool_part_soft"], trace["official_part_soft"])


def test_source_batch_topology_q_and_weights_immutable_across_all_conditions(
    candidate_and_batch
):
    candidate, batch = candidate_and_batch
    snapshot = probe.snapshot_batch(batch)
    q_before = np.array(candidate.learned_local_residual_slots.Q.numpy(), copy=True)
    weights_before = probe.model_weights_sha256(candidate)
    for condition in probe.CONDITIONS:
        _output, trace = probe.manual_forward(candidate, batch, condition)
        evidence = probe.validate_pathway_integrity(
            candidate, batch, snapshot, condition, trace
        )
        assert evidence["topology_unchanged"] is True
        for name in probe.TOPOLOGY_FIELDS:
            _arrays_equal(trace["topology"][name], batch[name])
    probe.assert_source_batch_unchanged(batch, snapshot)
    np.testing.assert_array_equal(candidate.learned_local_residual_slots.Q.numpy(), q_before)
    assert probe.model_weights_sha256(candidate) == weights_before


def test_paired_diagnostics_and_transition_accounting_exact():
    labels = np.array([0, 1, 2, 0], dtype=np.int64)
    p0_predictions = [0, 0, 2, 1]
    intervention_predictions = [1, 1, 2, 1]

    def probabilities(predictions):
        result = np.full((len(predictions), 7), 0.01, dtype=np.float64)
        for row, prediction in enumerate(predictions):
            result[row, prediction] = 0.94
        return result / result.sum(axis=1, keepdims=True)

    values = {condition: probabilities(p0_predictions) for condition in probe.CONDITIONS}
    values[probe.P1] = probabilities(intervention_predictions)
    paired = probe.paired_diagnostics(labels, values)[probe.P1]
    assert paired["prediction_disagreement_count"] == 2
    assert paired["prediction_disagreement_rate"] == 0.5
    assert paired["correctness_transitions"] == {
        "p0_correct_to_intervention_wrong": 1,
        "p0_wrong_to_intervention_correct": 1,
        "unchanged_correct": 1,
        "unchanged_wrong": 1,
    }
    assert len(paired["per_class_f1"]) == 7
    assert len(paired["p0_minus_condition_per_class_f1_delta"]) == 7


@pytest.mark.parametrize(
    "delta,expected",
    [
        (-2.0, "LOW_REMAINING_PRIOR_PATH_SENSITIVITY"),
        (4.999999, "LOW_REMAINING_PRIOR_PATH_SENSITIVITY"),
        (5.0, "MODERATE_REMAINING_PRIOR_PATH_SENSITIVITY"),
        (9.999999, "MODERATE_REMAINING_PRIOR_PATH_SENSITIVITY"),
        (10.0, "HIGH_REMAINING_PRIOR_PATH_SENSITIVITY"),
    ],
)
def test_sensitivity_thresholds_and_negative_delta(delta, expected):
    assert probe.classify_sensitivity(delta) == expected


def _labels(high_count):
    values = {
        condition: "LOW_REMAINING_PRIOR_PATH_SENSITIVITY"
        for condition in probe.INDIVIDUAL_CONDITIONS
    }
    for condition in probe.INDIVIDUAL_CONDITIONS[:high_count]:
        values[condition] = "HIGH_REMAINING_PRIOR_PATH_SENSITIVITY"
    return values


@pytest.mark.parametrize(
    "high_count,p9_delta,expected",
    [
        (0, 9.999, "LOW_REMAINING_EXPLICIT_PRIOR_SENSITIVITY"),
        (0, 10.0, "DISTRIBUTED_OR_INTERACTION_REMAINING_PRIOR_DEPENDENCY"),
        (1, 20.0, "SINGLE_HIGH_REMAINING_PRIOR_PATH"),
        (2, 0.0, "MULTIPLE_HIGH_REMAINING_PRIOR_PATHS"),
        (8, 20.0, "MULTIPLE_HIGH_REMAINING_PRIOR_PATHS"),
    ],
)
def test_overall_decision_branches(high_count, p9_delta, expected):
    assert probe.overall_decision(_labels(high_count), p9_delta) == expected


def test_registered_output_has_no_additive_or_percentage_contribution_fields():
    p0_macro = probe.P0_REFERENCE["macro_f1"]
    result = {
        "sample_count": 3589,
        "native_manual_equivalence": {"status": "PASS"},
        "metrics": {
            condition: {
                "accuracy": probe.P0_REFERENCE["accuracy"],
                "macro_f1": p0_macro - (0.11 if condition == probe.P1 else 0.01),
                "loss": probe.P0_REFERENCE["loss"],
            }
            for condition in probe.CONDITIONS
        },
    }
    result["metrics"][probe.P0] = dict(probe.P0_REFERENCE)
    gates = probe.evaluate_registered_gates(result, bounded_limit=None)
    encoded = json.dumps(gates).lower()
    assert gates["status"] == "VALID_REGISTERED_REMAINING_PRIOR_DECOMPOSITION"
    assert gates["overall_decision"] == "SINGLE_HIGH_REMAINING_PRIOR_PATH"
    assert "percentage_contribution" not in encoded
    assert "additive_contribution" not in encoded
    assert "sum_p" not in encoded


def test_gate_b_fails_closed_on_wrong_sample_or_metric():
    result = {
        "sample_count": 1,
        "native_manual_equivalence": {"status": "PASS"},
        "metrics": {condition: dict(probe.P0_REFERENCE) for condition in probe.CONDITIONS},
    }
    gates = probe.evaluate_registered_gates(result, bounded_limit=None)
    assert gates["status"] == "INVALID_P0_REFERENCE_REPRODUCTION"
    assert gates["per_condition_sensitivity"] is None
    assert gates["overall_decision"] is None


def test_no_optimizer_training_test_split_or_graph_rebuild_callable():
    source = PROBE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    call_names = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute):
            call_names.append(node.func.attr)
        elif isinstance(node.func, ast.Name):
            call_names.append(node.func.id)
    for forbidden in ("fit", "train_on_batch", "GradientTape", "apply_gradients", "minimize"):
        assert forbidden not in call_names
    main_source = inspect.getsource(probe.main)
    assert 'split="val"' in main_source
    assert 'split="test"' not in source
    assert "test.csv" not in source.lower()
    assert "collate_d16_graphs" not in source
    assert "build_graph" not in source


def test_fixed_condition_order_and_specs_are_exact():
    assert probe.CONDITIONS == (
        probe.P0,
        probe.P1,
        probe.P2,
        probe.P3,
        probe.P4,
        probe.P5,
        probe.P6,
        probe.P7,
        probe.P8,
        probe.P9,
    )
    assert set(probe.INTERVENTION_SPECS) == set(probe.CONDITIONS)
    assert probe.INDIVIDUAL_CONDITIONS == probe.CONDITIONS[1:9]


def test_frozen_package_diff_empty_payload_unchanged_and_diff_check_passes():
    assert subprocess.run(
        ["git", "cat-file", "-e", f"{BASE}^{{commit}}"], cwd=ROOT, check=False
    ).returncode == 0
    frozen = PACKAGE_ROOT.relative_to(ROOT).as_posix()
    changed = subprocess.run(
        ["git", "diff", "--name-only", BASE, "--", frozen],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    assert changed == ""
    assert probe.scientific_payload_checksum(PACKAGE_ROOT) == (
        probe.EXPECTED_SCIENTIFIC_PAYLOAD_SHA256
    )
    assert subprocess.run(["git", "diff", "--check"], cwd=ROOT).returncode == 0
