from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from _helpers import loaded


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = (
    PACKAGE_ROOT
    / "tools"
    / "evaluate_fixed_checkpoint_local_residual_slot_decomposition_probe.py"
)
SPEC = importlib.util.spec_from_file_location("local_residual_slot_probe", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


@pytest.fixture(scope="module")
def golden_runs():
    model, batch = loaded()
    before_batch = probe._snapshot_batch(batch)
    before_weights = probe.step6.model_weights_sha256(model)
    native = model(batch, training=False)
    step7_d0 = probe.step7.manual_forward(model, batch, probe.step7.CONDITION_D0)
    step7_d3 = probe.step7.manual_forward(model, batch, probe.step7.CONDITION_D3)
    runs = {
        condition: probe.manual_forward(model, batch, condition)
        for condition in probe.CONDITIONS
    }
    after_weights = probe.step6.model_weights_sha256(model)
    return {
        "model": model,
        "batch": batch,
        "before_batch": before_batch,
        "before_weights": before_weights,
        "after_weights": after_weights,
        "native": native,
        "step7_d0": step7_d0,
        "step7_d3": step7_d3,
        "runs": runs,
    }


def _assert_tensor_equal(actual, expected):
    np.testing.assert_array_equal(actual.numpy(), expected.numpy())


def test_exact_identities_condition_order_and_registered_contract():
    assert probe.EXPECTED_IMPLEMENTATION_BASE == (
        "cd6a6b751d52729f7330adad58d94fbe7d1a7ac4"
    )
    assert probe.EXPECTED_SCIENTIFIC_PAYLOAD_SHA256 == (
        "286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e"
    )
    assert probe.EXPECTED_EXECUTION_CONTRACT_SHA256 == (
        "14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22"
    )
    assert probe.EXPECTED_STEP7_TOOL_SHA256 == (
        "c0b1df778e469665dd6437c58831d29dcc34fbde44231db75894c5469a1ade78"
    )
    assert probe.EXPECTED_STEP6_SUPPORT_SHA256 == (
        "3a033c977a29e102cfed75282ae7c1062f41feac8bef1b955ae425ec7e4004b3"
    )
    assert probe.sha256_file(probe.STEP7_TOOL_PATH) == probe.EXPECTED_STEP7_TOOL_SHA256
    assert probe.sha256_file(probe.STEP6_SUPPORT_PATH) == probe.EXPECTED_STEP6_SUPPORT_SHA256
    assert probe.CONDITIONS == (
        "official_manual_forward",
        "mouth_local_residual_zero",
        "eye_local_residual_zero",
        "brow_local_residual_zero",
        "nose_cheek_local_residual_zero",
        "all_local_residuals_zero_anchor",
    )
    assert probe.LOCAL_PARTS == ("mouth", "eye", "brow", "nose_cheek")
    assert probe.INTERVENTION_SPECS[probe.CONDITION_S0]["zeroed_local_slots"] == []
    for condition, slot in probe.SLOT_BY_CONDITION.items():
        assert probe.INTERVENTION_SPECS[condition]["zeroed_local_slots"] == [slot]
        assert probe.INTERVENTION_SPECS[condition]["changed_pathway_arguments"] == [
            f"readout.part_embeddings.{slot}"
        ]
    assert probe.INTERVENTION_SPECS[probe.CONDITION_S5]["zeroed_local_slots"] == list(
        probe.LOCAL_PARTS
    )


def test_s0_exactly_matches_reviewed_step7_d0(golden_runs):
    output, trace = golden_runs["runs"][probe.CONDITION_S0]
    expected_output, expected_trace = golden_runs["step7_d0"]
    for name in ("logits", "probabilities", "predictions", "z_image", "node_embeddings"):
        _assert_tensor_equal(output[name], expected_output[name])
    for name in probe.PART_ORDER:
        _assert_tensor_equal(output["part_embeddings"][name], expected_output["part_embeddings"][name])
        _assert_tensor_equal(
            trace["pooled_before_readout_intervention"][name],
            expected_trace["pooled_before_readout_intervention"][name],
        )
    evidence = probe.native_manual_equivalence(
        golden_runs["native"], output, golden_runs["batch"]["sample_ids"]
    )
    assert evidence["gate_pass"] is True
    assert evidence["prediction_agreement"] == 1.0
    assert evidence["max_abs_logit_difference"] <= 1e-5
    assert evidence["max_abs_probability_difference"] <= 3e-6


def test_s5_exactly_matches_reviewed_step7_d3(golden_runs):
    output, trace = golden_runs["runs"][probe.CONDITION_S5]
    expected_output, expected_trace = golden_runs["step7_d3"]
    for name in ("logits", "probabilities", "predictions", "z_image", "node_embeddings"):
        _assert_tensor_equal(output[name], expected_output[name])
    for name in probe.PART_ORDER:
        _assert_tensor_equal(output["part_embeddings"][name], expected_output["part_embeddings"][name])
        _assert_tensor_equal(
            trace["readout_part_embeddings"][name],
            expected_trace["readout_part_embeddings"][name],
        )
        _assert_tensor_equal(
            trace["readout_valid_groups"][name],
            expected_trace["readout_valid_groups"][name],
        )
    assert trace["reviewed_step7_condition"] == probe.step7.CONDITION_D3


def test_each_s1_s4_changes_only_its_named_pooled_local_embedding(golden_runs):
    official_output, official_trace = golden_runs["runs"][probe.CONDITION_S0]
    for condition, slot in probe.SLOT_BY_CONDITION.items():
        output, trace = golden_runs["runs"][condition]
        assert trace["zeroed_local_slots"] == [slot]
        for name in probe.PART_ORDER:
            expected = (
                tf.zeros_like(official_trace["pooled_before_readout_intervention"][name])
                if name == slot
                else official_trace["pooled_before_readout_intervention"][name]
            )
            _assert_tensor_equal(trace["readout_part_embeddings"][name], expected)
            _assert_tensor_equal(
                trace["readout_valid_groups"][name],
                official_trace["readout_valid_groups"][name],
            )
        _assert_tensor_equal(trace["readout_part_soft"], official_trace["readout_part_soft"])
        _assert_tensor_equal(output["node_embeddings"], official_output["node_embeddings"])
        for key in ("node_features", "edge_features", "edge_index", "node_graph_index", "pre_context_h"):
            _assert_tensor_equal(trace["message_passing"][key], official_trace["message_passing"][key])
        integrity = probe.validate_slot_integrity(
            golden_runs["model"],
            golden_runs["batch"],
            golden_runs["before_batch"],
            condition,
            trace,
        )
        assert integrity["registered_slot_arguments_exact"] is True
        assert integrity["global_embedding_unchanged"] is True
        assert integrity["validity_flags_unchanged"] is True
        assert integrity["context_output_unchanged"] is True


def test_every_condition_preserves_source_batch_and_one_model_state(golden_runs):
    assert golden_runs["before_weights"] == golden_runs["after_weights"]
    probe._assert_source_unchanged(golden_runs["batch"], golden_runs["before_batch"])
    model_ids = {id(golden_runs["model"])}
    for condition, (_output, trace) in golden_runs["runs"].items():
        model_ids.add(id(golden_runs["model"]))
        integrity = probe.validate_slot_integrity(
            golden_runs["model"],
            golden_runs["batch"],
            golden_runs["before_batch"],
            condition,
            trace,
        )
        assert integrity["source_batch_unchanged"] is True
        assert integrity["labels_and_sample_ids_unchanged"] is True
        assert integrity["node_edge_coordinates_topology_unchanged"] is True
    assert len(model_ids) == 1


def _gate_result(sample_count=probe.EXPECTED_FULL_VALIDATION_SAMPLES):
    metrics = {
        condition: {"accuracy": 0.5, "macro_f1": probe.S0_REFERENCE["macro_f1"], "loss": 1.5}
        for condition in probe.CONDITIONS
    }
    metrics[probe.CONDITION_S0] = dict(probe.S0_REFERENCE)
    metrics[probe.CONDITION_S5] = dict(probe.S5_REFERENCE)
    return {
        "sample_count": sample_count,
        "metrics": metrics,
        "native_manual_equivalence": {"status": "PASS"},
    }


def test_gate_constants_threshold_boundaries_and_decision_rules_are_exact():
    assert probe.NATIVE_MANUAL_TOLERANCE == {
        "prediction_agreement": 1.0,
        "max_abs_logit_difference": 1e-5,
        "max_abs_probability_difference": 3e-6,
    }
    assert probe.S0_REFERENCE == {
        "accuracy": 0.63137364168292,
        "macro_f1": 0.5932591901893336,
        "loss": 1.1537981840361535,
    }
    assert probe.S5_REFERENCE == {
        "accuracy": 0.22596823627751464,
        "macro_f1": 0.1958426679087715,
        "loss": 1.883221954371022,
    }
    assert probe.REFERENCE_TOLERANCE == {"accuracy": 0.001, "macro_f1": 0.001, "loss": 0.005}

    result = _gate_result()
    base = probe.S0_REFERENCE["macro_f1"]
    result["metrics"][probe.CONDITION_S1]["macro_f1"] = base - 0.10
    result["metrics"][probe.CONDITION_S2]["macro_f1"] = base - 0.05
    result["metrics"][probe.CONDITION_S3]["macro_f1"] = base - 0.049999
    result["metrics"][probe.CONDITION_S4]["macro_f1"] = base + 0.01
    gates = probe.evaluate_registered_gates(result, bounded_limit=None)
    assert gates["status"] == "VALID_REGISTERED_SLOT_DECOMPOSITION"
    assert gates["per_slot_diagnostics"][probe.CONDITION_S1]["label"] == "HIGH_SLOT_SENSITIVITY"
    assert gates["per_slot_diagnostics"][probe.CONDITION_S2]["label"] == "MODERATE_SLOT_SENSITIVITY"
    assert gates["per_slot_diagnostics"][probe.CONDITION_S3]["label"] == "LOW_SLOT_SENSITIVITY"
    assert gates["per_slot_diagnostics"][probe.CONDITION_S4]["label"] == "LOW_SLOT_SENSITIVITY"
    assert gates["overall_decision"] == "SINGLE_HIGH_LOCAL_SLOT"

    result["metrics"][probe.CONDITION_S2]["macro_f1"] = base - 0.11
    assert probe.evaluate_registered_gates(result, None)["overall_decision"] == "MULTIPLE_HIGH_LOCAL_SLOTS"
    result["metrics"][probe.CONDITION_S1]["macro_f1"] = base - 0.09
    result["metrics"][probe.CONDITION_S2]["macro_f1"] = base - 0.09
    assert probe.evaluate_registered_gates(result, None)["overall_decision"] == (
        "NO_SINGLE_HIGH_LOCAL_SLOT_WITH_JOINT_DEPENDENCY"
    )

    bounded = probe.evaluate_registered_gates(_gate_result(2), bounded_limit=1)
    assert bounded["status"] == "BOUNDED_SMOKE_NO_SCIENTIFIC_INTERPRETATION"
    assert bounded["per_slot_diagnostics"] is None
    assert bounded["overall_decision"] is None
    invalid_s0 = _gate_result(3588)
    assert probe.evaluate_registered_gates(invalid_s0, None)["status"] == "INVALID_S0_REFERENCE_REPRODUCTION"
    invalid_s5 = _gate_result()
    invalid_s5["metrics"][probe.CONDITION_S5]["loss"] += 0.006
    assert probe.evaluate_registered_gates(invalid_s5, None)["status"] == "INVALID_D3_ANCHOR_REPRODUCTION"


def _minimal_batch():
    return {
        "labels": tf.constant([0, 1], dtype=tf.int64),
        "sample_ids": tf.constant([10, 11], dtype=tf.int64),
        "sentinel": tf.constant([3.0], dtype=tf.float32),
    }


def test_paired_fixed_order_determinism_and_no_extra_condition(monkeypatch):
    calls = []

    class FakeModel:
        autocast = False
        input_dtype = "float32"

        def __call__(self, batch, training=False):
            assert training is False
            logits = tf.constant([[4.0, 0, 0, 0, 0, 0, 0], [0, 4.0, 0, 0, 0, 0, 0]])
            return {"logits": logits, "probabilities": tf.nn.softmax(logits, axis=-1)}

    def fake_official(_model, _batch):
        logits = tf.constant([[4.0, 0, 0, 0, 0, 0, 0], [0, 4.0, 0, 0, 0, 0, 0]])
        return {"logits": logits, "probabilities": tf.nn.softmax(logits, axis=-1)}, {"official": True}

    def fake_forward(_model, _batch, condition, _official_output, _official_trace):
        calls.append(condition)
        index = probe.CONDITIONS.index(condition)
        logits = tf.constant(
            [[4.0 - index, float(index), 0, 0, 0, 0, 0], [float(index), 4.0 - index, 0, 0, 0, 0, 0]],
            dtype=tf.float32,
        )
        return {"logits": logits, "probabilities": tf.nn.softmax(logits, axis=-1)}, {"condition": condition}

    monkeypatch.setattr(probe.step6, "validate_batch_schema", lambda _batch: {})
    monkeypatch.setattr(probe, "_official_manual_state", fake_official)
    monkeypatch.setattr(probe, "_forward_from_official_state", fake_forward)
    monkeypatch.setattr(probe, "_snapshot_batch", lambda batch: {name: np.array(value.numpy(), copy=True) for name, value in batch.items()})
    monkeypatch.setattr(probe, "validate_slot_integrity", lambda *_args, **_kwargs: {"model_boundary_input_semantics": {"test_double": True}})
    monkeypatch.setattr(probe, "_assert_source_unchanged", lambda *_args: None)

    first = probe.evaluate_conditions(FakeModel(), [_minimal_batch()])
    assert calls == list(probe.CONDITIONS)
    calls.clear()
    second = probe.evaluate_conditions(FakeModel(), [_minimal_batch()])
    assert calls == list(probe.CONDITIONS)
    assert set(first["probabilities"]) == set(probe.CONDITIONS)
    np.testing.assert_array_equal(first["sample_ids"], [10, 11])
    assert probe._paired_predictions_csv(first) == probe._paired_predictions_csv(second)
    assert first["metrics"] == second["metrics"]


def test_checkpoint_load_is_compile_false_once_and_rejects_optimizer(tmp_path, monkeypatch):
    checkpoint = tmp_path / "fixed.keras"
    checkpoint.write_bytes(b"checkpoint")
    calls = []

    class FakeModel:
        optimizer = None

        @staticmethod
        def count_params():
            return probe.EXPECTED_PARAMETER_COUNT

    def fake_load(path, **kwargs):
        calls.append((Path(path), kwargs))
        return FakeModel()

    monkeypatch.setattr(probe.tf.keras.models, "load_model", fake_load)
    assert isinstance(probe.load_fixed_checkpoint(checkpoint), FakeModel)
    assert len(calls) == 1
    assert calls[0][1]["compile"] is False

    class OptimizerModel(FakeModel):
        optimizer = object()

    monkeypatch.setattr(probe.tf.keras.models, "load_model", lambda *_args, **_kwargs: OptimizerModel())
    with pytest.raises(probe.LocalResidualSlotProbeError, match="optimizer"):
        probe.load_fixed_checkpoint(checkpoint)

    class WrongSizeModel(FakeModel):
        @staticmethod
        def count_params():
            return probe.EXPECTED_PARAMETER_COUNT - 1

    monkeypatch.setattr(probe.tf.keras.models, "load_model", lambda *_args, **_kwargs: WrongSizeModel())
    with pytest.raises(probe.LocalResidualSlotProbeError, match="parameter count drift"):
        probe.load_fixed_checkpoint(checkpoint)


def test_manual_forward_rejects_unknown_condition_and_nonconforming_schema(golden_runs):
    with pytest.raises(probe.LocalResidualSlotProbeError, match="Unknown"):
        probe.manual_forward(golden_runs["model"], golden_runs["batch"], "S6")
    malformed = dict(golden_runs["batch"])
    malformed.pop("edge_index")
    with pytest.raises(probe.step6.PriorProbeError, match="fields drift"):
        probe.manual_forward(golden_runs["model"], malformed, probe.CONDITION_S0)


def test_main_constructs_validation_only_generator_and_one_bounded_run(tmp_path, monkeypatch):
    checkpoint = tmp_path / "fixed.keras"
    checkpoint.write_bytes(b"immutable")
    metadata = tmp_path / "fixed.metadata.json"
    metadata.write_text("{}", encoding="utf-8")
    resolved = tmp_path / "resolved_config.json"
    resolved.write_text(json.dumps({"seed": 42, "training": {"batch_size": 16}, "resources": {"memory_growth": False}}), encoding="utf-8")
    prior = tmp_path / "prior"
    cache = tmp_path / "cache"
    prior.mkdir()
    cache.mkdir()
    output = tmp_path / "output"
    events = []

    class FakeModel:
        pass

    class FakeGenerator:
        def __init__(self, **kwargs):
            events.append(("generator", kwargs))

        @staticmethod
        def iter_epoch(epoch, limit_batches=None):
            events.append(("iter_epoch", epoch, limit_batches))
            return iter(())

    monkeypatch.setattr(probe.step6, "validate_checkpoint_metadata", lambda *_: {})
    monkeypatch.setattr(probe.step6, "validate_frozen_contract", lambda *_: {
        "scientific_payload_sha256": probe.EXPECTED_SCIENTIFIC_PAYLOAD_SHA256,
        "execution_contract_sha256": probe.EXPECTED_EXECUTION_CONTRACT_SHA256,
    })
    monkeypatch.setattr(probe.step6, "configure_gpu_memory_growth", lambda requested: {
        "memory_growth_requested": requested,
        "memory_growth_status": "not_requested",
        "memory_growth_devices": [],
    })
    monkeypatch.setattr(probe, "load_fixed_checkpoint", lambda path: events.append(("load", path)) or FakeModel())
    monkeypatch.setattr(probe.step6, "model_weights_sha256", lambda _model: "w" * 64)
    monkeypatch.setattr(probe, "GraphBatchGenerator", FakeGenerator)
    monkeypatch.setattr(probe, "evaluate_conditions", lambda _model, batches: events.append(("evaluate", list(batches))) or {"ok": True})
    monkeypatch.setattr(probe, "write_probe_outputs", lambda _root, **_kwargs: {
        "registered_gates_and_diagnostics": {"status": "BOUNDED_SMOKE_NO_SCIENTIFIC_INTERPRETATION"}
    })
    assert probe.main([
        "--checkpoint", str(checkpoint),
        "--checkpoint-metadata", str(metadata),
        "--resolved-config", str(resolved),
        "--prior-root", str(prior),
        "--clean-graph-cache-dir", str(cache),
        "--output-root", str(output),
        "--limit-val-batches", "2",
    ]) == 0
    assert len([event for event in events if event[0] == "load"]) == 1
    generator = next(event[1] for event in events if event[0] == "generator")
    assert generator["split"] == "val"
    assert generator["shuffle"] is False
    assert ("iter_epoch", 0, 2) in events


def test_output_fresh_cli_closed_and_failure_evidence(tmp_path):
    destinations = {action.dest for action in probe.build_parser()._actions}
    assert "limit_val_batches" in destinations
    for forbidden in ("split", "condition", "slot", "intervention", "pair"):
        assert forbidden not in destinations

    output = tmp_path / "failure"
    probe.write_equivalence_failure(output, {"status": "INVALID_MANUAL_FORWARD_EQUIVALENCE"}, checkpoint_path=tmp_path / "x.keras")
    manifest = json.loads((output / "probe_manifest.json").read_text(encoding="utf-8"))
    assert manifest["scientific_interpretation"] is None
    assert manifest["training_performed"] is False
    assert manifest["test_access"] is False
    assert not list(output.glob("validation_metrics_*.json"))
    with pytest.raises(FileExistsError, match="Fresh probe output"):
        probe.write_equivalence_failure(output, {}, checkpoint_path=tmp_path / "x.keras")


def test_scientific_payload_and_static_isolation_contract_remain_exact():
    assert probe.step6.scientific_payload_checksum(PACKAGE_ROOT) == probe.EXPECTED_SCIENTIFIC_PAYLOAD_SHA256
    source = TOOL_PATH.read_text(encoding="utf-8")
    for forbidden in (
        'split="train"',
        'split="test"',
        "model.fit(",
        "GradientTape(",
        "apply_gradients(",
        "build_graph(",
        "raw_prior",
        "--split",
        "--condition",
        "--slot",
        "--pair",
    ):
        assert forbidden not in source
    assert 'split="val"' in source
    assert "compile=False" in source
    assert '"optimizer_created": False' in source
    assert '"test_split_constructed": False' in source
    assert "CONDITIONS[1:5]" in source
