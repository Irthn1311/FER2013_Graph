from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from _helpers import loaded
from lap_gnn_tf.conversion import load_pytorch_npz
from lap_gnn_tf.graph.batch import load_golden_batch
from lap_gnn_tf.model import LapGNN


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = (
    PACKAGE_ROOT
    / "tools"
    / "evaluate_fixed_checkpoint_direct_part_decomposition_probe.py"
)
SPEC = importlib.util.spec_from_file_location("direct_part_decomposition_probe", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


@pytest.fixture(scope="module")
def golden_runs():
    model, batch = loaded()
    before_batch = probe._snapshot_batch(batch)
    before_weights = probe.step6.model_weights_sha256(model)
    native = model(batch, training=False)
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
        "runs": runs,
    }


@pytest.fixture(scope="module")
def mixed_boundary_runs():
    """Reproduce the persisted outer-mixed/internal-float32 Keras boundary."""

    class BoundaryObservedLapGNN(LapGNN):
        def call(self, batch, *args, **kwargs):
            self.observed_boundary_dtypes = {
                name: value.dtype.name for name, value in batch.items()
            }
            return super().call(batch, *args, **kwargs)

    previous = tf.keras.mixed_precision.global_policy()
    try:
        tf.keras.mixed_precision.set_global_policy("float32")
        batch = load_golden_batch(
            str(PACKAGE_ROOT / "validation_assets" / "golden" / "graph_batch.npz")
        )
        model = BoundaryObservedLapGNN(dtype="mixed_float16")
        LapGNN.call(model, batch, training=False)
        model.build({name: value.shape for name, value in batch.items()})
        load_pytorch_npz(
            model,
            PACKAGE_ROOT / "validation_assets" / "golden" / "model_state.npz",
        )
        before_batch = probe._snapshot_batch(batch)
        native = model(batch, training=False)
        runs = {
            condition: probe.manual_forward(model, batch, condition)
            for condition in probe.CONDITIONS
        }
        yield {
            "model": model,
            "batch": batch,
            "before_batch": before_batch,
            "native": native,
            "runs": runs,
            "observed_boundary_dtypes": dict(model.observed_boundary_dtypes),
        }
    finally:
        tf.keras.mixed_precision.set_global_policy(previous)


def _assert_tensor_equal(actual, expected):
    np.testing.assert_array_equal(actual.numpy(), expected.numpy())


def test_condition_order_and_registered_semantics_are_exact():
    assert probe.CONDITIONS == (
        "official_manual_forward",
        "context_local_prior_neutralized",
        "readout_local_prior_neutralized",
        "local_part_residual_zero",
        "local_motif_validity_off",
        "full_direct_part_zero_anchor",
    )
    assert probe.INTERVENTION_SPECS[probe.CONDITION_D1][
        "changed_pathway_arguments"
    ] == ["context.part_soft"]
    assert probe.INTERVENTION_SPECS[probe.CONDITION_D2][
        "changed_pathway_arguments"
    ] == ["readout.part_soft"]
    assert probe.INTERVENTION_SPECS[probe.CONDITION_D3][
        "changed_pathway_arguments"
    ] == [
        "readout.part_embeddings.mouth",
        "readout.part_embeddings.eye",
        "readout.part_embeddings.brow",
        "readout.part_embeddings.nose_cheek",
    ]
    assert probe.INTERVENTION_SPECS[probe.CONDITION_D4][
        "changed_pathway_arguments"
    ] == [
        "readout.valid_groups.mouth",
        "readout.valid_groups.eye",
        "readout.valid_groups.brow",
        "readout.valid_groups.nose_cheek",
    ]
    assert probe.INTERVENTION_SPECS[probe.CONDITION_D5][
        "changed_pathway_arguments"
    ] == [
        "context.part_soft",
        "part_pool.part_soft",
        "part_pool.valid_part_mask",
        "readout.part_soft",
    ]


def test_d0_manual_forward_is_exactly_equivalent_to_native_on_golden(golden_runs):
    manual, trace = golden_runs["runs"][probe.CONDITION_D0]
    evidence = probe.native_manual_equivalence(
        golden_runs["native"], manual, golden_runs["batch"]["sample_ids"]
    )
    assert evidence["gate_pass"] is True
    assert evidence["prediction_agreement"] == 1.0
    assert evidence["max_abs_logit_difference"] <= 1e-5
    assert evidence["max_abs_probability_difference"] <= 3e-6
    integrity = probe.validate_pathway_integrity(
        golden_runs["model"],
        golden_runs["batch"],
        golden_runs["before_batch"],
        probe.CONDITION_D0,
        trace,
    )
    assert integrity["changed_pathway_arguments"] == []


def test_d1_d4_change_only_the_registered_intermediate_arguments(golden_runs):
    batch = golden_runs["batch"]
    official_part = tf.cast(batch["part_soft"], tf.float32)
    official_valid_mask = tf.cast(batch["valid_part_mask"], tf.float32)
    traces = {name: value[1] for name, value in golden_runs["runs"].items()}

    d1 = traces[probe.CONDITION_D1]
    _assert_tensor_equal(d1["context_part_soft"], tf.zeros_like(official_part))
    _assert_tensor_equal(d1["pool_part_soft"], official_part)
    _assert_tensor_equal(d1["pool_valid_part_mask"], official_valid_mask)
    _assert_tensor_equal(d1["readout_part_soft"], official_part)

    d2 = traces[probe.CONDITION_D2]
    _assert_tensor_equal(d2["context_part_soft"], official_part)
    _assert_tensor_equal(d2["pool_part_soft"], official_part)
    _assert_tensor_equal(d2["pool_valid_part_mask"], official_valid_mask)
    _assert_tensor_equal(d2["readout_part_soft"], tf.zeros_like(official_part))
    for name in probe.PART_ORDER:
        _assert_tensor_equal(
            d2["readout_part_embeddings"][name],
            d2["pooled_before_readout_intervention"][name],
        )
        _assert_tensor_equal(
            d2["readout_valid_groups"][name],
            d2["valid_before_readout_intervention"][name],
        )

    d3 = traces[probe.CONDITION_D3]
    for name in probe.LOCAL_PARTS:
        _assert_tensor_equal(
            d3["readout_part_embeddings"][name],
            tf.zeros_like(d3["pooled_before_readout_intervention"][name]),
        )
    _assert_tensor_equal(
        d3["readout_part_embeddings"]["global"],
        d3["pooled_before_readout_intervention"]["global"],
    )
    for name in probe.PART_ORDER:
        _assert_tensor_equal(
            d3["readout_valid_groups"][name],
            d3["valid_before_readout_intervention"][name],
        )
    _assert_tensor_equal(d3["readout_part_soft"], official_part)

    d4 = traces[probe.CONDITION_D4]
    for name in probe.PART_ORDER:
        _assert_tensor_equal(
            d4["readout_part_embeddings"][name],
            d4["pooled_before_readout_intervention"][name],
        )
    for name in probe.LOCAL_PARTS:
        _assert_tensor_equal(
            d4["readout_valid_groups"][name],
            tf.zeros_like(d4["valid_before_readout_intervention"][name], dtype=tf.bool),
        )
    _assert_tensor_equal(
        d4["readout_valid_groups"]["global"],
        tf.ones_like(d4["valid_before_readout_intervention"]["global"], dtype=tf.bool),
    )
    _assert_tensor_equal(d4["readout_part_soft"], official_part)

    for condition in probe.CONDITIONS[1:5]:
        evidence = probe.validate_pathway_integrity(
            golden_runs["model"],
            batch,
            golden_runs["before_batch"],
            condition,
            traces[condition],
        )
        assert evidence["registered_pathway_arguments_exact"] is True
        assert evidence["node_edge_topology_unchanged"] is True


def test_d5_exactly_matches_step6_c1_semantics(golden_runs):
    model = golden_runs["model"]
    batch = golden_runs["batch"]
    d5_output, d5_trace = golden_runs["runs"][probe.CONDITION_D5]
    step6_c1_batch = probe.step6.apply_intervention(
        batch, probe.step6.CONDITION_DIRECT_ZERO
    )
    step6_c1_output = model(step6_c1_batch, training=False)
    np.testing.assert_allclose(
        d5_output["logits"].numpy(),
        step6_c1_output["logits"].numpy(),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        d5_output["probabilities"].numpy(),
        step6_c1_output["probabilities"].numpy(),
        rtol=0.0,
        atol=0.0,
    )
    probe.validate_pathway_integrity(
        model,
        batch,
        golden_runs["before_batch"],
        probe.CONDITION_D5,
        d5_trace,
    )


def test_mixed_float16_boundary_reproduces_old_gate_a_failure_and_fix(
    mixed_boundary_runs, monkeypatch
):
    model = mixed_boundary_runs["model"]
    batch = mixed_boundary_runs["batch"]
    assert tf.keras.mixed_precision.global_policy().name == "float32"
    assert model.dtype_policy.name == "mixed_float16"
    assert model.compute_dtype == "float16"
    assert model.encoder.dtype_policy.name == "float32"
    assert batch["node_features"].dtype == tf.float32
    assert mixed_boundary_runs["observed_boundary_dtypes"]["node_features"] == "float16"
    assert mixed_boundary_runs["observed_boundary_dtypes"]["part_soft"] == "float16"
    assert mixed_boundary_runs["observed_boundary_dtypes"]["labels"] == "int64"

    with monkeypatch.context() as patch:
        patch.setattr(
            probe,
            "_normalize_model_boundary_inputs",
            lambda _model, source: dict(source),
        )
        legacy_manual, _ = probe.manual_forward(
            model, batch, probe.CONDITION_D0
        )
        with pytest.raises(probe.ManualForwardEquivalenceError) as captured:
            probe.evaluate_conditions(model, [batch])
    legacy_evidence = probe.native_manual_equivalence(
        mixed_boundary_runs["native"], legacy_manual, batch["sample_ids"]
    )
    assert legacy_evidence["gate_pass"] is False
    assert legacy_evidence["max_abs_logit_difference"] > 1e-5
    assert legacy_evidence["max_abs_probability_difference"] > 1e-6
    assert captured.value.evidence["status"] == (
        "INVALID_MANUAL_FORWARD_EQUIVALENCE"
    )
    assert captured.value.evidence["model_boundary_input_semantics"] == {
        "autocast": True,
        "input_dtype": "float16",
        "source_dtypes": {
            name: value.dtype.name for name, value in batch.items()
        },
        "effective_dtypes": {
            name: value.dtype.name for name, value in batch.items()
        },
    }
    assert len(captured.value.evidence["batches"]) == 1

    fixed_manual, fixed_trace = mixed_boundary_runs["runs"][probe.CONDITION_D0]
    fixed_evidence = probe.native_manual_equivalence(
        mixed_boundary_runs["native"], fixed_manual, batch["sample_ids"]
    )
    assert fixed_evidence["gate_pass"] is True
    assert fixed_evidence["prediction_agreement"] == 1.0
    assert fixed_evidence["max_abs_logit_difference"] == 0.0
    assert fixed_evidence["max_abs_probability_difference"] == 0.0
    integrity = probe.validate_pathway_integrity(
        model,
        batch,
        mixed_boundary_runs["before_batch"],
        probe.CONDITION_D0,
        fixed_trace,
    )
    assert integrity["model_boundary_input_semantics"]["autocast"] is True
    assert integrity["model_boundary_input_semantics"]["input_dtype"] == "float16"


def test_mixed_float16_boundary_preserves_d5_anchor_and_all_registered_semantics(
    mixed_boundary_runs
):
    model = mixed_boundary_runs["model"]
    batch = mixed_boundary_runs["batch"]
    d5_output, _ = mixed_boundary_runs["runs"][probe.CONDITION_D5]
    step6_c1_batch = probe.step6.apply_intervention(
        batch, probe.step6.CONDITION_DIRECT_ZERO
    )
    native_step6_c1 = model(step6_c1_batch, training=False)
    np.testing.assert_allclose(
        d5_output["logits"].numpy(),
        native_step6_c1["logits"].numpy(),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        d5_output["probabilities"].numpy(),
        native_step6_c1["probabilities"].numpy(),
        rtol=0.0,
        atol=0.0,
    )

    for condition in probe.CONDITIONS:
        _, trace = mixed_boundary_runs["runs"][condition]
        integrity = probe.validate_pathway_integrity(
            model,
            batch,
            mixed_boundary_runs["before_batch"],
            condition,
            trace,
        )
        assert integrity["changed_pathway_arguments"] == list(
            probe.INTERVENTION_SPECS[condition]["changed_pathway_arguments"]
        )
        assert integrity["registered_pathway_arguments_exact"] is True
        assert integrity["source_batch_unchanged"] is True
    probe._assert_source_unchanged(batch, mixed_boundary_runs["before_batch"])


def test_every_condition_preserves_source_batch_and_one_model_state(golden_runs):
    probe._assert_source_unchanged(
        golden_runs["batch"], golden_runs["before_batch"]
    )
    assert golden_runs["before_weights"] == golden_runs["after_weights"]
    for condition, (_, trace) in golden_runs["runs"].items():
        message = trace["message_passing"]
        _assert_tensor_equal(
            message["node_features"],
            tf.cast(golden_runs["batch"]["node_features"], tf.float32),
        )
        _assert_tensor_equal(
            message["edge_features"],
            tf.cast(golden_runs["batch"]["edge_features"], tf.float32),
        )
        _assert_tensor_equal(
            message["edge_index"],
            tf.cast(golden_runs["batch"]["edge_index"], tf.int64),
        )
        _assert_tensor_equal(
            message["pre_context_h"],
            golden_runs["runs"][probe.CONDITION_D0][1]["message_passing"][
                "pre_context_h"
            ],
        )
        assert condition in probe.CONDITIONS


def test_manual_equivalence_failure_uses_the_registered_fail_closed_label():
    native = {
        "logits": tf.constant([[2.0, 0.0]], dtype=tf.float32),
        "probabilities": tf.constant([[0.9, 0.1]], dtype=tf.float32),
    }
    manual = {
        "logits": tf.constant([[0.0, 2.0]], dtype=tf.float32),
        "probabilities": tf.constant([[0.1, 0.9]], dtype=tf.float32),
    }
    evidence = probe.native_manual_equivalence(
        native, manual, tf.constant([42], dtype=tf.int64)
    )
    assert evidence["gate_pass"] is False
    error = probe.ManualForwardEquivalenceError({"batches": [evidence]})
    assert str(error) == "INVALID_MANUAL_FORWARD_EQUIVALENCE"
    assert error.evidence["batches"][0]["prediction_agreement"] == 0.0


def test_gate_a_probability_calibration_and_unchanged_guards_are_exact():
    assert probe.NATIVE_MANUAL_TOLERANCE == {
        "prediction_agreement": 1.0,
        "max_abs_logit_difference": 1e-5,
        "max_abs_probability_difference": 3e-6,
    }
    sample_ids = tf.constant([100, 101], dtype=tf.int64)
    base_logits = tf.constant([[2.0, 0.0], [0.0, 2.0]], dtype=tf.float64)
    base_probabilities = tf.constant(
        [[0.75, 0.25], [0.25, 0.75]], dtype=tf.float64
    )

    def evidence(*, logits=base_logits, probabilities=base_probabilities):
        return probe.native_manual_equivalence(
            {
                "logits": base_logits,
                "probabilities": base_probabilities,
            },
            {"logits": logits, "probabilities": probabilities},
            sample_ids,
        )

    prediction_native_probabilities = tf.constant(
        [[0.500001, 0.499999], [0.25, 0.75]], dtype=tf.float64
    )
    prediction_mismatch = probe.native_manual_equivalence(
        {
            "logits": base_logits,
            "probabilities": prediction_native_probabilities,
        },
        {
            "logits": base_logits,
            "probabilities": tf.constant(
                [[0.499999, 0.500001], [0.25, 0.75]], dtype=tf.float64
            ),
        },
        sample_ids,
    )
    assert prediction_mismatch["prediction_agreement"] < 1.0
    assert prediction_mismatch["max_abs_logit_difference"] <= 1e-5
    assert prediction_mismatch["max_abs_probability_difference"] <= 3e-6
    assert prediction_mismatch["gate_pass"] is False

    logit_mismatch = evidence(
        logits=tf.constant(
            [[2.000011, 0.0], [0.0, 2.0]], dtype=tf.float64
        )
    )
    assert logit_mismatch["prediction_agreement"] == 1.0
    assert logit_mismatch["max_abs_probability_difference"] == 0.0
    assert logit_mismatch["max_abs_logit_difference"] > 1e-5
    assert logit_mismatch["gate_pass"] is False

    forensic_envelope = 2.205371856689453e-06
    envelope_evidence = evidence(
        probabilities=tf.constant(
            [
                [0.75 + forensic_envelope, 0.25 - forensic_envelope],
                [0.25, 0.75],
            ],
            dtype=tf.float64,
        )
    )
    assert envelope_evidence["max_abs_probability_difference"] == pytest.approx(
        forensic_envelope, rel=0.0, abs=1e-15
    )
    assert envelope_evidence["gate_pass"] is True

    within_threshold = 2.999e-6
    threshold_evidence = evidence(
        probabilities=tf.constant(
            [
                [0.75 + within_threshold, 0.25 - within_threshold],
                [0.25, 0.75],
            ],
            dtype=tf.float64,
        )
    )
    assert threshold_evidence["max_abs_probability_difference"] == pytest.approx(
        within_threshold, rel=0.0, abs=1e-15
    )
    assert threshold_evidence["max_abs_probability_difference"] <= 3e-6
    assert threshold_evidence["gate_pass"] is True

    above_threshold = evidence(
        probabilities=tf.constant(
            [[0.7500031, 0.2499969], [0.25, 0.75]], dtype=tf.float64
        )
    )
    assert above_threshold["max_abs_probability_difference"] > 3e-6
    assert above_threshold["gate_pass"] is False

    controlled_pre_hotfix = 1.744628e-4
    pre_hotfix_evidence = evidence(
        probabilities=tf.constant(
            [
                [0.75 + controlled_pre_hotfix, 0.25 - controlled_pre_hotfix],
                [0.25, 0.75],
            ],
            dtype=tf.float64,
        )
    )
    assert pre_hotfix_evidence["max_abs_probability_difference"] == pytest.approx(
        controlled_pre_hotfix, rel=0.0, abs=1e-15
    )
    assert pre_hotfix_evidence["max_abs_probability_difference"] > 58 * 3e-6
    assert pre_hotfix_evidence["gate_pass"] is False


def test_equivalence_failure_evidence_is_preserved_without_condition_metrics(tmp_path):
    output = tmp_path / "failure"
    checkpoint = tmp_path / "fixed.keras"
    checkpoint.write_bytes(b"immutable")
    evidence = {
        "status": "INVALID_MANUAL_FORWARD_EQUIVALENCE",
        "batches": [{"batch_index": 0, "gate_pass": False}],
    }
    probe.write_equivalence_failure(
        output, evidence, checkpoint_path=checkpoint
    )
    persisted = json.loads(
        (output / "native_manual_d0_equivalence.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (output / "probe_manifest.json").read_text(encoding="utf-8")
    )
    assert persisted == evidence
    assert manifest["status"] == "INVALID_MANUAL_FORWARD_EQUIVALENCE"
    assert manifest["scientific_interpretation"] is None
    assert not list(output.glob("validation_metrics_*.json"))


def _minimal_batch():
    return {
        "labels": tf.constant([0, 1], dtype=tf.int64),
        "sample_ids": tf.constant([10, 11], dtype=tf.int64),
        "sentinel": tf.constant([3.0], dtype=tf.float32),
    }


def test_paired_condition_order_sample_identity_and_determinism(monkeypatch):
    calls = []

    class FakeModel:
        def __call__(self, batch, training=False):
            assert training is False
            logits = tf.constant([[4.0, 0, 0, 0, 0, 0, 0], [0, 4.0, 0, 0, 0, 0, 0]])
            return {
                "logits": logits,
                "probabilities": tf.nn.softmax(logits, axis=-1),
            }

    def fake_manual(_model, batch, condition):
        calls.append(condition)
        index = probe.CONDITIONS.index(condition)
        logits = tf.constant(
            [[4.0 - index, float(index), 0, 0, 0, 0, 0],
             [float(index), 4.0 - index, 0, 0, 0, 0, 0]],
            dtype=tf.float32,
        )
        return {
            "logits": logits,
            "probabilities": tf.nn.softmax(logits, axis=-1),
        }, {"condition": condition}

    monkeypatch.setattr(probe.step6, "validate_batch_schema", lambda _batch: {})
    monkeypatch.setattr(probe, "manual_forward", fake_manual)
    monkeypatch.setattr(
        probe,
        "validate_pathway_integrity",
        lambda *_args, **_kwargs: {
            "model_boundary_input_semantics": {"test_double": True}
        },
    )
    first = probe.evaluate_conditions(FakeModel(), [_minimal_batch()])
    assert calls == list(probe.CONDITIONS)
    calls.clear()
    second = probe.evaluate_conditions(FakeModel(), [_minimal_batch()])
    assert calls == list(probe.CONDITIONS)
    np.testing.assert_array_equal(first["sample_ids"], [10, 11])
    assert probe._paired_predictions_csv(first) == probe._paired_predictions_csv(second)
    for condition in probe.CONDITIONS:
        np.testing.assert_array_equal(
            first["probabilities"][condition], second["probabilities"][condition]
        )
    assert first["metrics"] == second["metrics"]
    assert first["paired_diagnostics"] == second["paired_diagnostics"]
    assert first["integrity_counts"] == {condition: 1 for condition in probe.CONDITIONS}


def _gate_result(sample_count=3589):
    metrics = {
        condition: {
            "accuracy": 0.5,
            "macro_f1": 0.5,
            "loss": 1.5,
        }
        for condition in probe.CONDITIONS
    }
    metrics[probe.CONDITION_D0] = dict(probe.D0_REFERENCE)
    metrics[probe.CONDITION_D5] = dict(probe.D5_REFERENCE)
    metrics[probe.CONDITION_D1]["macro_f1"] = 0.48
    metrics[probe.CONDITION_D2]["macro_f1"] = 0.40
    metrics[probe.CONDITION_D3]["macro_f1"] = 0.43
    metrics[probe.CONDITION_D4]["macro_f1"] = 0.60
    return {
        "sample_count": sample_count,
        "metrics": metrics,
        "native_manual_equivalence": {"status": "PASS"},
    }


def test_registered_gates_and_non_additive_decision_rule_are_exact():
    bounded = probe.evaluate_registered_gates(_gate_result(8), bounded_limit=1)
    assert bounded["status"] == "BOUNDED_SMOKE_NO_SCIENTIFIC_INTERPRETATION"
    assert bounded["per_path_diagnostics"] is None
    assert bounded["overall_decision"] is None

    valid = probe.evaluate_registered_gates(_gate_result(), bounded_limit=None)
    assert valid["status"] == "VALID_REGISTERED_DECOMPOSITION"
    assert valid["per_path_diagnostics"][probe.CONDITION_D1]["label"] == (
        "HIGH_PATH_SENSITIVITY"
    )
    assert valid["per_path_diagnostics"][probe.CONDITION_D2]["label"] == (
        "HIGH_PATH_SENSITIVITY"
    )
    assert valid["per_path_diagnostics"][probe.CONDITION_D3]["label"] == (
        "HIGH_PATH_SENSITIVITY"
    )
    d4 = valid["per_path_diagnostics"][probe.CONDITION_D4]
    assert d4["label"] == "LOW_PATH_SENSITIVITY"
    assert d4["delta_f1_pp"] < 0
    assert "improved macro-F1" in d4["negative_effect_note"]
    assert valid["overall_decision"] == "MULTIPLE_HIGH_DIRECT_PATHS"
    assert "must not be summed" in valid["non_additivity_warning"]

    invalid_d0 = _gate_result(sample_count=3588)
    gate = probe.evaluate_registered_gates(invalid_d0, bounded_limit=None)
    assert gate["status"] == "INVALID_D0_REFERENCE_REPRODUCTION"
    assert gate["per_path_diagnostics"] is None

    invalid_d5 = _gate_result()
    invalid_d5["metrics"][probe.CONDITION_D5]["loss"] += 0.01
    gate = probe.evaluate_registered_gates(invalid_d5, bounded_limit=None)
    assert gate["status"] == "INVALID_C1_ANCHOR_REPRODUCTION"
    assert gate["overall_decision"] is None


def test_checkpoint_load_is_compile_false_once_and_rejects_optimizer(
    tmp_path, monkeypatch
):
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
    assert calls[0][0] == checkpoint
    assert calls[0][1]["compile"] is False

    class OptimizerModel(FakeModel):
        optimizer = object()

    monkeypatch.setattr(
        probe.tf.keras.models,
        "load_model",
        lambda *_args, **_kwargs: OptimizerModel(),
    )
    with pytest.raises(probe.DirectPartProbeError, match="optimizer"):
        probe.load_fixed_checkpoint(checkpoint)

    class WrongSizeModel(FakeModel):
        @staticmethod
        def count_params():
            return probe.EXPECTED_PARAMETER_COUNT - 1

    monkeypatch.setattr(
        probe.tf.keras.models,
        "load_model",
        lambda *_args, **_kwargs: WrongSizeModel(),
    )
    with pytest.raises(probe.DirectPartProbeError, match="parameter count drift"):
        probe.load_fixed_checkpoint(checkpoint)


def test_manual_forward_rejects_unknown_condition_and_nonconforming_schema(
    golden_runs,
):
    with pytest.raises(probe.DirectPartProbeError, match="Unknown"):
        probe.manual_forward(golden_runs["model"], golden_runs["batch"], "D6")
    malformed = dict(golden_runs["batch"])
    malformed.pop("edge_index")
    with pytest.raises(probe.step6.PriorProbeError, match="fields drift"):
        probe.manual_forward(golden_runs["model"], malformed, probe.CONDITION_D0)


def test_main_constructs_validation_only_generator_and_one_bounded_run(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "fixed.keras"
    checkpoint.write_bytes(b"immutable")
    metadata = tmp_path / "fixed.metadata.json"
    metadata.write_text("{}", encoding="utf-8")
    resolved = tmp_path / "resolved_config.json"
    resolved.write_text(
        json.dumps(
            {
                "seed": 42,
                "training": {"batch_size": 16},
                "resources": {"memory_growth": False},
            }
        ),
        encoding="utf-8",
    )
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
    monkeypatch.setattr(
        probe.step6,
        "validate_frozen_contract",
        lambda *_: {"scientific_payload_sha256": probe.EXPECTED_SCIENTIFIC_PAYLOAD_SHA256},
    )
    monkeypatch.setattr(
        probe.step6,
        "configure_gpu_memory_growth",
        lambda requested: {
            "memory_growth_requested": requested,
            "memory_growth_status": "not_requested",
            "memory_growth_devices": [],
        },
    )
    monkeypatch.setattr(probe, "load_fixed_checkpoint", lambda path: events.append(("load", path)) or FakeModel())
    monkeypatch.setattr(probe.step6, "model_weights_sha256", lambda _model: "w" * 64)
    monkeypatch.setattr(probe, "GraphBatchGenerator", FakeGenerator)
    monkeypatch.setattr(
        probe,
        "evaluate_conditions",
        lambda _model, batches: events.append(("evaluate", list(batches))) or {"ok": True},
    )
    monkeypatch.setattr(
        probe,
        "write_probe_outputs",
        lambda _root, **_kwargs: {
            "registered_gates_and_diagnostics": {
                "status": "BOUNDED_SMOKE_NO_SCIENTIFIC_INTERPRETATION"
            }
        },
    )
    assert probe.main(
        [
            "--checkpoint", str(checkpoint),
            "--checkpoint-metadata", str(metadata),
            "--resolved-config", str(resolved),
            "--prior-root", str(prior),
            "--clean-graph-cache-dir", str(cache),
            "--output-root", str(output),
            "--limit-val-batches", "2",
        ]
    ) == 0
    assert len([event for event in events if event[0] == "load"]) == 1
    generator = next(event[1] for event in events if event[0] == "generator")
    assert generator["split"] == "val"
    assert generator["shuffle"] is False
    assert generator["prior_root"] == prior.resolve()
    assert generator["clean_graph_cache_dir"] == cache.resolve()
    assert ("iter_epoch", 0, 2) in events


def test_output_must_be_fresh_and_cli_has_no_split_or_condition_selector(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="Fresh probe output"):
        probe.write_probe_outputs(
            output,
            result={},
            checkpoint_path=tmp_path / "x.keras",
            checkpoint_metadata_path=tmp_path / "x.metadata.json",
            checkpoint_sha256_before="a",
            checkpoint_sha256_after="a",
            model_weights_sha256_before="b",
            model_weights_sha256_after="b",
            resolved_config_path=tmp_path / "config.json",
            resolved_config_sha256="c",
            contract={},
            checkpoint_cross_check={},
            resources={},
            bounded_limit=1,
        )
    destinations = {action.dest for action in probe.build_parser()._actions}
    assert "limit_val_batches" in destinations
    assert "split" not in destinations
    assert "condition" not in destinations
    assert "intervention" not in destinations


def test_scientific_payload_and_static_isolation_contract_remain_exact():
    assert probe.step6.scientific_payload_checksum(PACKAGE_ROOT) == (
        probe.EXPECTED_SCIENTIFIC_PAYLOAD_SHA256
    )
    source = TOOL_PATH.read_text(encoding="utf-8")
    for forbidden in (
        'split="train"',
        'split="test"',
        "model.fit(",
        "GradientTape(",
        "apply_gradients(",
        "--split",
        "--condition",
    ):
        assert forbidden not in source
    assert 'split="val"' in source
    assert "compile=False" in source
    assert '"optimizer_created": False' in source
    assert '"test_split_constructed": False' in source
