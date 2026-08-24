from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from _helpers import loaded


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PACKAGE_ROOT / "tools" / "evaluate_gate_a_forensic_probe.py"
SPEC = importlib.util.spec_from_file_location("gate_a_forensic_probe", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class FakeForensicModel(tf.keras.Model):
    def __init__(self):
        super().__init__(dtype="float32", name="fake_forensic_model")
        self.marker = self.add_weight(
            name="immutable_marker", shape=(), initializer="zeros", trainable=True
        )
        self.native_call_count = 0

    def call(self, batch, training=False):
        assert training is False
        self.native_call_count += 1
        return _output_for_batch(batch, self.marker)


def _output_for_batch(batch, marker=0.0):
    sample_count = tf.shape(batch["labels"])[0]
    base = tf.constant(
        [[0.7, 0.2, 0.1, -0.1, -0.2, -0.3, -0.4]], dtype=tf.float32
    )
    logits = tf.tile(base, [sample_count, 1]) + tf.cast(marker, tf.float32) * 0.0
    return {
        "logits": logits,
        "probabilities": tf.nn.softmax(logits, axis=-1),
    }


@pytest.fixture()
def golden_batch():
    _, batch = loaded()
    return {name: tf.identity(value) for name, value in batch.items()}


def _manual_stub(calls, *, mutate_weights=False, fail_on_call=None):
    def manual(model, batch, condition):
        calls.append(condition)
        if fail_on_call is not None and len(calls) == fail_on_call:
            raise RuntimeError("synthetic late diagnostic failure")
        output = _output_for_batch(batch, model.marker)
        if mutate_weights:
            model.marker.assign_add(1.0)
        trace = {
            "model_boundary": {
                "autocast": bool(model.autocast),
                "input_dtype": tf.dtypes.as_dtype(model.input_dtype).name,
                "source_dtypes": {
                    name: value.dtype.name for name, value in batch.items()
                },
                "effective_dtypes": {
                    name: value.dtype.name for name, value in batch.items()
                },
            }
        }
        return output, trace

    return manual


def _second_unique_batch(batch):
    result = {name: tf.identity(value) for name, value in batch.items()}
    result["sample_ids"] = tf.cast(result["sample_ids"], tf.int64) + 10000
    return result


def test_only_native_and_manual_d0_are_executed_and_batches_stay_unchanged(
    tmp_path, monkeypatch, golden_batch
):
    model = FakeForensicModel()
    calls = []
    monkeypatch.setattr(probe.step7, "manual_forward", _manual_stub(calls))
    monkeypatch.setattr(
        probe,
        "EXPECTED_FULL_VALIDATION_SAMPLES",
        int(tf.shape(golden_batch["labels"])[0].numpy()),
    )
    before = probe.step7._snapshot_batch(golden_batch)
    weights_before = probe.step6.model_weights_sha256(model)
    result = probe.evaluate_forensic_batches(
        model,
        [golden_batch],
        tmp_path / "forensic",
        expected_model_weights_sha256=weights_before,
    )

    assert model.native_call_count == 2
    assert calls == [probe.step7.CONDITION_D0, probe.step7.CONDITION_D0]
    assert result["completed_batch_count"] == 1
    assert result["status"] == "COMPLETE"
    probe.step7._assert_source_unchanged(golden_batch, before)
    batch_payload = json.loads(
        (tmp_path / "forensic/batches/batch_00000.json").read_text()
    )
    assert batch_payload["executed_paths"] == ["native", "manual_d0"]
    assert batch_payload["intervention_conditions_executed"] == []
    assert set(batch_payload["comparisons"]) == set(probe.COMPARISON_ORDER)


def test_incremental_batch_evidence_survives_later_failure(
    tmp_path, monkeypatch, golden_batch
):
    model = FakeForensicModel()
    calls = []
    monkeypatch.setattr(
        probe.step7,
        "manual_forward",
        _manual_stub(calls, fail_on_call=3),
    )
    output_root = tmp_path / "forensic"
    with pytest.raises(RuntimeError, match="synthetic late diagnostic failure"):
        probe.evaluate_forensic_batches(
            model,
            [golden_batch, _second_unique_batch(golden_batch)],
            output_root,
            expected_model_weights_sha256=probe.step6.model_weights_sha256(model),
        )

    assert (output_root / "batches/batch_00000.json").is_file()
    assert not (output_root / "batches/batch_00001.json").exists()
    progress = json.loads((output_root / "progress.json").read_text())
    assert progress["completed_batch_count"] == 1
    assert progress["completed_batch_indices"] == [0]


def test_model_weight_mutation_fails_closed_after_persisting_batch(
    tmp_path, monkeypatch, golden_batch
):
    model = FakeForensicModel()
    calls = []
    monkeypatch.setattr(
        probe.step7,
        "manual_forward",
        _manual_stub(calls, mutate_weights=True),
    )
    output_root = tmp_path / "forensic"
    with pytest.raises(probe.GateAForensicError, match="Model weights changed"):
        probe.evaluate_forensic_batches(
            model,
            [golden_batch],
            output_root,
            expected_model_weights_sha256=probe.step6.model_weights_sha256(model),
        )
    assert (output_root / "batches/batch_00000.json").is_file()


def test_checkpoint_and_model_immutability_are_exact(tmp_path):
    checkpoint = tmp_path / "checkpoint.keras"
    checkpoint.write_bytes(b"immutable checkpoint")
    model = FakeForensicModel()
    checkpoint_before = probe.sha256_file(checkpoint)
    weights_before = probe.step6.model_weights_sha256(model)

    intact = probe.build_immutability_evidence(
        checkpoint,
        checkpoint_before,
        model,
        weights_before,
        source_batches_unchanged=True,
    )
    assert intact["checkpoint_unchanged"] is True
    assert intact["model_weights_unchanged"] is True

    checkpoint.write_bytes(b"changed checkpoint")
    model.marker.assign_add(1.0)
    changed = probe.build_immutability_evidence(
        checkpoint,
        checkpoint_before,
        model,
        weights_before,
        source_batches_unchanged=True,
    )
    assert changed["checkpoint_unchanged"] is False
    assert changed["model_weights_unchanged"] is False


def test_dtype_manifest_covers_required_model_boundaries():
    model, _ = loaded()
    manifest = probe.build_dtype_manifest(model)
    roles = [item["role"] for item in manifest["layers"]]
    assert manifest["op_determinism_enabled_by_tool"] is False
    for required in (
        "outer_lap_gnn",
        "encoder",
        "gnn_container",
        "gnn_layer",
        "part_global_context",
        "readout",
        "classifier",
    ):
        assert required in roles
    assert roles.count("gnn_layer") == len(model.gnn.layers_)
    for item in manifest["layers"]:
        for field in (
            "dtype_policy",
            "compute_dtype",
            "variable_dtype",
            "autocast",
        ):
            assert field in item


def test_static_contract_has_no_intervention_or_test_training_execution():
    source = TOOL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    accessed_attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    for forbidden in (
        "CONDITION_D1",
        "CONDITION_D2",
        "CONDITION_D3",
        "CONDITION_D4",
        "CONDITION_D5",
        "enable_op_determinism",
        "fit",
        "apply_gradients",
    ):
        assert forbidden not in accessed_attributes
    assert "step7.CONDITION_D0" in source
    assert 'split="val"' in source
    assert '"dataset_split": "val"' in source
    assert '"graph_rebuild_allowed": False' in source
    assert '"intervention_conditions_executed": []' in source
    assert '"test_split_constructed": False' in source
    assert '"optimizer_created": False' in source
    assert "--limit-val-batches" not in source


def test_exact_frozen_identity_constants_and_reviewed_tool_hashes():
    assert probe.EXPECTED_SCIENTIFIC_BASE_COMMIT == (
        "d14e7e1e3eec2ffbd5339b5a3bd0d5db5ab3de8b"
    )
    assert probe.EXPECTED_HOTFIX_ANCESTOR_COMMIT == (
        "a1b1d279bb9ec388f1d93ad86196e423dc750ad1"
    )
    assert probe.sha256_file(probe.STEP7_TOOL_PATH) == probe.EXPECTED_STEP7_TOOL_SHA256
    assert probe.sha256_file(probe.STEP6_SUPPORT_PATH) == probe.EXPECTED_STEP6_SUPPORT_SHA256
    assert probe.EXPECTED_SCIENTIFIC_PAYLOAD_SHA256 == (
        "286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e"
    )
    assert probe.EXPECTED_CHECKPOINT_SHA256 == (
        "9ec11bb819f97e4fbda432f68da76c1201b8a3f9e06fae9eb30489a528d6ac16"
    )
    assert probe.EXPECTED_CHECKPOINT_METADATA_SHA256 == (
        "e62cf8c86f0d6a56c3041911de6397d18f47276f79f03c6a50ca71fa47300a37"
    )
    assert probe.EXPECTED_RESOLVED_CONFIG_SHA256 == (
        "3c028dd2f32ebed3a252544e170220b150b5e29920cea865924dddce6aef5a32"
    )
