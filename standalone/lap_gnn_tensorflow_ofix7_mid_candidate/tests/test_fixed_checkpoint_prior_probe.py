from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from lap_gnn_tf.graph.batch import load_golden_batch
from lap_gnn_tf.model import LapGNN


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PACKAGE_ROOT / "tools" / "evaluate_fixed_checkpoint_prior_probe.py"
SPEC = importlib.util.spec_from_file_location("fixed_checkpoint_prior_probe", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def _batch() -> dict[str, tf.Tensor]:
    node_features = np.arange(5 * 37, dtype=np.float32).reshape(5, 37) + 1.0
    edge_features = np.arange(4 * 8, dtype=np.float32).reshape(4, 8) + 1.0
    return {
        "node_features": tf.constant(node_features),
        "edge_index": tf.constant(
            [[0, 1, 3, 4], [1, 2, 4, 3]], dtype=tf.int64
        ),
        "edge_features": tf.constant(edge_features),
        "node_types": tf.constant([0, 0, 1, 0, 1], dtype=tf.int8),
        "node_graph_index": tf.constant([0, 0, 0, 1, 1], dtype=tf.int64),
        "edge_graph_index": tf.constant([0, 0, 1, 1], dtype=tf.int64),
        "graph_node_counts": tf.constant([3, 2], dtype=tf.int64),
        "graph_edge_counts": tf.constant([2, 2], dtype=tf.int64),
        "labels": tf.constant([2, 5], dtype=tf.int64),
        "sample_ids": tf.constant([101, 202], dtype=tf.int64),
        "coordinates": tf.constant(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=tf.float32,
        ),
        "anchor_mask": tf.constant([False, False, True, False, True]),
        "part_soft": tf.constant(
            np.arange(5 * 13, dtype=np.float32).reshape(5, 13) + 0.25
        ),
        "face_mask": tf.constant([0.1, 0.2, 0.3, 0.4, 0.5], dtype=tf.float32),
        "valid_part_mask": tf.ones((2, 13), dtype=tf.float32),
        "valid_anchor_mask": tf.ones((2, 12), dtype=tf.float32),
        "detected": tf.constant([True, False]),
        "landmark_missing_flag": tf.constant([0, 1], dtype=tf.int64),
        "image_48": tf.reshape(
            tf.range(2 * 48 * 48, dtype=tf.float32), (2, 48, 48)
        ),
    }


def _snapshot(batch):
    return {name: np.array(tensor.numpy(), copy=True) for name, tensor in batch.items()}


def _assert_snapshot(batch, snapshot):
    assert set(batch) == set(snapshot)
    for name, tensor in batch.items():
        np.testing.assert_array_equal(tensor.numpy(), snapshot[name], err_msg=name)


def _real_config():
    return probe.load_config(
        PACKAGE_ROOT / "configs" / "fer2013_ofix7_mid_tensorflow_seed42.yaml"
    )


def _checkpoint_metadata(config):
    locked = config["locked"]
    return {
        "epoch": 31,
        "config_hash": probe.canonical_config_hash(config),
        "seed": config["seed"],
        "package_checksum": locked["package_checksum"],
        "execution_contract_sha256": locked["execution_contract_sha256"],
        "graph_signature": locked["graph_signature"],
        "feature_signature": locked["feature_signature"],
        "prior_signature": locked["prior_signature"],
        "dataset_split_signature": locked["dataset_split_signature"],
    }


def _issue7_persisted_resolved_config():
    config = _real_config()
    config["data"]["fer_csv"] = (
        "/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split/train.csv"
    )
    config["data"]["prior_dir"] = (
        "/kaggle/input/datasets/irthn1311/"
        "d16-mediapipe-pixel-priors-best-retry-rescue/outputs/"
        "d16_mediapipe_pixel_priors_best_retry_rescue"
    )
    config["resources"]["clean_graph_cache_dir"] = (
        "/kaggle/input/datasets/irthn1311/ofix7-mid-seed42-records"
    )
    config["resources"]["eval_batch_size"] = 32
    return config


def test_c0_is_identity_and_does_not_mutate_source():
    source = _batch()
    before = _snapshot(source)

    transformed = probe.apply_intervention(source, probe.CONDITION_OFFICIAL)
    integrity = probe.validate_intervention_integrity(
        source, transformed, probe.CONDITION_OFFICIAL
    )

    assert transformed is not source
    assert integrity["changed_tensor_fields"] == []
    for name in source:
        assert transformed[name] is source[name]
        np.testing.assert_array_equal(transformed[name].numpy(), before[name])
    _assert_snapshot(source, before)


def test_c1_changes_only_direct_part_fields():
    source = _batch()
    before = _snapshot(source)

    transformed = probe.apply_intervention(source, probe.CONDITION_DIRECT_ZERO)
    integrity = probe.validate_intervention_integrity(
        source, transformed, probe.CONDITION_DIRECT_ZERO
    )

    assert set(integrity["changed_tensor_fields"]) == {
        "part_soft",
        "valid_part_mask",
    }
    np.testing.assert_array_equal(
        transformed["part_soft"].numpy(), np.zeros_like(before["part_soft"])
    )
    np.testing.assert_array_equal(
        transformed["valid_part_mask"].numpy(),
        np.zeros_like(before["valid_part_mask"]),
    )
    for name in source:
        if name not in {"part_soft", "valid_part_mask"}:
            assert transformed[name] is source[name]
            np.testing.assert_array_equal(transformed[name].numpy(), before[name])
    _assert_snapshot(source, before)


def test_c2_zeros_exact_semantic_columns_and_preserves_visual_columns():
    source = _batch()
    before = _snapshot(source)

    transformed = probe.apply_intervention(source, probe.CONDITION_SEMANTIC_ZERO)
    integrity = probe.validate_intervention_integrity(
        source, transformed, probe.CONDITION_SEMANTIC_ZERO
    )

    assert set(integrity["changed_tensor_fields"]) == {
        "part_soft",
        "valid_part_mask",
        "node_features",
        "edge_features",
    }
    np.testing.assert_array_equal(
        transformed["node_features"][:, 5:32].numpy(),
        np.zeros_like(before["node_features"][:, 5:32]),
    )
    np.testing.assert_array_equal(
        transformed["node_features"][:, 0:5].numpy(), before["node_features"][:, 0:5]
    )
    np.testing.assert_array_equal(
        transformed["node_features"][:, 32:37].numpy(),
        before["node_features"][:, 32:37],
    )
    np.testing.assert_array_equal(
        transformed["edge_features"][:, 6:8].numpy(),
        np.zeros_like(before["edge_features"][:, 6:8]),
    )
    np.testing.assert_array_equal(
        transformed["edge_features"][:, 0:6].numpy(), before["edge_features"][:, 0:6]
    )
    _assert_snapshot(source, before)


@pytest.mark.parametrize("condition", probe.CONDITIONS)
def test_all_conditions_preserve_topology_and_sample_identity(condition):
    source = _batch()
    transformed = probe.apply_intervention(source, condition)
    probe.validate_intervention_integrity(source, transformed, condition)

    for name in (
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
    ):
        assert transformed[name] is source[name]
        np.testing.assert_array_equal(transformed[name].numpy(), source[name].numpy())


@pytest.mark.parametrize("condition", probe.CONDITIONS)
def test_transformations_are_deterministic_dtype_stable_and_non_mutating(condition):
    source = _batch()
    before = _snapshot(source)
    first = probe.apply_intervention(source, condition)
    second = probe.apply_intervention(source, condition)

    for name in source:
        assert first[name].shape == source[name].shape
        assert first[name].dtype == source[name].dtype
        np.testing.assert_array_equal(first[name].numpy(), second[name].numpy())
    _assert_snapshot(source, before)


def test_batch_schema_is_exact_37_node_8_edge_13_part_channels():
    dimensions = probe.validate_batch_schema(_batch())
    assert dimensions == {
        "graphs": 2,
        "nodes": 5,
        "edges": 4,
        "node_width": 37,
        "edge_width": 8,
        "part_width": 13,
    }

    wrong = _batch()
    wrong["node_features"] = wrong["node_features"][:, :36]
    with pytest.raises(probe.PriorProbeError, match="shape drift"):
        probe.validate_batch_schema(wrong)

    missing = _batch()
    del missing["sample_ids"]
    with pytest.raises(probe.PriorProbeError, match="fields drift"):
        probe.validate_batch_schema(missing)


def test_all_conditions_are_compatible_with_frozen_model_on_golden_batch():
    batch = load_golden_batch(
        str(PACKAGE_ROOT / "validation_assets" / "golden" / "graph_batch.npz")
    )
    model = LapGNN()

    for condition in probe.CONDITIONS:
        transformed = probe.apply_intervention(batch, condition)
        probe.validate_intervention_integrity(batch, transformed, condition)
        output = model(transformed, training=False)
        assert tuple(output["logits"].shape) == (int(batch["labels"].shape[0]), 7)
        assert tuple(output["probabilities"].shape) == (
            int(batch["labels"].shape[0]),
            7,
        )


def test_integrity_check_rejects_topology_mutation():
    source = _batch()
    transformed = probe.apply_intervention(source, probe.CONDITION_DIRECT_ZERO)
    transformed["edge_index"] = tf.reverse(transformed["edge_index"], axis=[1])

    with pytest.raises(probe.PriorProbeError, match="edge_index"):
        probe.validate_intervention_integrity(
            source, transformed, probe.CONDITION_DIRECT_ZERO
        )


def test_unknown_condition_fails_closed():
    with pytest.raises(probe.PriorProbeError, match="Unknown"):
        probe.apply_intervention(_batch(), "custom_intervention")


def test_frozen_contract_and_scientific_payload_guard(monkeypatch):
    config = _real_config()
    contract = probe.validate_frozen_contract(config)
    assert contract["node_width"] == 37
    assert contract["edge_width"] == 8
    assert contract["part_width"] == 13
    assert contract["scientific_payload_sha256"] == (
        probe.EXPECTED_SCIENTIFIC_PAYLOAD_SHA256
    )

    drifted = json.loads(json.dumps(config))
    drifted["locked"]["feature_signature"] = "0" * 64
    with pytest.raises(probe.PriorProbeError, match="signature drift"):
        probe.validate_frozen_contract(drifted)

    monkeypatch.setattr(probe, "NODE_FEATURE_NAMES", ["drifted"])
    with pytest.raises(probe.PriorProbeError, match="node feature order"):
        probe.validate_frozen_contract(config)


def test_checkpoint_metadata_cross_check_passes_and_fails_closed():
    config = _real_config()
    metadata = _checkpoint_metadata(config)
    result = probe.validate_checkpoint_metadata(config, metadata)
    assert result["checkpoint_epoch"] == 31
    assert "config_hash" in result["cross_checked_fields"]

    metadata["prior_signature"] = "wrong"
    with pytest.raises(probe.PriorProbeError, match="metadata mismatch"):
        probe.validate_checkpoint_metadata(config, metadata)


def test_issue7_persisted_config_identity_precedes_isolated_runtime_mutation(
    tmp_path,
):
    raw_config = _issue7_persisted_resolved_config()
    assert probe.canonical_config_hash(raw_config) == (
        "a4038682bf09c03786e86119001cf5f81ac5fec25d09062e6a0866484c32a3cf"
    )

    resolved = tmp_path / "resolved_config.json"
    resolved.write_text(
        json.dumps(raw_config, indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
        newline="",
    )
    assert probe.sha256_file(resolved) == (
        "3c028dd2f32ebed3a252544e170220b150b5e29920cea865924dddce6aef5a32"
    )
    persisted_identity = probe.load_persisted_resolved_config(resolved)
    metadata = _checkpoint_metadata(persisted_identity)
    assert probe.validate_checkpoint_metadata(
        persisted_identity, metadata
    )["checkpoint_epoch"] == 31

    legacy_yaml_loaded = probe.load_config(resolved)
    assert probe.canonical_config_hash(legacy_yaml_loaded) == (
        "916472d6e813b15dca6e7cd5016e87e95fad5d608bc1e341117675797cce8e54"
    )
    with pytest.raises(probe.PriorProbeError, match="config_hash"):
        probe.validate_checkpoint_metadata(legacy_yaml_loaded, metadata)

    runtime_config = probe.build_runtime_config(persisted_identity)
    assert runtime_config == persisted_identity
    assert runtime_config is not persisted_identity
    assert runtime_config["resources"] is not persisted_identity["resources"]
    runtime_config["resources"]["eval_batch_size"] = 64
    assert persisted_identity["resources"]["eval_batch_size"] == 32
    assert probe.validate_checkpoint_metadata(
        persisted_identity, metadata
    )["checkpoint_epoch"] == 31


def test_metadata_inference_is_unambiguous():
    checkpoint = Path("run/checkpoints/best_val_accuracy.keras")
    assert probe.infer_checkpoint_metadata_path(checkpoint) == Path(
        "run/checkpoints/best_val_accuracy.metadata.json"
    )


def test_checkpoint_load_is_keras_compile_false_only(tmp_path, monkeypatch):
    checkpoint = tmp_path / "fixed.keras"
    checkpoint.write_bytes(b"immutable-checkpoint")
    calls = []

    class FakeModel:
        optimizer = None
        weights = []

        @staticmethod
        def count_params():
            return probe.EXPECTED_PARAMETER_COUNT

    def fake_load(path, **kwargs):
        calls.append((Path(path), kwargs))
        return FakeModel()

    monkeypatch.setattr(probe.tf.keras.models, "load_model", fake_load)
    loaded = probe.load_fixed_checkpoint(checkpoint)
    assert isinstance(loaded, FakeModel)
    assert len(calls) == 1
    assert calls[0][0] == checkpoint
    assert calls[0][1]["compile"] is False
    assert calls[0][1]["custom_objects"] == {
        "LapGNN": probe.LapGNN,
        "lap_gnn_tf>LapGNN": probe.LapGNN,
    }

    wrong = tmp_path / "weights.h5"
    wrong.write_bytes(b"weights")
    with pytest.raises(probe.PriorProbeError, match=r"\.keras"):
        probe.load_fixed_checkpoint(wrong)


def test_conditions_are_evaluated_as_a_paired_set_per_original_batch(monkeypatch):
    calls = []

    def fake_step(batch):
        part_zero = bool(tf.reduce_all(batch["part_soft"] == 0).numpy())
        semantic_zero = bool(
            tf.reduce_all(batch["node_features"][:, 5:32] == 0).numpy()
        )
        if not part_zero:
            condition = probe.CONDITION_OFFICIAL
            predicted_class = 2
        elif not semantic_zero:
            condition = probe.CONDITION_DIRECT_ZERO
            predicted_class = 3
        else:
            condition = probe.CONDITION_SEMANTIC_ZERO
            predicted_class = 4
        calls.append(condition)
        probabilities = tf.one_hot(
            [predicted_class, predicted_class], depth=7, dtype=tf.float32
        )
        return tf.constant(1.0, tf.float32), probabilities

    monkeypatch.setattr(probe, "build_compiled_evaluation_step", lambda _model: fake_step)
    second_batch = _batch()
    second_batch["sample_ids"] = tf.constant([303, 404], dtype=tf.int64)
    result = probe.evaluate_conditions(object(), [_batch(), second_batch])

    assert calls == list(probe.CONDITIONS) * 2
    assert result["batch_count"] == 2
    assert result["sample_count"] == 4
    assert result["integrity_counts"] == {condition: 2 for condition in probe.CONDITIONS}


def test_output_contract_is_compact_paired_and_validation_only(tmp_path, monkeypatch):
    source = _batch()
    labels = source["labels"].numpy()
    sample_ids = source["sample_ids"].numpy()
    probabilities = {
        condition: np.eye(7, dtype=np.float64)[labels]
        for condition in probe.CONDITIONS
    }
    result = {
        "batch_count": 1,
        "sample_count": 2,
        "labels": labels,
        "sample_ids": sample_ids,
        "probabilities": probabilities,
        "metrics": {
            condition: {"accuracy": 1.0, "macro_f1": 1.0, "loss": 0.1}
            for condition in probe.CONDITIONS
        },
        "integrity_counts": {condition: 1 for condition in probe.CONDITIONS},
    }
    checkpoint = tmp_path / "fixed.keras"
    metadata = tmp_path / "fixed.metadata.json"
    config = tmp_path / "resolved_config.json"
    checkpoint.write_bytes(b"checkpoint")
    metadata.write_text("{}", encoding="utf-8")
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(probe, "environment_manifest", lambda: {"tensorflow": "test"})
    output = tmp_path / "probe-output"

    manifest = probe.write_probe_outputs(
        output,
        result=result,
        checkpoint_path=checkpoint,
        checkpoint_metadata_path=metadata,
        checkpoint_sha256_before="a" * 64,
        checkpoint_sha256_after="a" * 64,
        model_weights_sha256_before="b" * 64,
        model_weights_sha256_after="b" * 64,
        resolved_config_path=config,
        resolved_config_sha256="c" * 64,
        contract={"scientific_payload_sha256": probe.EXPECTED_SCIENTIFIC_PAYLOAD_SHA256},
        checkpoint_cross_check={"checkpoint_epoch": 31},
        resources={"dataset_split": "val", "shuffle": False},
        bounded_limit=1,
    )

    assert manifest["split"] == "val"
    assert manifest["validation_only"] is True
    assert manifest["topology_fixed"] is True
    assert manifest["checkpoint"]["unchanged"] is True
    assert manifest["checkpoint"]["model_weights_unchanged"] is True
    assert manifest["test_access"] == {
        "test_split_constructed": False,
        "test_metrics_created": False,
        "test_predictions_created": False,
        "test_inference_run": False,
    }
    assert sorted(path.name for path in output.iterdir()) == sorted(
        [
            "intervention_integrity.json",
            "paired_validation_predictions.csv",
            "probe_manifest.json",
            *(f"validation_metrics_{condition}.json" for condition in probe.CONDITIONS),
        ]
    )
    with (output / "paired_validation_predictions.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["sample_id"]) for row in rows] == [101, 202]
    assert all("official_probability_6" in row for row in rows)
    integrity = json.loads(
        (output / "intervention_integrity.json").read_text(encoding="utf-8")
    )
    assert integrity["source_batches_mutated"] is False
    assert integrity["test_split_constructed"] is False


def test_output_root_must_be_fresh(tmp_path):
    output = tmp_path / "exists"
    output.mkdir()
    with pytest.raises(FileExistsError, match="must not already exist"):
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
            bounded_limit=None,
        )


def test_main_constructs_only_paired_validation_and_rechecks_checkpoint(
    tmp_path, monkeypatch, capsys
):
    checkpoint = tmp_path / "fixed.keras"
    checkpoint.write_bytes(b"immutable-checkpoint")
    config = _real_config()
    resolved = tmp_path / "resolved_config.json"
    resolved.write_text(json.dumps(config), encoding="utf-8")
    persisted_config = probe.load_persisted_resolved_config(resolved)
    metadata = tmp_path / "fixed.metadata.json"
    metadata.write_text(
        json.dumps(_checkpoint_metadata(persisted_config)), encoding="utf-8"
    )
    prior_root = tmp_path / "priors"
    cache_root = tmp_path / "cache"
    prior_root.mkdir()
    cache_root.mkdir()
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

    def fake_load(path):
        events.append(("load", Path(path)))
        return FakeModel()

    def fake_evaluate(model, batches):
        assert isinstance(model, FakeModel)
        assert list(batches) == []
        events.append(("evaluate", model))
        return {"batch_count": 1, "sample_count": 1}

    captured = {}

    def fake_write(output_root, **kwargs):
        captured.update(kwargs)
        events.append(("write", Path(output_root)))
        return {"split": "val", "validation_only": True}

    monkeypatch.setattr(probe, "GraphBatchGenerator", FakeGenerator)
    monkeypatch.setattr(probe, "load_fixed_checkpoint", fake_load)
    monkeypatch.setattr(probe, "evaluate_conditions", fake_evaluate)
    monkeypatch.setattr(probe, "model_weights_sha256", lambda _model: "w" * 64)
    monkeypatch.setattr(probe, "write_probe_outputs", fake_write)
    monkeypatch.setattr(probe.tf.config, "list_physical_devices", lambda _kind: [])

    assert probe.main(
        [
            "--checkpoint", str(checkpoint),
            "--checkpoint-metadata", str(metadata),
            "--resolved-config", str(resolved),
            "--prior-root", str(prior_root),
            "--clean-graph-cache-dir", str(cache_root),
            "--output-root", str(output),
            "--limit-val-batches", "2",
        ]
    ) == 0

    generator_kwargs = next(event[1] for event in events if event[0] == "generator")
    assert generator_kwargs["split"] == "val"
    assert generator_kwargs["shuffle"] is False
    assert generator_kwargs["prior_root"] == prior_root
    assert generator_kwargs["clean_graph_cache_dir"] == cache_root
    assert events.count(("load", checkpoint)) == 1
    assert ("iter_epoch", 0, 2) in events
    expected_checkpoint_hash = probe.sha256_file(checkpoint)
    assert captured["checkpoint_sha256_before"] == expected_checkpoint_hash
    assert captured["checkpoint_sha256_after"] == expected_checkpoint_hash
    assert captured["model_weights_sha256_before"] == "w" * 64
    assert captured["model_weights_sha256_after"] == "w" * 64
    assert captured["bounded_limit"] == 2
    assert '"validation_only": true' in capsys.readouterr().out


def test_cli_is_narrow_validation_only_and_has_no_split_or_intervention_selector():
    destinations = {action.dest for action in probe.build_parser()._actions}
    assert {
        "checkpoint",
        "checkpoint_metadata",
        "resolved_config",
        "prior_root",
        "clean_graph_cache_dir",
        "output_root",
        "eval_batch_size",
        "graph_workers",
        "graph_cache_size",
        "limit_val_batches",
    } <= destinations
    assert "split" not in destinations
    assert "condition" not in destinations
    assert "intervention" not in destinations


def test_tool_has_no_training_test_lifecycle_or_raw_prior_corruption_path():
    source = TOOL_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "GradientTape",
        ".fit(",
        "run_training",
        "lap_gnn_tf.cli.train",
        "_zero_prior",
        "shuffle_prior",
        "forced_fallback",
        "attenuate_prior",
        'split="test"',
        '"--split"',
    ):
        assert forbidden not in source
    assert 'split="val"' in source
    assert "compile=False" in source
    assert source.count("tf.keras.models.load_model(") == 1


def test_scientific_payload_remains_exactly_frozen():
    assert probe.scientific_payload_checksum(PACKAGE_ROOT) == (
        "286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e"
    )
