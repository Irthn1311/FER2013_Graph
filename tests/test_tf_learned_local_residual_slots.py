from __future__ import annotations

import inspect
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "standalone" / "lap_gnn_tensorflow_ofix7_mid_candidate"
PACKAGE_SRC = PACKAGE_ROOT / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from lap_gnn_tf.conversion import load_pytorch_npz  # noqa: E402
from lap_gnn_tf.graph.batch import load_golden_batch  # noqa: E402
from lap_gnn_tf.model import build_model  # noqa: E402
from lap_gnn_tf.model.motif_layers import PART_ORDER, part_pool  # noqa: E402
from lap_gnn_tf.model.readout import MicroMotifSupportReadout  # noqa: E402
from lap_gnn_tf.signatures import scientific_payload_checksum  # noqa: E402
from research.candidates.tf_learned_local_residual_slots import (  # noqa: E402
    LearnedLocalResidualSlotLapGNN,
    LearnedLocalResidualSlotPool,
    build_candidate_model,
)


IMPLEMENTATION_BASE = "4c7d88e6d03f4aa35f657d7ade69f2436c3b89cd"
EXPECTED_PAYLOAD = "286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e"
BASELINE_PARAMETERS = 1_061_192
CANDIDATE_PARAMETERS = 1_061_576
PARAMETER_DELTA = 384
GOLDEN = PACKAGE_ROOT / "validation_assets" / "golden"
MODEL_PATH = (
    ROOT / "research/candidates/tf_learned_local_residual_slots/model.py"
)


@pytest.fixture(scope="module")
def golden_models():
    batch = load_golden_batch(str(GOLDEN / "graph_batch.npz"))
    baseline = build_model(batch)
    candidate = build_candidate_model(batch)
    baseline_mapping = load_pytorch_npz(baseline, GOLDEN / "model_state.npz")
    candidate_mapping = load_pytorch_npz(candidate, GOLDEN / "model_state.npz")
    return {
        "batch": batch,
        "baseline": baseline,
        "candidate": candidate,
        "baseline_mapping": baseline_mapping,
        "candidate_mapping": candidate_mapping,
    }


def _simple_slot_inputs():
    values = tf.reshape(tf.range(5 * 96, dtype=tf.float32), (5, 96)) / 100.0
    graph_index = tf.constant([0, 0, 1, 1, 1], dtype=tf.int32)
    return values, graph_index, tf.constant(2, dtype=tf.int32)


def _assign_deterministic_queries(layer):
    values = tf.reshape(
        tf.linspace(-0.04, 0.04, 4 * 96), (4, 96)
    )
    layer.Q.assign(values)


def test_exact_base_and_frozen_baseline_tree_are_unchanged():
    assert subprocess.run(
        ["git", "cat-file", "-e", f"{IMPLEMENTATION_BASE}^{{commit}}"],
        cwd=ROOT,
        check=False,
    ).returncode == 0
    relative_package = PACKAGE_ROOT.relative_to(ROOT).as_posix()
    changed = subprocess.run(
        ["git", "diff", "--name-only", IMPLEMENTATION_BASE, "--", relative_package],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    assert changed == ""
    assert scientific_payload_checksum(PACKAGE_ROOT) == EXPECTED_PAYLOAD


def test_exact_parameter_delta_is_only_one_float32_query_matrix(golden_models):
    baseline = golden_models["baseline"]
    candidate = golden_models["candidate"]
    slot_pool = candidate.learned_local_residual_slots

    assert baseline.count_params() == BASELINE_PARAMETERS
    assert candidate.count_params() == CANDIDATE_PARAMETERS
    assert candidate.count_params() - baseline.count_params() == PARAMETER_DELTA
    assert len(slot_pool.trainable_variables) == 1
    assert slot_pool.trainable_variables[0] is slot_pool.Q
    assert tuple(slot_pool.Q.shape) == (4, 96)
    assert slot_pool.Q.dtype == "float32"
    assert int(slot_pool.Q.shape.num_elements()) == PARAMETER_DELTA
    assert golden_models["baseline_mapping"]["complete"] is True
    assert golden_models["candidate_mapping"]["complete"] is True

    source = inspect.getsource(LearnedLocalResidualSlotPool)
    assert "RandomNormal(stddev=0.02)" in source
    for forbidden in (
        "Dense(",
        "bias",
        "regularizer",
        "orthogonal",
        "auxiliary",
    ):
        assert forbidden not in source


def test_slot_pool_has_only_registered_nonsemantic_inputs():
    parameters = tuple(inspect.signature(LearnedLocalResidualSlotPool.call).parameters)
    assert parameters == ("self", "h", "node_graph_index", "num_graphs")
    source = inspect.getsource(LearnedLocalResidualSlotPool)
    for forbidden in (
        "part_soft",
        "valid_part_mask",
        "GROUP_INDICES",
        "mouth",
        "eye",
        "brow",
        "nose_cheek",
        "edge_features",
        "node_features",
        "landmark",
    ):
        assert forbidden not in source


def test_attention_is_graph_local_normalized_and_diagnostics_have_exact_shapes():
    layer = LearnedLocalResidualSlotPool()
    _assign_deterministic_queries(layer)
    h, graph_index, num_graphs = _simple_slot_inputs()
    result = layer(h, graph_index, num_graphs)
    mass = tf.math.unsorted_segment_sum(
        result["attention_weights"], graph_index, num_graphs
    )
    np.testing.assert_allclose(mass.numpy(), np.ones((2, 4)), rtol=0, atol=1e-6)
    assert result["slot_embeddings"].shape == (2, 4, 96)
    assert result["attention_weights"].shape == (5, 4)
    assert result["attention_entropy"].shape == (2, 4)
    assert result["attention_peak"].shape == (2, 4)


def test_cross_graph_isolation():
    layer = LearnedLocalResidualSlotPool()
    _assign_deterministic_queries(layer)
    h, graph_index, num_graphs = _simple_slot_inputs()
    before = layer(h, graph_index, num_graphs)["slot_embeddings"]
    perturbation = tf.where(
        tf.equal(graph_index[:, None], 1),
        tf.ones_like(h) * 1000.0,
        tf.zeros_like(h),
    )
    after = layer(h + perturbation, graph_index, num_graphs)["slot_embeddings"]
    np.testing.assert_array_equal(after.numpy()[0], before.numpy()[0])


def test_node_permutation_invariance_within_graph():
    layer = LearnedLocalResidualSlotPool()
    _assign_deterministic_queries(layer)
    h, graph_index, num_graphs = _simple_slot_inputs()
    permutation = tf.constant([1, 0, 4, 2, 3], dtype=tf.int32)
    expected = layer(h, graph_index, num_graphs)["slot_embeddings"]
    actual = layer(
        tf.gather(h, permutation),
        tf.gather(graph_index, permutation),
        num_graphs,
    )["slot_embeddings"]
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=0, atol=1e-6)


def test_official_global_residual_identity_and_registered_residual_order(golden_models):
    candidate = golden_models["candidate"]
    batch = golden_models["batch"]
    output = candidate(batch, training=False)
    official, _ = part_pool(
        output["node_embeddings"],
        batch["part_soft"],
        tf.cast(batch["node_graph_index"], tf.int32),
        batch["valid_part_mask"],
        tf.shape(batch["labels"])[0],
    )
    np.testing.assert_array_equal(
        output["official_global_residual"].numpy(), official["global"].numpy()
    )
    assert output["residual_stack"].shape == (8, 5, 96)
    assert output["residual_flat"].shape == (8, 480)
    np.testing.assert_array_equal(
        output["residual_stack"][:, :4, :].numpy(),
        output["learned_local_residual_slots"].numpy(),
    )
    np.testing.assert_array_equal(
        output["residual_stack"][:, 4, :].numpy(),
        output["official_global_residual"].numpy(),
    )


def test_learned_attention_never_enters_unchanged_support_branch(golden_models):
    candidate = golden_models["candidate"]
    batch = golden_models["batch"]
    assert type(candidate.readout) is MicroMotifSupportReadout
    original_q = candidate.learned_local_residual_slots.Q.numpy().copy()
    before = candidate(batch, training=False, collect_intermediates=True)
    try:
        candidate.learned_local_residual_slots.Q.assign(
            tf.ones((4, 96), dtype=tf.float32) * 20.0
        )
        after = candidate(batch, training=False, collect_intermediates=True)
    finally:
        candidate.learned_local_residual_slots.Q.assign(original_q)

    for name in (
        "micro_major_motif_tokens",
        "micro_major_motif_transformed_tokens",
        "micro_motif_tokens",
        "micro_motif_transformed_tokens",
        "micro_support_gate",
    ):
        np.testing.assert_array_equal(
            after["intermediates"][name].numpy(),
            before["intermediates"][name].numpy(),
        )


def test_test_only_official_local_stub_reproduces_frozen_baseline(
    golden_models, monkeypatch
):
    baseline = golden_models["baseline"]
    candidate = golden_models["candidate"]
    batch = golden_models["batch"]

    def official_local_control(h, node_graph_index, num_graphs):
        pooled, _ = part_pool(
            h,
            batch["part_soft"],
            node_graph_index,
            batch["valid_part_mask"],
            num_graphs,
        )
        slots = tf.stack([pooled[name] for name in PART_ORDER[:4]], axis=1)
        return {
            "slot_embeddings": slots,
            "attention_weights": tf.zeros(
                (tf.shape(h)[0], 4), dtype=tf.float32
            ),
            "attention_entropy": tf.zeros((num_graphs, 4), dtype=tf.float32),
            "attention_peak": tf.zeros((num_graphs, 4), dtype=tf.float32),
        }

    monkeypatch.setattr(
        candidate.learned_local_residual_slots, "call", official_local_control
    )
    expected = baseline(batch, training=False, collect_intermediates=True)
    actual = candidate(batch, training=False, collect_intermediates=True)

    for name in ("logits", "probabilities", "z_image", "node_embeddings"):
        np.testing.assert_allclose(
            actual[name].numpy(), expected[name].numpy(), rtol=0, atol=1e-5
        )
    for name in (
        "micro_major_motif_tokens",
        "micro_major_motif_transformed_tokens",
        "micro_motif_tokens",
        "micro_motif_transformed_tokens",
        "micro_support_gate",
        "pooled_graph_embedding",
        "classifier_input",
    ):
        np.testing.assert_allclose(
            actual["intermediates"][name].numpy(),
            expected["intermediates"][name].numpy(),
            rtol=0,
            atol=1e-5,
        )


def test_candidate_serialization_preserves_query_matrix_exactly(tmp_path):
    batch = load_golden_batch(str(GOLDEN / "graph_batch.npz"))
    candidate = build_candidate_model(batch)
    load_pytorch_npz(candidate, GOLDEN / "model_state.npz")
    _assign_deterministic_queries(candidate.learned_local_residual_slots)
    expected_q = candidate.learned_local_residual_slots.Q.numpy().copy()
    expected_logits = candidate(batch, training=False)["logits"].numpy()

    path = tmp_path / "learned_local_residual_slots.keras"
    candidate.save(path)
    restored = tf.keras.models.load_model(path, compile=False)
    assert isinstance(restored, LearnedLocalResidualSlotLapGNN)
    np.testing.assert_array_equal(
        restored.learned_local_residual_slots.Q.numpy(), expected_q
    )
    np.testing.assert_allclose(
        restored(batch, training=False)["logits"].numpy(),
        expected_logits,
        rtol=0,
        atol=1e-6,
    )


def test_candidate_has_no_public_mode_selector_or_training_test_lifecycle(golden_models):
    parameters = tuple(
        inspect.signature(LearnedLocalResidualSlotLapGNN.call).parameters
    )
    assert parameters == ("self", "batch", "training", "collect_intermediates")
    source = MODEL_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "model.fit(",
        "GradientTape(",
        "apply_gradients(",
        "optimizer=",
        'split="test"',
        "--residual-mode",
        "local_residual_override",
        "diversity_loss",
        "entropy_loss",
        "orthogonality_loss",
    ):
        assert forbidden not in source
    assert getattr(golden_models["candidate"], "optimizer", None) is None
