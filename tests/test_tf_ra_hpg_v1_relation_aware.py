from __future__ import annotations

import ast
import csv
import hashlib
import inspect
from pathlib import Path
import subprocess

import numpy as np
import pytest
import tensorflow as tf

from research.candidates.tf_cf_hpg_v1_2_tokenizer.model import (
    build_cf_hpg_v1_2_tokenizer,
)
from research.candidates.tf_ra_hpg_v1_relation_aware import data
from research.candidates.tf_ra_hpg_v1_relation_aware.graph import (
    HybridGraphBuilder,
    learned_relation_adjacency,
    normalized_grid_centers,
    spatial_8_neighbor_adjacency,
)
from research.candidates.tf_ra_hpg_v1_relation_aware.model import (
    RAHPGRelationAware,
    RelationAwareDynamicGraphBlock,
    build_ra_hpg_v1_relation_aware,
    patchify_and_scale,
    pool_2x2_mean,
    relation_aware_mean_aggregate,
    relative_edge_tensor,
)
from research.candidates.tf_ra_hpg_v1_relation_aware import (
    train_validation_only as training,
)


ROOT = Path(__file__).resolve().parents[1]
BASE = "8468dfe40ae2e6229c835b3356bb61eac1f22c77"
CANDIDATE = ROOT / "research/candidates/tf_ra_hpg_v1_relation_aware"
V1_2_CANDIDATE = ROOT / "research/candidates/tf_cf_hpg_v1_2_tokenizer"
ACCEPTED_CANDIDATES = (
    ROOT / "research/candidates/tf_cf_hpg",
    ROOT / "research/candidates/tf_cf_hpg_v1_1_resolution",
    V1_2_CANDIDATE,
    ROOT / "research/candidates/tf_cf_hpg_v1_3_multiscale_readout",
)
FROZEN_PATHS = (
    ROOT / "standalone/lap_gnn_tensorflow_ofix7_mid_candidate",
    ROOT / "research/candidates/tf_learned_local_residual_slots",
    ROOT / "tools/run_issue40_step13_execution.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_patchification_is_exact_144_by_16_row_major_and_scaled():
    image = tf.reshape(tf.range(48 * 48, dtype=tf.float32), [1, 48, 48, 1])
    patches = patchify_and_scale(image).numpy()
    assert patches.shape == (1, 144, 16)
    expected_first = image.numpy()[0, :4, :4, 0].reshape(-1) / 127.5 - 1.0
    expected_second = image.numpy()[0, :4, 4:8, 0].reshape(-1) / 127.5 - 1.0
    np.testing.assert_allclose(patches[0, 0], expected_first, atol=0.0)
    np.testing.assert_allclose(patches[0, 1], expected_second, atol=0.0)


def test_tokenizer_order_shapes_and_first_graph_input_match_v1_2(monkeypatch):
    parent = build_cf_hpg_v1_2_tokenizer()
    candidate = build_ra_hpg_v1_relation_aware()
    tokenizer_layers = (
        "raw_projection",
        "raw_local_refinement",
        "position_projection",
        "patch_norm",
    )
    for name in tokenizer_layers:
        source = getattr(parent, name)
        target = getattr(candidate, name)
        assert [tuple(w.shape) for w in target.weights] == [
            tuple(w.shape) for w in source.weights
        ]
        target.set_weights(source.get_weights())

    parent_inputs = []
    candidate_inputs = []
    parent_call = parent.fine_graph_builder.call
    candidate_call = candidate.fine_graph_builder.call

    def capture_parent(nodes):
        parent_inputs.append(tf.identity(nodes))
        return parent_call(nodes)

    def capture_candidate(nodes):
        candidate_inputs.append(tf.identity(nodes))
        return candidate_call(nodes)

    monkeypatch.setattr(parent.fine_graph_builder, "call", capture_parent)
    monkeypatch.setattr(candidate.fine_graph_builder, "call", capture_candidate)
    images = tf.random.stateless_uniform([1, 48, 48, 1], [55, 1], maxval=255.0)
    parent(images, training=False)
    candidate(images, training=False)
    assert len(parent_inputs) == 1
    assert len(candidate_inputs) == 2
    np.testing.assert_allclose(candidate_inputs[0], parent_inputs[0], atol=0.0)

    source = inspect.getsource(RAHPGRelationAware.call)
    ordered = (
        "nodes = self.raw_projection(patches)",
        "nodes = self.raw_local_activation(nodes)",
        "nodes = self.raw_local_refinement(nodes)",
        "nodes = nodes + self.position_projection(positions)",
        "nodes = self.patch_norm(nodes)",
        "nodes = self.patch_activation(nodes)",
        "nodes = self.patch_dropout(nodes, training=training)",
    )
    offsets = [source.index(fragment) for fragment in ordered]
    assert offsets == sorted(offsets)


def test_grid_coordinates_and_exact_hybrid_topology_are_frozen():
    coordinates = normalized_grid_centers(12).numpy()
    axis = (np.arange(12, dtype=np.float32) + 0.5) * (2.0 / 12.0) - 1.0
    expected = np.asarray([(x, y) for y in axis for x in axis], np.float32)
    np.testing.assert_allclose(coordinates, expected, atol=0.0)

    nodes = tf.random.stateless_normal([2, 144, 8], seed=[55, 2])
    spatial = spatial_8_neighbor_adjacency(12).numpy()
    learned = learned_relation_adjacency(nodes, 4).numpy()
    hybrid = HybridGraphBuilder(12, 4)(nodes).numpy()
    np.testing.assert_array_equal(hybrid, np.logical_or(spatial[None], learned))
    assert np.all(learned.sum(axis=-1) == 4)
    assert not np.diagonal(hybrid, axis1=1, axis2=2).any()


@pytest.mark.parametrize(
    ("builder_name", "block_name", "expected_nodes"),
    (("fine_graph_builder", "fine_blocks", 144), ("coarse_graph_builder", "coarse_blocks", 36)),
)
def test_each_graph_is_recomputed_immediately_before_each_block(
    monkeypatch, builder_name, block_name, expected_nodes
):
    model = build_ra_hpg_v1_relation_aware()
    builder = getattr(model, builder_name)
    blocks = getattr(model, block_name)
    original_builder_call = builder.call
    events = []
    adjacencies = []

    def counted_builder(nodes):
        adjacency = original_builder_call(nodes)
        events.append(("builder", id(nodes), id(adjacency), int(nodes.shape[1])))
        adjacencies.append(adjacency)
        return adjacency

    monkeypatch.setattr(builder, "call", counted_builder)
    for index, block in enumerate(blocks):
        original_block_call = block.call

        def counted_block(nodes, adjacency, training=False, return_debug=False, *, original=original_block_call, block_index=index):
            events.append(("block", id(nodes), id(adjacency), int(nodes.shape[1])))
            return original(nodes, adjacency, training=training, return_debug=return_debug)

        monkeypatch.setattr(block, "call", counted_block)

    model(tf.random.stateless_uniform([1, 48, 48, 1], [55, 3]), training=False)
    scale_events = [event for event in events if event[-1] == expected_nodes]
    assert [event[0] for event in scale_events] == ["builder", "block", "builder", "block"]
    assert len(adjacencies) == 2
    assert adjacencies[0] is not adjacencies[1]
    assert scale_events[0][1] == scale_events[1][1]
    assert scale_events[2][1] == scale_events[3][1]
    assert scale_events[1][2] == id(adjacencies[0])
    assert scale_events[3][2] == id(adjacencies[1])


def test_relative_tensor_orientation_is_exactly_z_j_minus_z_i():
    nodes = tf.constant([[[1.0, 4.0], [3.0, 2.0], [0.0, 8.0]]])
    relative = relative_edge_tensor(nodes).numpy()
    assert relative.shape == (1, 3, 3, 2)
    np.testing.assert_array_equal(relative[0, 0, 1], [2.0, -2.0])
    np.testing.assert_array_equal(relative[0, 1, 0], [-2.0, 2.0])


def test_relation_gate_shape_bounds_and_zero_gate_golden_weighted_mean():
    nodes = tf.constant([[[1.0, 4.0], [3.0, 2.0], [0.0, 8.0]]])
    adjacency = tf.constant(
        [[[False, True, True], [True, False, False], [True, True, False]]]
    )
    gate = tf.keras.layers.Dense(1, use_bias=True)
    gate(relative_edge_tensor(nodes))
    gate.set_weights(
        [np.zeros((2, 1), np.float32), np.zeros((1,), np.float32)]
    )
    message, gates, relative = relation_aware_mean_aggregate(nodes, adjacency, gate)
    assert gates.shape == (1, 3, 3, 1)
    assert np.all(gates.numpy() >= 0.0) and np.all(gates.numpy() <= 1.0)
    np.testing.assert_allclose(gates, 0.5, atol=0.0)
    expected = []
    for source in range(3):
        active = np.flatnonzero(adjacency.numpy()[0, source])
        expected.append(0.5 * relative.numpy()[0, source, active].mean(axis=0))
    np.testing.assert_allclose(message[0], expected, atol=0.0)


def test_inactive_and_self_edges_contribute_exactly_zero():
    nodes = tf.constant([[[0.0], [2.0], [1000.0]]])
    adjacency = tf.constant(
        [[[False, True, False], [True, False, False], [True, False, False]]]
    )
    gate = tf.keras.layers.Dense(1)
    gate(relative_edge_tensor(nodes))
    gate.set_weights(
        [np.zeros((1, 1), np.float32), np.zeros((1,), np.float32)]
    )
    message, _, _ = relation_aware_mean_aggregate(nodes, adjacency, gate)
    np.testing.assert_allclose(message.numpy()[0, :, 0], [1.0, -1.0, -500.0], atol=0.0)


@pytest.mark.parametrize("width", [96, 128])
def test_registered_graph_updater_and_four_x_ffn_shapes(width):
    block = RelationAwareDynamicGraphBlock(width)
    nodes = tf.random.stateless_normal([1, 5, width], [55, width])
    adjacency = tf.constant(
        [[[False, True, False, False, True],
          [True, False, True, False, False],
          [False, True, False, True, False],
          [False, False, True, False, True],
          [True, False, False, True, False]]]
    )
    output, debug = block(nodes, adjacency, return_debug=True)
    assert output.shape == nodes.shape
    assert debug["relation_gates"].shape == (1, 5, 5, 1)
    assert block.relation_gate.kernel.shape == (width, 1)
    assert block.relation_gate.bias.shape == (1,)
    assert block.graph_dense_in.kernel.shape == (2 * width, width)
    assert block.graph_dense_out.kernel.shape == (width, width)
    assert block.ffn_dense_in.kernel.shape == (width, 4 * width)
    assert block.ffn_dense_out.kernel.shape == (4 * width, width)


def test_hierarchy_and_coarse_only_readout_are_exactly_frozen():
    nodes = tf.reshape(tf.range(144, dtype=tf.float32), [1, 144, 1])
    pooled = pool_2x2_mean(nodes).numpy().reshape(6, 6)
    grid = np.arange(144, dtype=np.float32).reshape(12, 12)
    expected = np.asarray(
        [
            [grid[y : y + 2, x : x + 2].mean() for x in range(0, 12, 2)]
            for y in range(0, 12, 2)
        ]
    )
    np.testing.assert_allclose(pooled, expected, atol=0.0)
    model = build_ra_hpg_v1_relation_aware()
    logits, debug = model(tf.zeros([2, 48, 48, 1]), return_debug=True)
    assert logits.shape == (2, 7)
    assert debug["final_nodes"].shape == (2, 36, 128)
    assert debug["coarse_readout"].shape == (2, 256)
    assert debug["readout_hidden"].shape == (2, 128)
    assert model.readout_norm.gamma.shape == (256,)
    assert model.readout_dense.kernel.shape == (256, 128)
    source = inspect.getsource(RAHPGRelationAware.call)
    assert "fine_readout" not in source
    assert "fused_readout" not in source


def test_exact_block_inventory_and_model_identity():
    model = build_ra_hpg_v1_relation_aware()
    assert isinstance(model, RAHPGRelationAware)
    assert len(model.fine_blocks) == 2
    assert len(model.coarse_blocks) == 2
    assert all(
        isinstance(block, RelationAwareDynamicGraphBlock)
        for block in (*model.fine_blocks, *model.coarse_blocks)
    )
    assert model.count_params() == training.EXPECTED_PARAMETER_COUNT == 626_987
    assert sum(int(tf.size(value)) for value in model.trainable_variables) == 626_987
    assert len(model.trainable_variables) == training.EXPECTED_TRAINABLE_VARIABLE_COUNT == 74
    assert len(model.variables) == training.EXPECTED_KERAS_VARIABLE_COUNT == 84
    assert training.validate_model_identity(model) == {
        "parameters": 626_987,
        "trainable_variables": 74,
        "keras_variables": 84,
    }


def test_model_identity_fails_closed_for_the_v1_2_parent():
    with pytest.raises(training.ValidationOnlyHarnessError, match="identity drift"):
        training.validate_model_identity(build_cf_hpg_v1_2_tokenizer())


def test_no_forbidden_architecture_or_semantic_paths():
    model = build_ra_hpg_v1_relation_aware()
    layer_types = {type(layer).__name__ for layer in model._flatten_layers()}
    forbidden_layers = {
        "Conv2D",
        "DepthwiseConv2D",
        "SeparableConv2D",
        "MultiHeadAttention",
        "Attention",
        "Softmax",
        "BatchNormalization",
    }
    assert not forbidden_layers & layer_types
    source = "\n".join(
        (CANDIDATE / name).read_text(encoding="utf-8").lower()
        for name in ("model.py", "graph.py")
    )
    forbidden_tokens = (
        "mediapipe",
        "landmark",
        "roi",
        "part_soft",
        "hog",
        "lbp",
        "gabor",
        "laplacian",
        "pretrained",
        "transformer",
    )
    assert all(token not in source for token in forbidden_tokens)
    model_source = (CANDIDATE / "model.py").read_text(encoding="utf-8").lower()
    assert "max_relative_aggregate" not in model_source


def test_graph_and_data_are_byte_identical_to_accepted_v1_2():
    for name in ("graph.py", "data.py"):
        assert _sha256(CANDIDATE / name) == _sha256(V1_2_CANDIDATE / name)


def test_training_configuration_is_exactly_frozen_from_v1_2():
    assert training.TRAINING_CONFIG == {
        "seed": 42,
        "optimizer": "AdamW",
        "learning_rate": 3e-4,
        "weight_decay": 5e-4,
        "global_clipnorm": 1.0,
        "batch_size": 64,
        "max_epochs": 100,
        "warmup_epochs": 5,
        "cosine_final_learning_rate": 1e-6,
        "loss": "categorical_crossentropy_from_logits",
        "label_smoothing": 0.05,
        "checkpoint": "earliest_strict_max_val_accuracy",
        "early_stopping_monitor": "val_loss",
        "early_stopping_patience": 15,
        "early_stopping_min_delta": 0.0,
    }
    schedule = training.WarmupCosine(steps_per_epoch=10)
    assert float(schedule(0)) == 0.0
    assert float(schedule(50)) == pytest.approx(3e-4)
    assert float(schedule(1000)) == pytest.approx(1e-6)
    optimizer = training.build_optimizer(10)
    assert isinstance(optimizer, tf.keras.optimizers.AdamW)
    assert float(optimizer.weight_decay) == pytest.approx(5e-4)
    assert float(optimizer.global_clipnorm) == pytest.approx(1.0)


def test_exact_v1_2_comparator_and_zero_deltas():
    reference = {
        "clean_train_accuracy": 0.614894284022432,
        "clean_train_macro_f1": 0.5614369765915708,
        "validation_accuracy": 0.5806631373641683,
        "validation_macro_f1": 0.5238975323290902,
    }
    assert training.CF_HPG_V1_2_REFERENCE == reference
    assert training.outcome_deltas(
        validation_accuracy=reference["validation_accuracy"],
        validation_macro_f1=reference["validation_macro_f1"],
        clean_train_accuracy=reference["clean_train_accuracy"],
        clean_train_macro_f1=reference["clean_train_macro_f1"],
    ) == {
        "delta_val_accuracy_pp": 0.0,
        "delta_val_macro_pp": 0.0,
        "delta_clean_train_accuracy_pp": 0.0,
        "delta_clean_train_macro_pp": 0.0,
    }
    source = inspect.getsource(training.main)
    assert '"deltas_vs_cf_hpg_v1_2_pp"' in source


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        ((0.7000, 0.6700, 0.7800, 0.7500), "RA_HPG_V1_STRETCH_PASS"),
        ((0.6500, 0.6200, 0.7300, 0.7000), "RA_HPG_V1_PASS"),
        ((0.6306631373641683, 0.55, 0.724894284022432, 0.70), "RELATION_OVERFIT_SHIFT"),
        ((0.6106631373641683, 0.55, 0.734894284022432, 0.66), "RELATION_OVERFIT_SHIFT"),
        ((0.6306631373641683, 0.55, 0.664894284022432, 0.60), "RELATION_STRONG_SIGNAL"),
        ((0.6106631373641683, 0.55, 0.65, 0.59), "RELATION_PARTIAL_SIGNAL"),
        ((0.6006631373641683, 0.55, 0.664894284022432, 0.61), "RELATION_FIT_WITHOUT_VAL_GAIN"),
        ((0.60, 0.54, 0.65, 0.59), "RELATION_UNDERFIT_REMAINS"),
        ((0.64, 0.55, 0.65, 0.60), "RELATION_INCONCLUSIVE"),
    ],
    ids=(
        "stretch",
        "pass",
        "overfit-beats-strong",
        "overfit-beats-partial",
        "strong",
        "partial",
        "fit-without-val-gain",
        "underfit",
        "inconclusive",
    ),
)
def test_registered_decision_precedence_and_reachability(metrics, expected):
    validation_accuracy, validation_macro, train_accuracy, train_macro = metrics
    assert training.classify_outcome(
        validation_accuracy=validation_accuracy,
        validation_macro_f1=validation_macro,
        clean_train_accuracy=train_accuracy,
        clean_train_macro_f1=train_macro,
    ) == expected


@pytest.mark.parametrize(
    ("validation_delta", "train_delta", "expected"),
    [
        (3.0, 0.0, "RELATION_PARTIAL_SIGNAL"),
        (2.99999, 0.0, "RELATION_UNDERFIT_REMAINS"),
        (5.0, 5.0, "RELATION_STRONG_SIGNAL"),
        (5.0, 4.99999, "RELATION_INCONCLUSIVE"),
        (2.99999, 5.0, "RELATION_FIT_WITHOUT_VAL_GAIN"),
    ],
)
def test_registered_delta_boundaries(validation_delta, train_delta, expected):
    reference = training.CF_HPG_V1_2_REFERENCE
    assert training.classify_outcome(
        validation_accuracy=reference["validation_accuracy"] + validation_delta / 100.0,
        validation_macro_f1=0.55,
        clean_train_accuracy=reference["clean_train_accuracy"] + train_delta / 100.0,
        clean_train_macro_f1=0.60,
    ) == expected


def test_cli_exposes_only_train_validation_and_output_inputs():
    options = {
        option
        for action in training.build_parser()._actions
        for option in action.option_strings
        if option not in {"--help", "-h"}
    }
    assert options == {"--train-csv", "--val-csv", "--output-root"}


def test_loader_opens_only_explicit_tiny_csv(tmp_path, monkeypatch):
    source = tmp_path / "tiny.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["emotion", "pixels"])
        writer.writeheader()
        writer.writerow({"emotion": 2, "pixels": " ".join(["0"] * 2304)})
    opened = []
    original_open = Path.open

    def tracked_open(path, *args, **kwargs):
        opened.append(Path(path).resolve())
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    images, labels = data.load_fer_csv(source, expected_samples=1)
    assert opened == [source.resolve()]
    assert images.shape == (1, 48, 48, 1)
    assert labels.tolist() == [2]


@pytest.mark.parametrize(
    "relative_path",
    ["test.csv", "test_private.csv", "test-final.csv", "test/sample.csv", "testing/sample.csv", "test_split/sample.csv", "test-split/sample.csv"],
)
def test_forbidden_split_paths_fail_before_open(tmp_path, monkeypatch, relative_path):
    opened = []

    def forbidden_open(*args, **kwargs):
        opened.append((args, kwargs))
        raise AssertionError("Forbidden source must not be opened")

    monkeypatch.setattr(Path, "open", forbidden_open)
    with pytest.raises(data.CFHPGDataError, match="forbidden"):
        data.load_fer_csv(tmp_path / relative_path, expected_samples=3_589)
    assert opened == []


def test_cli_rejects_forbidden_path_before_loader_or_output(tmp_path, monkeypatch):
    calls = []

    def unexpected_loader(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("No loader call is allowed")

    monkeypatch.setattr(training, "load_fer_csv", unexpected_loader)
    output = tmp_path / "output"
    with pytest.raises(data.CFHPGDataError, match="forbidden"):
        training.main(
            ["--train-csv", str(tmp_path / "train.csv"), "--val-csv", str(tmp_path / "test.csv"), "--output-root", str(output)]
        )
    assert calls == []
    assert not output.exists()


def test_synthetic_tf_function_forward_backward_is_finite_and_updates():
    model = build_ra_hpg_v1_relation_aware()
    optimizer = training.build_optimizer(steps_per_epoch=2)
    optimizer.iterations.assign(1)
    images = tf.random.stateless_uniform([2, 48, 48, 1], [55, 11], maxval=255.0)
    labels = tf.constant([1, 5], tf.int32)
    before = [value.numpy().copy() for value in model.trainable_variables]

    @tf.function
    def train_step(batch_images, batch_labels):
        with tf.GradientTape() as tape:
            logits = model(batch_images, training=True)
            loss = tf.reduce_mean(
                training.sparse_smoothed_cross_entropy(batch_labels, logits)
            )
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return logits, loss, [tf.reduce_all(tf.math.is_finite(g)) for g in gradients]

    logits, loss, finite_gradients = train_step(images, labels)
    assert bool(tf.reduce_all(tf.math.is_finite(logits)))
    assert bool(tf.math.is_finite(loss))
    assert all(bool(value) for value in finite_gradients)
    assert any(
        not np.array_equal(old, new.numpy())
        for old, new in zip(before, model.trainable_variables)
    )


def test_keras_round_trip_reproduces_synthetic_logits(tmp_path):
    model = build_ra_hpg_v1_relation_aware()
    images = tf.random.stateless_uniform([1, 48, 48, 1], [55, 12], maxval=255.0)
    expected = model(images, training=False)
    path = tmp_path / "synthetic.keras"
    model.save(path)
    restored = tf.keras.models.load_model(path, compile=False)
    actual = restored(images, training=False)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize("path", ACCEPTED_CANDIDATES)
def test_accepted_cf_candidates_are_unchanged_from_base(path):
    completed = subprocess.run(
        ["git", "diff", "--quiet", BASE, "--", str(path.relative_to(ROOT))],
        cwd=ROOT,
        check=False,
    )
    assert completed.returncode == 0


@pytest.mark.parametrize("path", FROZEN_PATHS)
def test_generation_one_step13_and_frozen_package_are_unchanged(path):
    completed = subprocess.run(
        ["git", "diff", "--quiet", BASE, "--", str(path.relative_to(ROOT))],
        cwd=ROOT,
        check=False,
    )
    assert completed.returncode == 0


def test_candidate_imports_no_pytorch_runtime():
    imported = set()
    for path in CANDIDATE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
    assert "torch" not in imported
