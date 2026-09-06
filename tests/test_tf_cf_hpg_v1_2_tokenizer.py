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

from research.candidates.tf_cf_hpg_v1_2_tokenizer import data
from research.candidates.tf_cf_hpg_v1_2_tokenizer.graph import (
    HybridGraphBuilder,
    learned_relation_adjacency,
    max_relative_aggregate,
    normalized_grid_centers,
    spatial_8_neighbor_adjacency,
)
from research.candidates.tf_cf_hpg_v1_2_tokenizer.model import (
    CFHPGTokenizer,
    MaxRelativeGraphBlock,
    build_cf_hpg_v1_2_tokenizer,
    patchify_and_scale,
    pool_2x2_mean,
)
from research.candidates.tf_cf_hpg_v1_2_tokenizer import (
    train_validation_only as training,
)
from research.candidates.tf_cf_hpg_v1_1_resolution.model import (
    build_cf_hpg_v1_1_resolution,
)


ROOT = Path(__file__).resolve().parents[1]
BASE = "934057f60a6cb3bc7ec247e000573a40c390a426"
CANDIDATE = ROOT / "research/candidates/tf_cf_hpg_v1_2_tokenizer"
V1_0_CANDIDATE = ROOT / "research/candidates/tf_cf_hpg"
V1_1_CANDIDATE = ROOT / "research/candidates/tf_cf_hpg_v1_1_resolution"
FROZEN = ROOT / "standalone/lap_gnn_tensorflow_ofix7_mid_candidate"
STEP13_PATHS = (
    ROOT
    / "research/candidates/tf_learned_local_residual_slots/evaluate_remaining_prior_probe.py",
    ROOT / "tools/run_issue40_step13_execution.py",
)


def test_patchification_is_exact_144_by_16_row_major_and_scaled():
    image = tf.reshape(tf.range(48 * 48, dtype=tf.float32), [1, 48, 48, 1])
    patches = patchify_and_scale(image).numpy()
    assert patches.shape == (1, 144, 16)
    expected_first = image.numpy()[0, :4, :4, 0].reshape(-1) / 127.5 - 1.0
    expected_second = image.numpy()[0, :4, 4:8, 0].reshape(-1) / 127.5 - 1.0
    np.testing.assert_allclose(patches[0, 0], expected_first)
    np.testing.assert_allclose(patches[0, 1], expected_second)


def test_patchification_uses_only_reshape_transpose_and_scaling():
    source = inspect.getsource(patchify_and_scale)
    assert "extract_patches" not in source
    assert "Conv2D" not in source
    assert "convolution" not in source.lower()
    assert "tf.reshape" in source
    assert "tf.transpose" in source


def test_raw_tokenizer_has_exact_registered_order_before_positional_fusion():
    source = inspect.getsource(CFHPGTokenizer.call)
    ordered_fragments = (
        "nodes = self.raw_projection(patches)",
        "nodes = self.raw_local_activation(nodes)",
        "nodes = self.raw_local_refinement(nodes)",
        "nodes = nodes + self.position_projection(positions)",
        "nodes = self.patch_norm(nodes)",
        "nodes = self.patch_activation(nodes)",
        "nodes = self.patch_dropout(nodes, training=training)",
    )
    offsets = [source.index(fragment) for fragment in ordered_fragments]
    assert offsets == sorted(offsets)
    assert source.count("self.position_projection(positions)") == 1


def test_tokenizer_adds_exactly_one_dense_relative_to_v1_1():
    previous = build_cf_hpg_v1_1_resolution()
    tokenizer = build_cf_hpg_v1_2_tokenizer()
    previous_dense = {
        layer.name
        for layer in previous._flatten_layers()
        if isinstance(layer, tf.keras.layers.Dense)
    }
    tokenizer_dense = {
        layer.name
        for layer in tokenizer._flatten_layers()
        if isinstance(layer, tf.keras.layers.Dense)
    }
    assert tokenizer_dense - previous_dense == {"raw_local_refinement"}
    assert not previous_dense - tokenizer_dense
    assert tokenizer.raw_local_refinement.kernel.shape == (96, 96)
    assert tokenizer.raw_local_refinement.bias.shape == (96,)


def test_coordinates_are_exact_grid_centers_in_row_major_xy_order():
    coordinates = normalized_grid_centers(12).numpy()
    axis = (np.arange(12, dtype=np.float32) + 0.5) * (2.0 / 12.0) - 1.0
    expected = np.asarray([(x, y) for y in axis for x in axis], np.float32)
    np.testing.assert_allclose(coordinates, expected, atol=0.0)


def test_exact_spatial_eight_neighborhood_has_no_self_and_is_bidirectional():
    adjacency = spatial_8_neighbor_adjacency(12).numpy()
    assert adjacency.shape == (144, 144)
    assert not np.diag(adjacency).any()
    np.testing.assert_array_equal(adjacency, adjacency.T)
    assert adjacency[0].sum() == 3
    assert adjacency[1].sum() == 5
    assert adjacency[13].sum() == 8
    assert np.flatnonzero(adjacency[0]).tolist() == [1, 12, 13]


def test_learned_relation_graph_has_exact_top_four_and_excludes_self():
    nodes = tf.random.stateless_normal([2, 144, 8], seed=[42, 1])
    adjacency = learned_relation_adjacency(nodes).numpy()
    assert adjacency.shape == (2, 144, 144)
    assert np.all(adjacency.sum(axis=-1) == 4)
    assert not np.diagonal(adjacency, axis1=1, axis2=2).any()


def test_hybrid_graph_is_boolean_union_and_deduplicates_overlap():
    nodes = tf.random.stateless_normal([1, 144, 8], seed=[42, 2])
    spatial = spatial_8_neighbor_adjacency(12).numpy()
    learned = learned_relation_adjacency(nodes).numpy()[0]
    hybrid = HybridGraphBuilder(12)(nodes).numpy()[0]
    np.testing.assert_array_equal(hybrid, np.logical_or(spatial, learned))
    assert hybrid.dtype == np.bool_


def _capture_graph_reuse(monkeypatch, model, builder_name, block_name, images):
    builder = getattr(model, builder_name)
    blocks = getattr(model, block_name)
    original_builder_call = builder.call
    builder_calls = []
    seen_adjacencies = []

    def counted_builder(nodes):
        builder_calls.append(1)
        return original_builder_call(nodes)

    monkeypatch.setattr(builder, "call", counted_builder)
    for block in blocks:
        original_block_call = block.call

        def captured(nodes, adjacency, training=False, original=original_block_call):
            seen_adjacencies.append(adjacency)
            return original(nodes, adjacency, training=training)

        monkeypatch.setattr(block, "call", captured)
    model(images, training=False)
    return builder_calls, seen_adjacencies


def test_fine_graph_is_built_once_and_same_tensor_reused_twice(monkeypatch):
    model = build_cf_hpg_v1_2_tokenizer()
    calls, seen = _capture_graph_reuse(
        monkeypatch,
        model,
        "fine_graph_builder",
        "fine_blocks",
        tf.zeros([1, 48, 48, 1]),
    )
    assert len(calls) == 1
    assert len(seen) == 2 and seen[0] is seen[1]


def test_hierarchy_is_exact_parameter_free_two_by_two_arithmetic_mean():
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
    assert not tf.keras.layers.Layer().weights


def test_coarse_graph_has_registered_spatial_and_top_four_semantics():
    nodes = tf.random.stateless_normal([2, 36, 128], seed=[42, 3])
    hybrid = HybridGraphBuilder(6, 4)(nodes).numpy()
    spatial = spatial_8_neighbor_adjacency(6).numpy()
    learned = learned_relation_adjacency(nodes, 4).numpy()
    np.testing.assert_array_equal(hybrid, np.logical_or(spatial[None], learned))


def test_coarse_graph_is_built_once_and_same_tensor_reused_twice(monkeypatch):
    model = build_cf_hpg_v1_2_tokenizer()
    calls, seen = _capture_graph_reuse(
        monkeypatch,
        model,
        "coarse_graph_builder",
        "coarse_blocks",
        tf.zeros([1, 48, 48, 1]),
    )
    assert len(calls) == 1
    assert len(seen) == 2 and seen[0] is seen[1]


def test_max_relative_aggregation_matches_golden_tensor():
    nodes = tf.constant([[[1.0, 4.0], [3.0, 2.0], [0.0, 8.0]]])
    adjacency = tf.constant(
        [[[False, True, True], [True, False, False], [True, True, False]]]
    )
    expected = np.asarray([[[2.0, 4.0], [-2.0, 2.0], [3.0, -4.0]]])
    np.testing.assert_allclose(max_relative_aggregate(nodes, adjacency), expected)


def test_model_output_shape_and_registered_stage_inventory():
    model = build_cf_hpg_v1_2_tokenizer()
    assert isinstance(model, CFHPGTokenizer)
    logits, debug = model(tf.zeros([3, 48, 48, 1]), return_debug=True)
    assert logits.shape == (3, 7)
    assert debug["fine_adjacency"].shape == (3, 144, 144)
    assert debug["coarse_adjacency"].shape == (3, 36, 36)
    assert debug["final_nodes"].shape == (3, 36, 128)
    assert len(model.fine_blocks) == len(model.coarse_blocks) == 2
    assert all(isinstance(layer, MaxRelativeGraphBlock) for layer in model.fine_blocks)


def test_parameter_budget_and_exact_variable_inventory():
    model = build_cf_hpg_v1_2_tokenizer()
    assert model.count_params() == training.EXPECTED_PARAMETER_COUNT == 420_839
    assert sum(int(tf.size(value)) for value in model.trainable_variables) == 420_839
    assert len(model.trainable_variables) == training.EXPECTED_TRAINABLE_VARIABLE_COUNT == 66
    assert len(model.variables) == training.EXPECTED_KERAS_VARIABLE_COUNT == 76
    assert training.validate_model_identity(model) == {
        "parameters": 420_839,
        "trainable_variables": 66,
        "keras_variables": 76,
    }
    assert model.count_params() <= 500_000 < 600_000


def test_all_three_model_identity_fields_fail_closed():
    with pytest.raises(training.ValidationOnlyHarnessError, match="identity drift"):
        training.validate_model_identity(build_cf_hpg_v1_1_resolution())


def test_model_round_trip_uses_only_registered_candidate_classes(tmp_path):
    model = build_cf_hpg_v1_2_tokenizer()
    expected = model(tf.zeros([1, 48, 48, 1]), training=False)
    path = tmp_path / "synthetic.keras"
    model.save(path)
    restored = tf.keras.models.load_model(path, compile=False)
    actual = restored(tf.zeros([1, 48, 48, 1]), training=False)
    np.testing.assert_allclose(actual, expected, atol=0.0)


def test_architecture_has_no_forbidden_layers_or_softmax():
    model = build_cf_hpg_v1_2_tokenizer()
    layer_types = {type(layer).__name__ for layer in model._flatten_layers()}
    assert not {"Conv2D", "DepthwiseConv2D", "SeparableConv2D"} & layer_types
    assert "Softmax" not in layer_types


def test_candidate_model_and_graph_sources_have_no_semantic_or_handcrafted_path():
    source = "\n".join(
        (CANDIDATE / name).read_text(encoding="utf-8").lower()
        for name in ("model.py", "graph.py")
    )
    forbidden = (
        "mediapipe",
        "landmark",
        "roi",
        "part_soft",
        "hog",
        "lbp",
        "gabor",
        "laplacian",
        "pretrained",
        "load_weights",
        "attention",
        "slot",
    )
    assert all(token not in source for token in forbidden)


def test_graph_and_data_sources_are_byte_identical_to_v1_1():
    for name in ("graph.py", "data.py"):
        v1_1 = (V1_1_CANDIDATE / name).read_bytes()
        tokenizer = (CANDIDATE / name).read_bytes()
        assert hashlib.sha256(tokenizer).digest() == hashlib.sha256(v1_1).digest()


def test_train_augmentation_is_stateless_and_validation_is_unchanged():
    image = tf.reshape(tf.range(48 * 48, dtype=tf.float32) % 256, [48, 48, 1])
    first = data.augment_train_image(image, tf.constant(7), 42)
    second = data.augment_train_image(image, tf.constant(7), 42)
    np.testing.assert_allclose(first, second, atol=0.0)
    images = tf.stack([image, image])
    labels = tf.constant([1, 2])
    validation_one = next(iter(data.build_dataset(images, labels, training=False)))
    validation_two = next(iter(data.build_dataset(images, labels, training=False)))
    np.testing.assert_allclose(validation_one[0], validation_two[0], atol=0.0)
    np.testing.assert_array_equal(validation_one[0], images)


def test_augmentation_keeps_class_identity_and_needs_no_semantic_remap():
    images = tf.zeros([2, 48, 48, 1], tf.float32)
    labels = tf.constant([3, 6], tf.int32)
    augmented_images, augmented_labels = next(
        iter(data.build_dataset(images, labels, training=True, batch_size=2))
    )
    assert augmented_images.shape == images.shape
    assert sorted(augmented_labels.numpy().tolist()) == [3, 6]
    assert data.GEOMETRIC_FILL_MODE == "REFLECT"
    assert data.ERASE_FILL_NORMALIZED == 0.0


def test_exact_future_optimizer_schedule_loss_and_lifecycle_config():
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


def test_v1_1_comparator_and_delta_formulas_are_exactly_locked():
    assert training.V1_1_REFERENCE == {
        "clean_train_accuracy": 0.605385070883695,
        "clean_train_macro_f1": 0.5474748347135944,
        "validation_accuracy": 0.5734187796043466,
        "validation_macro_f1": 0.5154247791468033,
    }
    assert training.outcome_deltas(
        validation_accuracy=0.5734187796043466,
        validation_macro_f1=0.5154247791468033,
        clean_train_accuracy=0.605385070883695,
        clean_train_macro_f1=0.5474748347135944,
    ) == {
        "delta_val_accuracy_pp": 0.0,
        "delta_val_macro_pp": 0.0,
        "delta_clean_train_accuracy_pp": 0.0,
        "delta_clean_train_macro_pp": 0.0,
    }


def test_earliest_strict_maximum_checkpoint_policy():
    assert training.earliest_strict_max_epoch([0.5, 0.6, 0.6, 0.59]) == 1
    assert training.earliest_strict_max_epoch([0.5, 0.5, 0.7, 0.7]) == 2
    with pytest.raises(training.ValidationOnlyHarnessError):
        training.earliest_strict_max_epoch([])


def test_clean_train_evaluation_explicitly_disables_augmentation(monkeypatch):
    observed = {}

    def capture(images, labels, *, training, batch_size=64, seed=42):
        observed.update(training=training, batch_size=batch_size)
        return "clean"

    monkeypatch.setattr(data, "build_dataset", capture)
    assert data.build_clean_evaluation_dataset([1], [0], batch_size=8) == "clean"
    assert observed == {"training": False, "batch_size": 8}


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        ((0.7000, 0.6700, 0.7800, 0.7500), "CF_HPG_V1_2_STRETCH_PASS"),
        ((0.6500, 0.6200, 0.7300, 0.7000), "CF_HPG_V1_2_PASS"),
        ((0.6234187796043466, 0.56, 0.655385070883695, 0.60), "TOKENIZER_STRONG_SIGNAL"),
        ((0.6034187796043466, 0.55, 0.64, 0.58), "TOKENIZER_PARTIAL_SIGNAL"),
        ((0.63, 0.56, 0.75, 0.70), "TOKENIZER_OVERFIT_SHIFT"),
        ((0.60, 0.54, 0.65, 0.59), "TOKENIZER_UNDERFIT_REMAINS"),
        ((0.60, 0.55, 0.70, 0.63), "TOKENIZER_INCONCLUSIVE"),
    ],
)
def test_exact_decision_boundaries(metrics, expected):
    validation_accuracy, validation_macro, train_accuracy, train_macro = metrics
    assert (
        training.classify_outcome(
            validation_accuracy=validation_accuracy,
            validation_macro_f1=validation_macro,
            clean_train_accuracy=train_accuracy,
            clean_train_macro_f1=train_macro,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        ((0.63, 0.56, 0.75, 0.70), "TOKENIZER_OVERFIT_SHIFT"),
        ((0.6084187796043466, 0.55, 0.72, 0.66), "TOKENIZER_OVERFIT_SHIFT"),
        (
            (0.6234187796043466, 0.56, 0.67, 0.60),
            "TOKENIZER_STRONG_SIGNAL",
        ),
        ((0.6084187796043466, 0.55, 0.65, 0.59), "TOKENIZER_PARTIAL_SIGNAL"),
        ((0.7000, 0.6700, 0.7800, 0.7500), "CF_HPG_V1_2_STRETCH_PASS"),
        ((0.6500, 0.6200, 0.7300, 0.7000), "CF_HPG_V1_2_PASS"),
    ],
    ids=(
        "overfit-beats-strong",
        "overfit-beats-partial",
        "strong-without-overfit",
        "partial-without-overfit",
        "stretch-precedes-diagnostic-branches",
        "pass-precedes-diagnostic-branches",
    ),
)
def test_authoritative_diagnostic_precedence(metrics, expected):
    validation_accuracy, validation_macro, train_accuracy, train_macro = metrics
    assert (
        training.classify_outcome(
            validation_accuracy=validation_accuracy,
            validation_macro_f1=validation_macro,
            clean_train_accuracy=train_accuracy,
            clean_train_macro_f1=train_macro,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("validation_accuracy", "clean_train_accuracy", "expected"),
    [
        (training.V1_1_REFERENCE["validation_accuracy"] + 0.03, 0.65, "TOKENIZER_PARTIAL_SIGNAL"),
        (training.V1_1_REFERENCE["validation_accuracy"] + 0.029999, 0.65, "TOKENIZER_UNDERFIT_REMAINS"),
        (training.V1_1_REFERENCE["validation_accuracy"] + 0.05, training.V1_1_REFERENCE["clean_train_accuracy"] + 0.05, "TOKENIZER_STRONG_SIGNAL"),
        (training.V1_1_REFERENCE["validation_accuracy"] + 0.05, training.V1_1_REFERENCE["clean_train_accuracy"] + 0.049999, "TOKENIZER_UNDERFIT_REMAINS"),
    ],
    ids=("partial-lower-inclusive", "partial-below-lower", "strong-five-inclusive", "strong-needs-both-five"),
)
def test_exact_delta_boundaries(validation_accuracy, clean_train_accuracy, expected):
    assert training.classify_outcome(
        validation_accuracy=validation_accuracy,
        validation_macro_f1=0.55,
        clean_train_accuracy=clean_train_accuracy,
        clean_train_macro_f1=0.60,
    ) == expected


def test_cli_exposes_only_train_validation_and_output_inputs():
    option_strings = {
        option
        for action in training.build_parser()._actions
        for option in action.option_strings
        if option not in {"--help", "-h"}
    }
    assert option_strings == {"--train-csv", "--val-csv", "--output-root"}


def test_loader_opens_only_the_explicitly_supplied_tiny_csv(tmp_path, monkeypatch):
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
    [
        "test.csv",
        "test_private.csv",
        "test-final.csv",
        "test/sample.csv",
        "testing/sample.csv",
        "test_split/sample.csv",
        "test-split/sample.csv",
    ],
)
@pytest.mark.parametrize("expected_samples", [None, 3_589])
def test_forbidden_split_paths_fail_before_open(
    tmp_path, monkeypatch, relative_path, expected_samples
):
    source = tmp_path / relative_path
    opened = []

    def forbidden_open(*args, **kwargs):
        opened.append((args, kwargs))
        raise AssertionError("Forbidden source must not be opened")

    monkeypatch.setattr(Path, "open", forbidden_open)
    with pytest.raises(data.CFHPGDataError, match="forbidden"):
        data.load_fer_csv(source, expected_samples=expected_samples)
    assert opened == []


@pytest.mark.parametrize(
    "basename", ["train.csv", "val.csv", "validation.csv", "tiny.csv", "contest_data.csv"]
)
def test_normal_and_harmless_substring_csv_names_remain_allowed(tmp_path, basename):
    source = tmp_path / basename
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["emotion", "pixels"])
        writer.writeheader()
        writer.writerow({"emotion": 4, "pixels": " ".join(["1"] * 2304)})
    images, labels = data.load_fer_csv(source, expected_samples=1)
    assert images.shape == (1, 48, 48, 1)
    assert labels.tolist() == [4]


def test_cli_rejects_forbidden_validation_path_before_any_loader_or_output(
    tmp_path, monkeypatch
):
    loader_calls = []

    def unexpected_loader(*args, **kwargs):
        loader_calls.append((args, kwargs))
        raise AssertionError("CLI must reject all paths before loading either source")

    monkeypatch.setattr(training, "load_fer_csv", unexpected_loader)
    output_root = tmp_path / "output"
    with pytest.raises(data.CFHPGDataError, match="forbidden"):
        training.main(
            [
                "--train-csv",
                str(tmp_path / "train.csv"),
                "--val-csv",
                str(tmp_path / "test.csv"),
                "--output-root",
                str(output_root),
            ]
        )
    assert loader_calls == []
    assert not output_root.exists()


def test_synthetic_tf_function_forward_backward_is_finite_and_updates():
    model = build_cf_hpg_v1_2_tokenizer()
    optimizer = training.build_optimizer(steps_per_epoch=2)
    optimizer.iterations.assign(1)
    images = tf.random.stateless_uniform([2, 48, 48, 1], [42, 11], maxval=255.0)
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
    assert tf.reduce_all(tf.math.is_finite(logits))
    assert tf.math.is_finite(loss)
    assert all(bool(value) for value in finite_gradients)
    assert any(
        not np.array_equal(old, new.numpy())
        for old, new in zip(before, model.trainable_variables)
    )


@pytest.mark.parametrize("candidate", [V1_0_CANDIDATE, V1_1_CANDIDATE])
def test_v1_0_and_v1_1_candidates_are_unchanged_from_base(candidate):
    completed = subprocess.run(
        ["git", "diff", "--quiet", BASE, "--", str(candidate.relative_to(ROOT))],
        cwd=ROOT,
        check=False,
    )
    assert completed.returncode == 0


def test_generation_one_and_step13_sources_are_unchanged_from_base():
    paths = [
        str(FROZEN.relative_to(ROOT)),
        *(str(path.relative_to(ROOT)) for path in STEP13_PATHS),
    ]
    completed = subprocess.run(
        ["git", "diff", "--quiet", BASE, "--", *paths], cwd=ROOT, check=False
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
