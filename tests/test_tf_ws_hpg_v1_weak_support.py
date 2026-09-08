"""Golden implementation tests for preregistered WS-HPG v1.0."""

from __future__ import annotations

import inspect
import builtins
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from research.candidates.tf_ws_hpg_v1_weak_support import graph, layers, model, support


def synthetic_inputs(batch=2, support_value=1.0):
    return {
        "images": tf.reshape(tf.linspace(0.0, 1.0, batch * 48 * 48), [batch, 48, 48, 1]),
        "support": tf.fill([batch, 48, 48, 1], tf.cast(support_value, tf.float32)),
    }


def test_patch_token_shapes_and_support_absent_from_features():
    built = model.build_ws_hpg_v1_weak_support()
    inputs = synthetic_inputs()
    assert model.patchify(inputs["images"]).shape == (2, 256, 9)
    assert built.tokenize(inputs["images"]).shape == (2, 256, 64)
    changed = dict(inputs, support=tf.zeros_like(inputs["support"]))
    np.testing.assert_array_equal(built.tokenize(inputs["images"]), built.tokenize(changed["images"]))


def test_fine_spatial_topology_is_sparse_eight_neighbor_only():
    edges = graph.spatial_8_edge_index(16).numpy()
    assert edges.shape == (2, 1860)
    assert not np.any(edges[0] == edges[1])
    for source, destination in edges.T:
        sy, sx = divmod(source, 16)
        dy, dx = divmod(destination, 16)
        assert max(abs(sy - dy), abs(sx - dx)) == 1
    assert "cosine" not in inspect.getsource(model.WSHPGWeakSupport.__init__)


def test_support_uses_only_extent_and_failure_means_no_prior():
    points_a = np.array([[10.0, 12.0], [37.0, 36.0], [22.0, 28.0]])
    points_b = points_a[[2, 0, 1]]
    kwargs = {"coordinate_system": support.FER_PIXEL_COORDINATE_SYSTEM}
    np.testing.assert_array_equal(
        support.support_from_landmarks(points_a, **kwargs),
        support.support_from_landmarks(points_b, **kwargs),
    )
    for invalid in (None, [], [[20.0, 10.0], [20.0, 35.0]], [[np.nan, 8.0], [30.0, 40.0]], [[-1.0, 5.0], [30.0, 40.0]]):
        np.testing.assert_array_equal(
            support.support_from_landmarks(invalid, **kwargs),
            np.ones((48, 48, 1), np.float32),
        )
    field = support.support_from_landmarks(points_a, **kwargs)
    assert field.min() >= 0.0 and field.max() <= 1.0
    assert not hasattr(field, "landmarks")


def test_pixel_space_extent_produces_nontrivial_support():
    field = support.support_from_landmarks(
        np.array([[10.0, 12.0], [37.0, 36.0], [23.0, 20.0]]),
        coordinate_system=support.FER_PIXEL_COORDINATE_SYSTEM,
    )
    assert field.max() == 1.0
    assert field.min() < 1.0
    assert not np.all(field == 1.0)
    assert field[24, 24, 0] > field[0, 0, 0]


def test_coordinate_units_are_explicit_and_normalized_mode_fails_closed():
    normalized = np.array([[0.2, 0.3], [0.8, 0.7]])
    with pytest.raises(TypeError):
        support.support_from_landmarks(normalized)
    with pytest.raises(ValueError, match="FER pixel coordinates"):
        support.support_from_landmarks(
            normalized, coordinate_system="normalized_xy_0_1"
        )


def test_detector_is_called_once_and_no_identity_survives():
    class Detector:
        def __init__(self):
            self.calls = []

        def detect(self, image):
            self.calls.append(image)
            return np.array([[10.0, 12.0], [37.0, 36.0]], np.float32)

    detector = Detector()
    result = support.build_support_from_clean_image(np.zeros((48, 48, 1)), detector)
    assert len(detector.calls) == 1 and result.shape == (48, 48, 1)
    assert result.min() < 1.0


def test_exact_mediapipe_detector_output_contract_integration(monkeypatch):
    from lap_gnn_tf.priors.mediapipe_priors import MediaPipeFaceDetector

    class Landmark:
        def __init__(self, x, y):
            self.x, self.y = x, y

    class Face:
        landmark = [Landmark(10.0 / 47.0, 12.0 / 47.0), Landmark(37.0 / 47.0, 36.0 / 47.0)]

    class FaceMesh:
        def process(self, _rgb):
            return type("Result", (), {"multi_face_landmarks": [Face()]})()

    detector = object.__new__(MediaPipeFaceDetector)
    detector.available = True
    detector.detection_size = 48
    detector.preprocess_mode = "raw"
    detector.padding_pixels = 0
    detector.face_mesh = FaceMesh()
    detector.landmarker = None
    detector.mp = None
    detected = detector.detect(np.zeros((48, 48), np.uint8))
    np.testing.assert_allclose(detected, [[10.0, 12.0], [37.0, 36.0]])
    field = support.build_support_from_clean_image(np.zeros((48, 48), np.uint8), detector)
    assert field.max() == 1.0 and field.min() < 1.0


def test_patch_support_mean_and_range():
    field = tf.reshape(tf.range(48 * 48, dtype=tf.float32), [1, 48, 48, 1]) / (48 * 48)
    scores = support.patch_support_scores(field)
    assert scores.shape == (1, 256, 1)
    assert float(tf.reduce_min(scores)) >= 0 and float(tf.reduce_max(scores)) <= 1
    np.testing.assert_allclose(scores[0, 0, 0], tf.reduce_mean(field[0, :3, :3, 0]))


def test_support_gate_floor_no_deletion_and_ones_identity():
    block = layers.FineWeakSupportRelationBlock(4)
    nodes = tf.random.stateless_normal([1, 4, 4], [2, 3])
    coords = graph.normalized_grid_coordinates(2)
    edges = graph.spatial_8_edge_index(2)
    _, zero_debug = block(nodes, coords, edges, tf.zeros([1, 4, 1]), return_debug=True)
    _, one_debug = block(nodes, coords, edges, tf.ones([1, 4, 1]), return_debug=True)
    np.testing.assert_allclose(zero_debug["support_gate"], 0.25)
    np.testing.assert_allclose(one_debug["support_gate"], 1.0)
    assert float(tf.reduce_min(zero_debug["support_gate"])) >= 0.25


def test_relation_inputs_and_generic_geometry_contract():
    block = layers.FineWeakSupportRelationBlock(4)
    nodes = tf.random.stateless_normal([1, 4, 4], [5, 7])
    edges = graph.spatial_8_edge_index(2)
    _, debug = block(nodes, graph.normalized_grid_coordinates(2), edges, tf.ones([1, 4, 1]), return_debug=True)
    assert debug["gate_input"].shape[-1] == 7  # abs feature relation + dx/dy/distance
    assert debug["geometry"].shape[-1] == 3
    assert debug["content_input"].shape[-1] == 8  # z_j and z_j-z_i
    assert "support" not in block.gate_hidden.name and block.gate_hidden.kernel.shape[0] == 7


def test_pna_lite_manual_golden_and_degree_one_finite():
    messages = tf.constant([[1.0, 3.0], [3.0, 7.0], [5.0, 11.0]])
    destinations = tf.constant([0, 0, 1])
    mean, maximum, std = graph.sparse_pna_lite(messages, destinations, 2)
    np.testing.assert_allclose(mean, [[2, 5], [5, 11]])
    np.testing.assert_allclose(maximum, [[3, 7], [5, 11]])
    np.testing.assert_allclose(std[0], np.sqrt([1 + 1e-6, 4 + 1e-6]), rtol=1e-6)
    np.testing.assert_allclose(std[1], [0.001, 0.001], rtol=1e-5)
    assert np.isfinite(std).all()


def test_sparse_aggregation_matches_tiny_dense_reference():
    rng = np.random.default_rng(4)
    messages = rng.normal(size=(12, 3)).astype(np.float32)
    destinations = np.repeat(np.arange(4), 3).astype(np.int32)
    mean, maximum, std = graph.sparse_pna_lite(messages, destinations, 4)
    grouped = messages.reshape(4, 3, 3)
    np.testing.assert_allclose(mean, grouped.mean(1), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(maximum, grouped.max(1))
    np.testing.assert_allclose(std, np.sqrt(np.maximum(grouped.var(1), 0) + 1e-6), rtol=1e-5, atol=1e-6)


def test_no_dense_fine_message_allocation_source_path():
    source = inspect.getsource(layers._SparseRelationCore)
    assert "unsorted_segment" not in source  # aggregation delegated to sparse helper
    assert "[:, None" not in source and "N,N,D" not in source
    assert "gather" in source


def test_pool1_and_pool2_shapes_and_mean_max_golden():
    pool1 = layers.MeanMaxPool2x2(16, 64, 96)
    inputs = tf.reshape(tf.range(256 * 64, dtype=tf.float32), [1, 256, 64])
    mean, maximum = pool1.grouped_statistics(inputs)
    expected_ids = np.array([0, 1, 16, 17])
    raw = inputs.numpy()[0, expected_ids]
    np.testing.assert_allclose(mean[0, 0], raw.mean(0))
    np.testing.assert_allclose(maximum[0, 0], raw.max(0))
    assert pool1(inputs).shape == (1, 64, 96)
    assert layers.MeanMaxPool2x2(8, 96, 128)(tf.zeros([1, 64, 96])).shape == (1, 16, 128)


def test_support_is_structurally_inaccessible_after_pool1():
    mid_signature = inspect.signature(layers.SupportFreeRelationBlock.call)
    assert "support" not in mid_signature.parameters
    call_source = inspect.getsource(model.WSHPGWeakSupport.call)
    suffix = call_source.split("nodes = self.pool_1(nodes)", 1)[1]
    assert "fine_support" not in suffix and "pixel_support" not in suffix


def test_hybrid_graph_union_no_self_no_duplicates_and_current_representation():
    nodes_a = tf.random.stateless_normal([1, 64, 96], [19, 23])
    nodes_b = tf.reverse(nodes_a, axis=[1])
    edge_a = graph.hybrid_spatial_cosine_edges(nodes_a, 8).numpy()
    edge_b = graph.hybrid_spatial_cosine_edges(nodes_b, 8).numpy()
    assert edge_a.shape[0] == 2 and not np.any(edge_a[0] == edge_a[1])
    assert len(set(map(tuple, edge_a.T))) == edge_a.shape[1]
    spatial = set(map(tuple, graph.spatial_8_edge_index(8).numpy().T))
    assert spatial <= set(map(tuple, edge_a.T))
    assert not np.array_equal(edge_a, edge_b)


def test_graph_rebuilt_before_each_mid_and_coarse_block(monkeypatch):
    calls = []
    original = model.hybrid_spatial_cosine_edges
    def recording(nodes, grid_size, top_k):
        calls.append((grid_size, top_k))
        return original(nodes, grid_size, top_k)
    monkeypatch.setattr(model, "hybrid_spatial_cosine_edges", recording)
    built = model.build_ws_hpg_v1_weak_support()
    calls.clear()
    built(synthetic_inputs(1))
    assert calls == [(8, 4), (8, 4), (4, 4), (4, 4)]


def test_shapes_identity_logits_and_edge_counts():
    built = model.build_ws_hpg_v1_weak_support()
    logits, debug = built(synthetic_inputs(2), return_debug=True)
    assert logits.shape == (2, 7)
    assert debug["fine_nodes"].shape == (2, 256, 64)
    assert debug["mid_nodes"].shape == (2, 64, 96)
    assert debug["coarse_nodes"].shape == (2, 16, 128)
    assert int(debug["fine_edge_count"]) == 2 * 1860
    assert built.count_params() == 707_213
    assert len(built.trainable_variables) == 118
    assert len(built.variables) == 138


def test_tf_function_forward_backward_and_optimizer_step():
    tf.keras.utils.set_random_seed(21)
    built = model.build_ws_hpg_v1_weak_support()
    inputs = synthetic_inputs(1)
    @tf.function
    def step(batch):
        with tf.GradientTape() as tape:
            logits = built(batch, training=True)
            loss = tf.reduce_mean(tf.square(logits))
        gradients = tape.gradient(loss, built.trainable_variables)
        return logits, loss, gradients
    logits, loss, gradients = step(inputs)
    assert np.isfinite(logits).all() and np.isfinite(loss)
    assert all(g is not None and bool(tf.reduce_all(tf.math.is_finite(g))) for g in gradients)
    before = [v.numpy().copy() for v in built.trainable_variables]
    tf.keras.optimizers.SGD(1e-4).apply_gradients(zip(gradients, built.trainable_variables))
    assert any(not np.array_equal(a, b.numpy()) for a, b in zip(before, built.trainable_variables))


def test_keras_round_trip_identical_logits(tmp_path):
    built = model.build_ws_hpg_v1_weak_support()
    inputs = synthetic_inputs(1)
    expected = built(inputs, training=False).numpy()
    destination = tmp_path / "ws_hpg.keras"
    built.save(destination)
    restored = tf.keras.models.load_model(destination)
    np.testing.assert_array_equal(restored(inputs, training=False), expected)


def test_geometric_transform_exactly_coupled():
    image = tf.reshape(tf.range(48 * 48, dtype=tf.float32), [1, 48, 48, 1])
    params = support.GeometricTransform(True, 0.13, 2.0, -3.0)
    _, _, debug = support.apply_geometric_transform(image, image, params, return_parameters=True)
    np.testing.assert_array_equal(debug["image_transform"], debug["support_transform"])


def test_failure_all_ones_support_remains_no_prior_after_geometry():
    image = tf.zeros([1, 48, 48, 1])
    _, transformed = support.apply_geometric_transform(
        image, tf.ones_like(image), support.GeometricTransform(True, 0.13, 2.0, -3.0)
    )
    np.testing.assert_array_equal(transformed, np.ones((1, 48, 48, 1), np.float32))


def test_photometric_and_erasing_leave_support_unchanged():
    inputs = synthetic_inputs(1, 0.7)
    _, after_photo = support.photometric_image_only(inputs["images"], inputs["support"], 0.2, 1.3)
    _, after_erase = support.erase_image_only(inputs["images"], inputs["support"], 2, 3, 10, 11)
    np.testing.assert_array_equal(after_photo, inputs["support"])
    np.testing.assert_array_equal(after_erase, inputs["support"])


def test_support_override_changes_no_architecture_or_weights():
    built = model.build_ws_hpg_v1_weak_support()
    inputs = synthetic_inputs(1, 0.2)
    before = [v.numpy().copy() for v in built.variables]
    normal = built(inputs, support_override="normal")
    no_prior = built(inputs, support_override="ones")
    assert normal.shape == no_prior.shape == (1, 7)
    assert all(np.array_equal(value, variable.numpy()) for value, variable in zip(before, built.variables))
    with pytest.raises(ValueError):
        built(inputs, support_override="anatomical")


def test_package_has_no_data_or_training_lifecycle_and_no_torch_import():
    package = Path(model.__file__).parent
    names = {p.name for p in package.iterdir() if p.is_file()}
    assert not names.intersection({"data.py", "train.py", "train_validation_only.py"})
    source = "\n".join(p.read_text(encoding="utf-8") for p in package.glob("*.py"))
    assert "import torch" not in source and "test.csv" not in source and "FER2013" not in source


def test_synthetic_forward_has_no_filesystem_or_test_split_access(monkeypatch):
    def forbidden_open(*args, **kwargs):
        raise AssertionError(f"unexpected filesystem access: {args!r}")
    built = model.build_ws_hpg_v1_weak_support()
    monkeypatch.setattr(builtins, "open", forbidden_open)
    assert built(synthetic_inputs(1)).shape == (1, 7)


def test_fresh_runtime_does_not_import_pytorch(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    code = (
        "import sys; sys.path.insert(0, r'" + str(repository) + "'); "
        "import research.candidates.tf_ws_hpg_v1_weak_support; "
        "assert not any(k == 'torch' or k.startswith('torch.') for k in sys.modules)"
    )
    completed = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr


def test_fresh_import_without_pythonpath_from_outside_repo(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    probe = repository / "research/candidates/tf_ws_hpg_v1_weak_support/synthetic_benchmark.py"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    completed = subprocess.run([sys.executable, str(probe), "--help"], cwd=tmp_path, env=env, text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "Synthetic-only runtime" in completed.stdout
