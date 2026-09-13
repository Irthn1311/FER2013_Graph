import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from pixel_relational_motif_e0.controls import (
    destroy_ordered_relations_keyed,
    keyed_relation_permutations,
)
from pixel_relational_motif_e0.descriptor import VALID_LOCATIONS, extract_raw_relations
from pixel_relational_motif_e0.e02_runner import (
    BOOTSTRAP_REPLICATES,
    CONTROL_SEEDS,
    K,
    MatchedPool,
    PixelData,
    bind_e02_split,
    dense_features,
    fit_control_dictionary,
    load_actual_dictionary,
)
from pixel_relational_motif_e0.probe import paired_bootstrap_vs_control_mean


def _relations(n=19):
    return np.arange(n * 24, dtype=np.float64).reshape(n, 24)


def test_keyed_relation_control_shape_multiset_and_no_cross_pixel_mixing():
    s = _relations()
    image_ids = np.repeat(np.arange(4), [5, 5, 5, 4])
    pixel_ids = np.arange(len(s))
    original = s.copy()
    out = destroy_ordered_relations_keyed(
        s, image_ids, pixel_ids, control_seed=42, chunk_size=3
    )
    assert out.shape == (len(s), 24)
    assert np.array_equal(s, original)
    assert np.array_equal(np.sort(out, axis=1), np.sort(s, axis=1))
    assert all(set(out[i].tolist()) == set(s[i].tolist()) for i in range(len(s)))


def test_keyed_relation_control_is_repeatable_seed_distinct_and_chunk_invariant():
    rng = np.random.default_rng(4)
    s = rng.normal(size=(31, 24))
    image_ids = np.repeat(np.arange(7), [5, 5, 5, 5, 5, 5, 1])
    pixel_ids = np.arange(31, dtype=np.int64) * 3
    a = destroy_ordered_relations_keyed(
        s, image_ids, pixel_ids, control_seed=42, chunk_size=1
    )
    b = destroy_ordered_relations_keyed(
        s, image_ids, pixel_ids, control_seed=42, chunk_size=17
    )
    c = destroy_ordered_relations_keyed(
        s, image_ids, pixel_ids, control_seed=43, chunk_size=7
    )
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert not np.array_equal(
        keyed_relation_permutations(image_ids, pixel_ids, control_seed=42),
        keyed_relation_permutations(image_ids, pixel_ids, control_seed=43),
    )


def test_keyed_relation_control_is_processing_order_invariant():
    rng = np.random.default_rng(8)
    s = rng.normal(size=(23, 24))
    image_ids = rng.integers(0, 9, size=23)
    pixel_ids = rng.choice(VALID_LOCATIONS, size=23, replace=False)
    canonical = destroy_ordered_relations_keyed(
        s, image_ids, pixel_ids, control_seed=46
    )
    order = rng.permutation(len(s))
    reordered = destroy_ordered_relations_keyed(
        s[order], image_ids[order], pixel_ids[order], control_seed=46, chunk_size=5
    )
    restored = np.empty_like(reordered)
    restored[order] = reordered
    assert np.array_equal(canonical, restored)


def test_relation_control_does_not_accept_or_change_log_sigma():
    s = _relations(3)
    log_sigma = np.array([-1.0, 0.0, 1.0])
    before = log_sigma.copy()
    destroy_ordered_relations_keyed(
        s, np.array([1, 1, 1]), np.array([0, 1, 2]), control_seed=44
    )
    assert np.array_equal(log_sigma, before)
    assert "log_sigma" not in inspect.signature(destroy_ordered_relations_keyed).parameters


def test_descriptor_and_dense_histogram_contract():
    image = np.arange(48 * 48, dtype=np.float64).reshape(48, 48) / (48 * 48)
    s, log_sigma, coords = extract_raw_relations(image)
    assert s.shape == (1936, 24)
    assert log_sigma.shape == (1936,)
    assert coords.shape == (1936, 2)

    class IdentityTransform:
        @staticmethod
        def transform(s, log_sigma):
            return np.zeros((len(s), 12), dtype=np.float64)

    class UniformModel:
        n_components = K

        @staticmethod
        def predict_proba(x):
            return np.full((len(x), K), 1.0 / K)

    data = PixelData(
        images_uint8=np.zeros((2, 48, 48), dtype=np.uint8),
        canonical_ids=np.array([0, 1], dtype=np.int32),
        sha256="0" * 64,
        role="train",
    )
    features = dense_features(data, IdentityTransform(), UniformModel(), control_seed=None, progress_every=0)
    assert features.shape == (2, 128)
    assert np.all(np.isfinite(features))
    assert np.all(features >= 0)
    assert np.allclose(features.sum(axis=1), 1.0)


def test_control_fit_refits_independent_pca_and_gmm_without_label_argument(monkeypatch):
    created_transforms = []
    created_models = []

    class FakePCA:
        mean_ = np.zeros(24)
        components_ = np.zeros((11, 24))
        explained_variance_ = np.ones(11)

    class FakeTransform:
        def __init__(self, *args, **kwargs):
            self.pca = None
            self.log_sigma_mean = None
            self.log_sigma_std = None
            created_transforms.append(self)

        def fit(self, s, log_sigma):
            self.pca = FakePCA()
            self.log_sigma_mean = float(np.mean(log_sigma))
            self.log_sigma_std = 1.0
            return self

        def transform(self, s, log_sigma):
            return np.zeros((len(s), 12))

    class FakeGMM:
        def __init__(self, n_components, variance_floor, **kwargs):
            self.n_components = n_components
            self.weights_ = None
            created_models.append(self)

        def fit(self, x):
            self.weights_ = np.full(K, 1 / K)
            return self

    monkeypatch.setattr("pixel_relational_motif_e0.e02_runner.DescriptorTransform", FakeTransform)
    monkeypatch.setattr("pixel_relational_motif_e0.e02_runner.DiagonalGaussianMixture", FakeGMM)
    monkeypatch.setattr(
        "pixel_relational_motif_e0.e02_runner.variance_floor_from_data",
        lambda x, fraction: np.ones(12),
    )
    pool = MatchedPool(
        s=np.arange(200 * 24, dtype=np.float32).reshape(200, 24),
        log_sigma=np.linspace(-1, 1, 200),
        image_ids=np.repeat(np.arange(10), 20),
        pixel_indices=np.tile(np.arange(20), 10),
        identity_sha256="x" * 64,
    )
    a = fit_control_dictionary(pool, control_seed=42)
    b = fit_control_dictionary(pool, control_seed=43)
    assert a.transform is not b.transform
    assert a.transform.pca is not b.transform.pca
    assert a.model is not b.model
    assert len(created_transforms) == len(created_models) == 2
    assert a.model.n_components == b.model.n_components == 128
    assert "labels" not in inspect.signature(fit_control_dictionary).parameters


@pytest.mark.parametrize(
    "role,path",
    [
        ("test", "val.csv"),
        ("private_test", "val.csv"),
        ("privatetest", "val.csv"),
        ("final_test", "val.csv"),
        ("public", "test.csv"),
        ("public", "/data/PrivateTest.csv"),
        ("public", "/data/final-test.csv"),
    ],
)
def test_e02_private_test_roles_and_paths_fail_closed(role, path):
    with pytest.raises(ValueError):
        bind_e02_split(role, path)


def test_frozen_actual_dictionary_sha_guard(tmp_path: Path):
    wrong = tmp_path / "e01_dictionary.npz"
    wrong.write_bytes(b"not the frozen v533 dictionary")
    with pytest.raises(ValueError, match="locked to v533 dictionary"):
        load_actual_dictionary(wrong)


def test_paired_bootstrap_is_deterministic_at_locked_2000_replicates():
    y = np.array([0, 1, 2, 0, 1, 2] * 3)
    actual = y.copy()
    controls = np.stack([np.roll(y, i + 1) for i in range(len(CONTROL_SEEDS))])
    a = paired_bootstrap_vs_control_mean(
        y, actual, controls, n_replicates=BOOTSTRAP_REPLICATES, seed=42
    )
    b = paired_bootstrap_vs_control_mean(
        y, actual, controls, n_replicates=BOOTSTRAP_REPLICATES, seed=42
    )
    assert np.array_equal(a["delta_accuracy"], b["delta_accuracy"])
    assert np.array_equal(a["delta_macro_f1"], b["delta_macro_f1"])


def test_strict_json_rejects_nonfinite_and_accepts_e02_style_payload():
    payload = {
        "k": 128,
        "control_seeds": list(CONTROL_SEEDS),
        "metrics": {"accuracy": 0.5, "warning": None},
    }
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    with pytest.raises(ValueError):
        json.dumps({"bad": np.nan}, allow_nan=False)
