import numpy as np
import pytest

from pixel_relational_motif_e0.descriptor import DescriptorTransform, extract_raw_relations, VALID_LOCATIONS
from pixel_relational_motif_e0.e01_runner import (
    _null_jaccard_from_sizes,
    _reject_private_path,
    _within_one_se_candidates,
    compute_exact_decile_edges,
    sample_fixed_stratified_pool,
    stream_anchor_stability,
)


def _images(seed=0, n=20):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, 48, 48), dtype=np.uint8)


def test_private_path_guard_fails_closed():
    with pytest.raises(ValueError):
        _reject_private_path(__import__("pathlib").Path("/data/PrivateTest.csv"))
    with pytest.raises(ValueError):
        _reject_private_path(__import__("pathlib").Path("/data/final-test.csv"))


def test_exact_deciles_and_priority_pool_are_deterministic():
    images = _images(3, 20)
    ids = np.arange(20, dtype=np.int64)
    edges = compute_exact_decile_edges(images, ids)
    assert edges.shape == (11,)
    assert np.all(np.diff(edges) >= 0)

    a = sample_fixed_stratified_pool(
        images,
        ids,
        edges,
        per_decile=20,
        seed=42,
        chunk_images=5,
    )
    b = sample_fixed_stratified_pool(
        images,
        ids,
        edges,
        per_decile=20,
        seed=42,
        chunk_images=7,
    )
    assert a.s.shape == (200, 24)
    assert a.log_sigma.shape == (200,)
    assert a.image_ids.shape == (200,)
    assert int(a.observed_bin_counts.sum()) == 20 * VALID_LOCATIONS
    assert np.array_equal(a.s, b.s)
    assert np.array_equal(a.log_sigma, b.log_sigma)
    assert np.array_equal(a.image_ids, b.image_ids)


def test_count_only_jaccard_null_is_deterministic_and_finite():
    sizes = np.array([10, 12, 8, 11], dtype=np.int64)
    a = _null_jaccard_from_sizes(10, sizes, universe_size=100, n_replicates=50, seed=9)
    b = _null_jaccard_from_sizes(10, sizes, universe_size=100, n_replicates=50, seed=9)
    assert np.array_equal(a, b)
    assert np.all(np.isfinite(a))
    assert np.all((a >= 0) & (a <= 1))


def test_one_se_candidates_use_image_level_ll():
    ll = {
        32: np.array([0.90, 1.10, 1.00, 1.00]),
        64: np.array([1.01, 1.01, 1.01, 1.01]),
        96: np.array([0.50, 0.50, 0.50, 0.50]),
    }
    candidates, stats = _within_one_se_candidates(ll)
    assert 64 in candidates
    assert 96 not in candidates
    assert all("mean" in stats[k] and "se" in stats[k] for k in stats)


class _FakeModel:
    def __init__(self, means, inverse=False):
        self.n_components = 2
        self.means_ = np.asarray(means, dtype=np.float64)
        self.variances_ = np.ones_like(self.means_)
        self.inverse = bool(inverse)

    def predict(self, x):
        base = (np.asarray(x)[:, 0] >= 0).astype(np.int64)
        return 1 - base if self.inverse else base


def test_stream_anchor_stability_matches_permuted_components_without_materializing_anchor():
    images = _images(11, 4)
    s0, l0, _ = extract_raw_relations(images[0].astype(np.float64) / 255.0)
    s1, l1, _ = extract_raw_relations(images[1].astype(np.float64) / 255.0)
    transform = DescriptorTransform().fit(
        np.concatenate([s0, s1], axis=0),
        np.concatenate([l0, l1], axis=0),
    )
    d = 12
    means_a = np.zeros((2, d), dtype=np.float64)
    means_a[0, 0] = -1.0
    means_a[1, 0] = 1.0
    means_b = means_a[::-1].copy()
    models = [_FakeModel(means_a, inverse=False), _FakeModel(means_b, inverse=True)]

    result, counts = stream_anchor_stability(
        models,
        images,
        np.array([2, 3], dtype=np.int64),
        transform,
        n_null=100,
        q=0.05,
        seed=42,
    )
    assert counts.shape == (2, 2)
    assert np.allclose(result.observed_jaccard, 1.0)
    assert result.match_distances.shape == (2, 2)
    assert np.all(result.stable_mask)
