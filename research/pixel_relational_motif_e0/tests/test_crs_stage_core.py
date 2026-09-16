import numpy as np
from scipy import sparse

from pixel_relational_motif_e0 import crs_stage as c


def _toy_map() -> np.ndarray:
    return (
        np.arange(c.PRIMITIVE_SIDE * c.PRIMITIVE_SIDE, dtype=np.int32)
        .reshape(c.PRIMITIVE_SIDE, c.PRIMITIVE_SIDE)
        % c.PRIMITIVE_K
    )


def test_registered_shapes_and_centers():
    centers = c.valid_center_indices()
    assert centers.shape == (1296, 2)
    assert tuple(centers[0]) == (4, 4)
    assert tuple(centers[-1]) == (39, 39)
    assert c.primitive_center_crop(_toy_map()).shape == (36, 36)


def test_composition_descriptor_count_invariants():
    x = c.composition_descriptors(_toy_map(), normalize=False)
    assert x.shape == (1296, 1152)
    blocks = x.reshape(1296, 9, 128)
    assert np.all(blocks.sum(axis=2) == 9)
    assert np.all(x.sum(axis=1) == 81)

    xn = c.composition_descriptors(_toy_map(), normalize=True)
    assert np.allclose(np.linalg.norm(xn, axis=1), 1.0, atol=1e-6)


def test_sparse_descriptor_conversion_is_exact_and_bounded():
    dense = c.composition_descriptors(_toy_map(), normalize=True)
    csr = c.descriptors_to_csr(dense)
    assert sparse.isspmatrix_csr(csr)
    assert csr.shape == dense.shape
    assert np.allclose(csr.toarray(), dense)
    # Each of the nine 3x3 cell histograms can contain at most nine non-zero bins.
    assert np.all(np.diff(csr.indptr) <= 81)
    row_norm = np.sqrt(np.asarray(csr.multiply(csr).sum(axis=1)).reshape(-1))
    assert np.allclose(row_norm, 1.0, atol=1e-6)


def test_control_preserves_exact_cell_histogram_multiset_and_norm():
    dense = c.composition_descriptors(_toy_map(), normalize=True)
    subset_idx = np.array([0, 1, 35, 36, 100, 1295], dtype=np.int32)
    actual = dense[subset_idx]
    control = c.permute_cell_blocks(
        actual,
        split_id="train",
        image_id=7,
        center_indices=subset_idx,
    )

    a = actual.reshape(-1, 9, 128)
    z = control.reshape(-1, 9, 128)
    assert np.allclose(np.linalg.norm(actual, axis=1), np.linalg.norm(control, axis=1))
    for row in range(len(a)):
        original = sorted(block.tobytes() for block in a[row])
        permuted = sorted(block.tobytes() for block in z[row])
        assert original == permuted


def test_keyed_permutation_is_deterministic_and_depends_on_key():
    a = c.keyed_cell_permutation(
        split_id="train", image_id=3, center_y=4, center_x=4
    )
    b = c.keyed_cell_permutation(
        split_id="train", image_id=3, center_y=4, center_x=4
    )
    assert np.array_equal(a, b)
    assert sorted(a.tolist()) == list(range(9))

    variants = {
        tuple(
            c.keyed_cell_permutation(
                split_id="train", image_id=3, center_y=4, center_x=x
            )
        )
        for x in range(4, 12)
    }
    assert len(variants) > 1


def test_dictionary_sampling_is_exact_and_deterministic():
    a = c.sample_dictionary_centers(11)
    b = c.sample_dictionary_centers(11)
    d = c.sample_dictionary_centers(12)
    assert np.array_equal(a, b)
    assert len(a) == 24
    assert len(np.unique(a)) == 24
    assert np.all((a >= 0) & (a < 1296))
    assert not np.array_equal(a, d)


def test_pyramid_contracts():
    p = _toy_map()
    f = c.p_image_feature(p)
    assert f.shape == (640,)
    assert np.isclose(np.linalg.norm(f), 1.0)

    ids = np.arange(1296, dtype=np.int32).reshape(36, 36) % 512
    g = c.crs_image_feature(ids)
    assert g.shape == (2560,)
    assert np.isclose(np.linalg.norm(g), 1.0)


def test_spherical_kmeans_is_deterministic_and_unit_norm():
    rng = np.random.default_rng(0)
    x = np.vstack(
        [
            rng.normal(loc=-1.0, scale=0.2, size=(80, 12)),
            rng.normal(loc=1.0, scale=0.2, size=(80, 12)),
            rng.normal(loc=0.0, scale=0.2, size=(80, 12)),
        ]
    ).astype(np.float32)

    spec = dict(
        n_clusters=6,
        n_init=2,
        max_iter=30,
        batch_size=64,
        random_state=42,
    )
    a = c.SphericalKMeans(**spec).fit(x)
    b = c.SphericalKMeans(**spec).fit(x)

    assert np.allclose(np.linalg.norm(a.cluster_centers_, axis=1), 1.0, atol=1e-6)
    assert np.array_equal(a.labels_, b.labels_)
    assert np.allclose(a.cluster_centers_, b.cluster_centers_)
    assert np.isclose(a.objective_, b.objective_)
    assert np.array_equal(a.predict(x, batch_size=53), a.labels_)


def test_sparse_and_dense_spherical_kmeans_are_equivalent():
    rng = np.random.default_rng(7)
    # Sparse non-negative histogram-like toy data closer to the CRS domain.
    x = np.zeros((180, 48), dtype=np.float32)
    for row in range(len(x)):
        ids = rng.choice(48, size=8, replace=False)
        x[row, ids] = rng.integers(1, 5, size=8).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    xs = sparse.csr_matrix(x)

    spec = dict(
        n_clusters=9,
        n_init=2,
        max_iter=30,
        batch_size=41,
        random_state=42,
    )
    dense_result = c.SphericalKMeans(**spec).fit(x)
    sparse_result = c.SphericalKMeans(**spec).fit(xs)

    assert dense_result.converged_ == sparse_result.converged_
    assert dense_result.init_index_ == sparse_result.init_index_
    assert np.array_equal(dense_result.labels_, sparse_result.labels_)
    assert np.allclose(
        dense_result.cluster_centers_, sparse_result.cluster_centers_, atol=1e-6
    )
    assert np.isclose(dense_result.objective_, sparse_result.objective_, atol=1e-5)
    assert np.array_equal(sparse_result.predict(x), sparse_result.predict(xs))
