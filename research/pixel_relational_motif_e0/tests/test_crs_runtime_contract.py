import numpy as np

from pixel_relational_motif_e0.crs_runtime import (
    composition_descriptor_subset_fast,
    sampled_training_descriptors,
)
from pixel_relational_motif_e0.crs_stage import (
    PRIMITIVE_K,
    PRIMITIVE_SIDE,
    composition_descriptors,
    sample_dictionary_centers,
)


def _toy_map() -> np.ndarray:
    rng = np.random.default_rng(123)
    return rng.integers(
        0,
        PRIMITIVE_K,
        size=(PRIMITIVE_SIDE, PRIMITIVE_SIDE),
        dtype=np.int16,
    )


def test_fast_subset_matches_registered_dense_descriptors_exactly():
    primitive_map = _toy_map()
    indices = np.array([0, 1, 35, 36, 217, 511, 900, 1295], dtype=np.int32)
    dense = composition_descriptors(primitive_map, normalize=True)[indices]
    fast = composition_descriptor_subset_fast(primitive_map, indices, normalize=True)
    assert fast.shape == dense.shape
    assert np.allclose(fast, dense, atol=0.0, rtol=0.0)


def test_sampled_training_descriptors_use_exact_registered_centers_for_m_and_c():
    primitive_map = _toy_map()
    indices, m, c = sampled_training_descriptors(primitive_map, image_id=17)
    expected = sample_dictionary_centers(17)

    assert np.array_equal(indices, expected)
    assert m.shape == (24, 1152)
    assert c.shape == (24, 1152)
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0, atol=1e-6)
    assert np.allclose(np.linalg.norm(c, axis=1), 1.0, atol=1e-6)

    mb = m.reshape(24, 9, 128)
    cb = c.reshape(24, 9, 128)
    for row in range(24):
        assert sorted(block.tobytes() for block in mb[row]) == sorted(
            block.tobytes() for block in cb[row]
        )
