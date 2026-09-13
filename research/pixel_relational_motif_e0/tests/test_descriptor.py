import numpy as np
from pixel_relational_motif_e0.descriptor import DescriptorTransform, extract_raw_relations, VALID_LOCATIONS


def test_descriptor_shape_and_valid_centers_only():
    img = np.arange(48 * 48, dtype=np.float64).reshape(48, 48) / (48 * 48)
    s, l, coords = extract_raw_relations(img)
    assert s.shape == (VALID_LOCATIONS, 24) == (1936, 24)
    assert l.shape == (1936,)
    assert coords.shape == (1936, 2)
    assert tuple(coords[0]) == (2, 2)
    assert tuple(coords[-1]) == (45, 45)


def test_constant_image_is_finite_and_zero_shape_relations():
    s, l, _ = extract_raw_relations(np.zeros((48, 48)))
    assert np.all(np.isfinite(s)) and np.all(np.isfinite(l))
    assert np.array_equal(s, np.zeros_like(s))


def test_transform_is_12d_and_reuses_frozen_pca():
    rng = np.random.default_rng(1)
    imgs = rng.random((3, 48, 48))
    raw = [extract_raw_relations(x) for x in imgs]
    s = np.concatenate([z[0] for z in raw])
    l = np.concatenate([z[1] for z in raw])
    tr = DescriptorTransform().fit(s, l)
    components = tr.pca.components_.copy()
    out = tr.transform(raw[0][0], raw[0][1])
    assert out.shape == (1936, 12)
    _ = tr.transform(raw[1][0], raw[1][1])
    assert np.array_equal(components, tr.pca.components_)
    assert tr.pca.whiten is False
