import numpy as np
import pytest
from pixel_relational_motif_e0.partition import make_image_partition, assert_e0_role
from pixel_relational_motif_e0.controls import destroy_ordered_relations


def test_image_partition_disjoint_and_complete():
    p = make_image_partition(100, seed=7)
    p.assert_valid(100)
    assert len(np.intersect1d(p.fit, p.heldout)) == 0
    assert len(np.intersect1d(p.fit, p.anchor)) == 0


def test_private_test_role_fails_closed():
    with pytest.raises(ValueError):
        assert_e0_role("PrivateTest")
    with pytest.raises(ValueError):
        assert_e0_role("test")


def test_relation_destroy_preserves_each_row_multiset_and_is_deterministic():
    x = np.arange(6 * 24).reshape(6, 24)
    a = destroy_ordered_relations(x, seed=11)
    b = destroy_ordered_relations(x, seed=11)
    assert np.array_equal(a, b)
    assert all(np.array_equal(np.sort(x[i]), np.sort(a[i])) for i in range(len(x)))
    assert np.any(a != x)
