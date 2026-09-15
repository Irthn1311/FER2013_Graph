import numpy as np
from scipy import sparse

from pixel_relational_motif_e0.e0r_geometry import (
    GEOMETRY_DIM,
    GEOMETRY_SEEDS,
    NESTED_DIM,
    build_geometry_csr,
    geometry_row,
    nested_csr,
    sparse_sha256,
    train_pair_distance_median,
)
from pixel_relational_motif_e0.e0r_occurrence import CompactOccurrences
from pixel_relational_motif_e0.e0r_runner import registered_joint_verdict


def _occ(components, xy):
    return CompactOccurrences(
        np.asarray(components, dtype=np.int16),
        np.linspace(0.9, 0.5, len(components)),
        np.asarray([p[1] for p in xy], dtype=np.int16),
        np.asarray([p[0] for p in xy], dtype=np.int16),
    )


def test_geometry_seed_list_exactly_42_through_61():
    assert GEOMETRY_SEEDS == tuple(range(42, 62))


def test_train_distance_median_uses_ordered_pairs_and_handles_no_pairs():
    one = _occ([1], [(2, 2)])
    two = _occ([1, 2], [(2, 2), (5, 6)])
    assert train_pair_distance_median([one]) is None
    assert np.isclose(train_pair_distance_median([two]), 5 / 47)


def test_ordered_pair_geometry_mass_and_component_pair_totals_survive_shuffle():
    occurrences = _occ([1, 2, 1], [(2, 2), (8, 3), (5, 11)])
    actual_index, actual_data = geometry_row(
        occurrences, distance_median=0.2, canonical_image_id=99
    )
    shuffled_index, shuffled_data = geometry_row(
        occurrences, distance_median=0.2, canonical_image_id=99, shuffle_seed=42
    )
    assert np.isclose(actual_data.sum(), 1.0)
    assert np.isclose(shuffled_data.sum(), 1.0)

    def pair_totals(index, data):
        out = np.zeros(128 * 128)
        np.add.at(out, index // 8, data)
        return out

    assert np.allclose(pair_totals(actual_index, actual_data), pair_totals(shuffled_index, shuffled_data))


def test_geometry_shuffle_is_canonical_id_keyed_and_row_order_invariant():
    occurrences = [
        _occ([1, 2], [(2, 2), (8, 3)]),
        _occ([3, 4, 5], [(5, 5), (9, 6), (4, 12)]),
    ]
    ids = np.array([100, 200])
    canonical = build_geometry_csr(occurrences, ids, distance_median=0.2, shuffle_seed=51)
    reordered = build_geometry_csr(occurrences[::-1], ids[::-1], distance_median=0.2, shuffle_seed=51)
    assert (canonical - reordered[::-1]).nnz == 0


def test_sparse_dimensions_zero_one_node_and_identical_unary_block():
    occurrences = [_occ([], []), _occ([1], [(2, 2)]), _occ([1, 2], [(2, 2), (5, 6)])]
    ids = np.array([0, 1, 2])
    actual_g = build_geometry_csr(occurrences, ids, distance_median=0.1)
    shuffled_g = build_geometry_csr(occurrences, ids, distance_median=0.1, shuffle_seed=60)
    unary = np.zeros((3, 128), dtype=np.float32)
    unary[1, 1] = 1
    unary[2, [1, 2]] = 0.5
    actual = nested_csr(unary, actual_g)
    shuffled = nested_csr(unary, shuffled_g)
    assert sparse.isspmatrix_csr(actual) and sparse.isspmatrix_csr(shuffled)
    assert actual_g.shape == shuffled_g.shape == (3, GEOMETRY_DIM)
    assert actual.shape == shuffled.shape == (3, NESTED_DIM)
    assert actual_g[0].nnz == actual_g[1].nnz == 0
    assert (actual[:, :128] - shuffled[:, :128]).nnz == 0


def test_sparse_hash_deterministic_without_dense_materialization(monkeypatch):
    matrix = sparse.csr_matrix(([1.0, 2.0], ([0, 1], [7, 9])), shape=(2, NESTED_DIM))
    monkeypatch.setattr(sparse.csr_matrix, "toarray", lambda self: (_ for _ in ()).throw(AssertionError("dense")))
    assert sparse_sha256(matrix) == sparse_sha256(matrix.copy())


def test_r2_registered_joint_verdict_strict_and_mixed():
    assert registered_joint_verdict(np.array([0.1, 0.2]), np.array([0.1, 0.2]), stage="R2") == "E0.R2 SUPPORTED"
    assert registered_joint_verdict(np.array([0.0, 0.2]), np.array([0.1, 0.2]), stage="R2") == "E0.R2 MIXED — NOT SUPPORTED BY REGISTERED JOINT CRITERION"
    assert registered_joint_verdict(np.array([-0.1, 0.2]), np.array([0.0, 0.2]), stage="R2") == "E0.R2 NOT SUPPORTED"
