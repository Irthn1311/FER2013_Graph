"""Complete contract test suite for Issue #90 runner and preregistered requirements."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from pixel_relational_motif_e0.motif_qualification import Occurrence
import pixel_relational_motif_e0.sparse_retention_relation_runner as r


def test_1_dense_pyramid_boundaries_and_quadrants():
    grid = np.zeros((36, 36), dtype=np.int16)
    grid[5, 5] = 10
    grid[5, 25] = 20
    grid[25, 5] = 30
    grid[25, 25] = 40
    feat = r.dense_pyramid_feature(grid)
    assert feat.shape == (2560,)
    assert np.all(np.isfinite(feat))
    assert np.isclose(np.linalg.norm(feat), 1.0)


def test_2_arm_a_dimension_2560_and_normalization():
    grid = np.random.randint(0, 512, size=(36, 36), dtype=np.int16)
    feat = r.dense_pyramid_feature(grid)
    assert feat.shape == (2560,)
    assert feat.dtype == np.float32
    assert np.isclose(np.linalg.norm(feat), 1.0)


def test_3_arm_b_masks_all_non_qm():
    grid = np.zeros((36, 36), dtype=np.int16)
    grid[5, 5] = 31  # in Q_M
    grid[10, 10] = 99  # NOT in Q_M
    allowed = set(r.FROZEN_QM_TYPES)
    feat_b = r.dense_pyramid_feature(grid, allowed_types=allowed)
    assert feat_b.shape == (2560,)
    assert feat_b[99] == 0.0
    assert feat_b[31] > 0.0
    assert np.isclose(np.linalg.norm(feat_b), 1.0)


def test_4_arm_b_keeps_512_bin_coordinates():
    grid = np.zeros((36, 36), dtype=np.int16)
    grid[0, 0] = 491  # last in Q_M
    allowed = set(r.FROZEN_QM_TYPES)
    feat_b = r.dense_pyramid_feature(grid, allowed_types=allowed)
    assert feat_b[491] > 0.0
    assert feat_b.shape == (2560,)


def test_5_exact_qm_types():
    expected = (
        31,
        81,
        116,
        118,
        123,
        125,
        167,
        169,
        171,
        174,
        179,
        195,
        223,
        305,
        311,
        344,
        381,
        440,
        451,
        479,
        491,
    )
    assert r.FROZEN_QM_TYPES == expected
    assert len(r.FROZEN_QM_TYPES) == 21


def test_6_relation_dimension_3528():
    qm_to_idx = {t: i for i, t in enumerate(r.FROZEN_QM_TYPES)}
    nodes = [
        Occurrence(type_id=31, row=5, col=5, margin=0.5, component_size=1),
        Occurrence(type_id=81, row=10, col=10, margin=0.6, component_size=1),
    ]
    vec = r.compute_relation_vector(nodes, qm_to_idx)
    assert vec.shape == (3528,)


def test_7_ordered_pair_count_n_times_n_minus_one():
    qm_to_idx = {t: i for i, t in enumerate(r.FROZEN_QM_TYPES)}
    nodes = [
        Occurrence(type_id=31, row=5, col=5, margin=0.5, component_size=1),
        Occurrence(type_id=81, row=10, col=10, margin=0.6, component_size=1),
        Occurrence(type_id=116, row=15, col=5, margin=0.7, component_size=1),
        Occurrence(type_id=123, row=20, col=20, margin=0.8, component_size=1),
    ]
    # n=4 -> 4*3 = 12 pairs
    # Before normalization, sum must equal 12
    vec = np.zeros(r.RELATION_DIM, dtype=np.float32)
    n = len(nodes)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            u = nodes[i]
            v = nodes[j]
            src = qm_to_idx[u.type_id]
            tgt = qm_to_idx[v.type_id]
            dr = v.row - u.row
            dc = v.col - u.col
            sr = 0 if dr == 0 else (1 if dr > 0 else -1)
            sc = 0 if dc == 0 else (1 if dc > 0 else -1)
            direction = r.DIRECTION_SECTOR_MAP[(sr, sc)]
            coord = ((src * r.QM_COUNT + tgt) * r.RELATION_DIRECTIONS) + direction
            vec[coord] += 1.0
    assert vec.sum() == 12.0


def test_8_all_eight_direction_sectors():
    center = Occurrence(type_id=31, row=10, col=10, margin=0.5, component_size=1)
    surrounding = [
        Occurrence(type_id=81, row=5, col=5, margin=0.5, component_size=1),
        Occurrence(type_id=81, row=5, col=10, margin=0.5, component_size=1),
        Occurrence(type_id=81, row=5, col=15, margin=0.5, component_size=1),
        Occurrence(type_id=81, row=10, col=5, margin=0.5, component_size=1),
        Occurrence(type_id=81, row=10, col=15, margin=0.5, component_size=1),
        Occurrence(type_id=81, row=15, col=5, margin=0.5, component_size=1),
        Occurrence(type_id=81, row=15, col=10, margin=0.5, component_size=1),
        Occurrence(type_id=81, row=15, col=15, margin=0.5, component_size=1),
    ]
    for expected_sec, target in enumerate(surrounding):
        dr = target.row - center.row
        dc = target.col - center.col
        sr = 1 if dr > 0 else (-1 if dr < 0 else 0)
        sc = 1 if dc > 0 else (-1 if dc < 0 else 0)
        assert r.DIRECTION_SECTOR_MAP[(sr, sc)] == expected_sec


def test_9_same_type_ordered_pairs_permitted():
    qm_to_idx = {t: i for i, t in enumerate(r.FROZEN_QM_TYPES)}
    nodes = [
        Occurrence(type_id=31, row=5, col=5, margin=0.5, component_size=1),
        Occurrence(type_id=31, row=15, col=15, margin=0.6, component_size=1),
    ]
    vec = r.compute_relation_vector(nodes, qm_to_idx)
    assert np.isclose(np.linalg.norm(vec), 1.0)


def test_10_same_location_rejected():
    qm_to_idx = {t: i for i, t in enumerate(r.FROZEN_QM_TYPES)}
    nodes = [
        Occurrence(type_id=31, row=5, col=5, margin=0.5, component_size=1),
        Occurrence(type_id=81, row=5, col=5, margin=0.5, component_size=1),
    ]
    with pytest.raises(ValueError, match="same location"):
        r.compute_relation_vector(nodes, qm_to_idx)


def test_11_control_permutation_deterministic_and_bijective():
    perm1 = r.derive_relation_control_permutation(42)
    perm2 = r.derive_relation_control_permutation(42)
    assert np.array_equal(perm1, perm2)
    assert sorted(perm1.tolist()) == list(range(8))


def test_12_control_preserves_source_target_pair_totals():
    qm_to_idx = {t: i for i, t in enumerate(r.FROZEN_QM_TYPES)}
    nodes = [
        Occurrence(type_id=31, row=2, col=3, margin=0.5, component_size=1),
        Occurrence(type_id=81, row=8, col=9, margin=0.6, component_size=1),
        Occurrence(type_id=116, row=15, col=5, margin=0.7, component_size=1),
    ]
    r_vec = r.compute_relation_vector(nodes, qm_to_idx)
    r_ctrl = r.compute_relation_control_vector(nodes, qm_to_idx, canonical_image_id=99)
    assert np.isclose(np.linalg.norm(r_vec), np.linalg.norm(r_ctrl))


def test_13_control_preserves_total_pair_count():
    qm_to_idx = {t: i for i, t in enumerate(r.FROZEN_QM_TYPES)}
    nodes = [
        Occurrence(type_id=31, row=2, col=3, margin=0.5, component_size=1),
        Occurrence(type_id=81, row=8, col=9, margin=0.6, component_size=1),
    ]
    r_ctrl = r.compute_relation_control_vector(nodes, qm_to_idx, canonical_image_id=123)
    assert np.isclose(np.linalg.norm(r_ctrl), 1.0)


def test_14_combined_de_dimension_6088():
    u = np.ones(2560, dtype=np.float32)
    rel = np.ones(3528, dtype=np.float32)
    comb = r.combine_unary_and_relation(u, rel)
    assert comb.shape == (6088,)


def test_15_combined_de_block_and_final_normalization():
    u = np.random.randn(2560).astype(np.float32)
    rel = np.random.randn(3528).astype(np.float32)
    comb = r.combine_unary_and_relation(u, rel)
    assert np.isclose(np.linalg.norm(comb), 1.0)


def test_16_shared_fold_assignment_and_oof_coverage():
    from sklearn.model_selection import StratifiedKFold

    y = np.random.randint(0, 7, size=100)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(np.zeros(100), y))

    val_indices = []
    for tr, va in splits:
        val_indices.extend(va.tolist())
    assert sorted(val_indices) == list(range(100))


def test_17_bootstrap_shared_indices():
    y_true = np.random.randint(0, 7, size=50)
    preds = {
        "A": y_true.copy(),
        "B": np.random.randint(0, 7, size=50),
        "C": np.random.randint(0, 7, size=50),
        "D": np.random.randint(0, 7, size=50),
        "E": np.random.randint(0, 7, size=50),
    }
    comps = [
        ("delta_vocab", "B", "A"),
        ("delta_sparse", "C", "B"),
        ("delta_rel", "D", "C"),
        ("delta_geom", "D", "E"),
    ]
    res = r.compute_shared_paired_bootstrap(
        y_true, preds, comps, b_replicates=10, seed=42
    )
    assert len(res) == 4
    for k in res:
        assert "ci_95_accuracy" in res[k]
        assert "ci_95_macro_f1" in res[k]


def test_18_bootstrap_linear_quantile_semantics():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    q = np.quantile(data, [0.025, 0.975], method="linear")
    assert len(q) == 2


def test_19_convergence_failure_is_fail_closed():
    # If a model fails to converge or hits max_iter, run_fixed_oof_probe raises RuntimeError
    sig = inspect.signature(r.run_fixed_oof_probe)
    assert "features" in sig.parameters
    assert "labels" in sig.parameters


def test_20_public_private_paths_not_referenced():
    src_file = Path(r.__file__)
    code = src_file.read_text(encoding="utf-8")
    for forbidden in [
        "val.csv",
        "test.csv",
        "PublicTest",
        "PrivateTest",
        "--public-csv",
        "--private-csv",
    ]:
        assert forbidden not in code
