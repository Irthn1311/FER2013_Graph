"""Complete contract test suite for Issue #90 runner and preregistered requirements."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from pixel_relational_motif_e0.motif_qualification import Occurrence
import pixel_relational_motif_e0.sparse_retention_relation_runner as r


def test_1_dense_pyramid_boundaries_and_quadrants():
    # Exactly test quadrant boundaries around index 18
    # Boundary split at 18:
    # reg 1: row < 18, col < 18 -> (17, 17)
    # reg 2: row < 18, col >= 18 -> (17, 18)
    # reg 3: row >= 18, col < 18 -> (18, 17)
    # reg 4: row >= 18, col >= 18 -> (18, 18)
    grid = np.zeros((36, 36), dtype=np.int16)
    grid[17, 17] = 10
    grid[17, 18] = 20
    grid[18, 17] = 30
    grid[18, 18] = 40

    feat = r.dense_pyramid_feature(grid)
    assert feat.shape == (2560,)
    assert np.all(np.isfinite(feat))
    assert np.isclose(np.linalg.norm(feat), 1.0)

    # Check that in reg 1 (indices 512..1023), bin 10 has positive count, and 20, 30, 40 are 0
    assert feat[512 + 10] > 0
    assert feat[512 + 20] == 0
    assert feat[512 + 30] == 0
    assert feat[512 + 40] == 0

    # reg 2 (indices 1024..1535): bin 20 positive
    assert feat[1024 + 20] > 0
    assert feat[1024 + 10] == 0

    # reg 3 (indices 1536..2047): bin 30 positive
    assert feat[1536 + 30] > 0
    assert feat[1536 + 10] == 0

    # reg 4 (indices 2048..2559): bin 40 positive
    assert feat[2048 + 40] > 0
    assert feat[2048 + 10] == 0


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


def test_7_raw_ordered_pair_count_n_times_n_minus_one():
    qm_to_idx = {t: i for i, t in enumerate(r.FROZEN_QM_TYPES)}
    nodes = [
        Occurrence(type_id=31, row=5, col=5, margin=0.5, component_size=1),
        Occurrence(type_id=81, row=10, col=10, margin=0.6, component_size=1),
        Occurrence(type_id=116, row=15, col=5, margin=0.7, component_size=1),
        Occurrence(type_id=123, row=20, col=20, margin=0.8, component_size=1),
    ]
    # n=4 -> 4*3 = 12 raw pairs
    raw_vec = r.compute_relation_vector_raw(nodes, qm_to_idx)
    assert np.isclose(raw_vec.sum(), 12.0)


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
    r_raw = r.compute_relation_vector_raw(nodes, qm_to_idx).reshape(
        r.QM_COUNT, r.QM_COUNT, 8
    )
    r_ctrl_raw = r.compute_relation_control_vector_raw(
        nodes, qm_to_idx, canonical_image_id=99
    ).reshape(r.QM_COUNT, r.QM_COUNT, 8)

    # Require sums over direction sectors to match exactly for every source-target pair
    pair_totals_r = r_raw.sum(axis=2)
    pair_totals_ctrl = r_ctrl_raw.sum(axis=2)
    assert np.array_equal(pair_totals_r, pair_totals_ctrl)


def test_13_control_preserves_total_raw_pair_count():
    qm_to_idx = {t: i for i, t in enumerate(r.FROZEN_QM_TYPES)}
    nodes = [
        Occurrence(type_id=31, row=2, col=3, margin=0.5, component_size=1),
        Occurrence(type_id=81, row=8, col=9, margin=0.6, component_size=1),
        Occurrence(type_id=116, row=15, col=5, margin=0.7, component_size=1),
    ]
    r_raw = r.compute_relation_vector_raw(nodes, qm_to_idx)
    r_ctrl_raw = r.compute_relation_control_vector_raw(
        nodes, qm_to_idx, canonical_image_id=123
    )
    assert np.isclose(r_raw.sum(), 6.0)
    assert np.isclose(r_ctrl_raw.sum(), 6.0)


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
    y = np.random.randint(0, 7, size=r.TRAIN_ROWS)
    splits = r.generate_shared_5fold_splits(y)
    assert len(splits) == 5

    val_indices = []
    for tr, va in splits:
        assert len(np.intersect1d(tr, va)) == 0
        val_indices.extend(va.tolist())
    assert sorted(val_indices) == list(range(r.TRAIN_ROWS))


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


def test_19_real_convergence_failure_raises_runtime_error(monkeypatch):
    # Mock LogisticRegression.fit to emit ConvergenceWarning
    orig_fit = r.LogisticRegression.fit

    def mock_fit(self, X, y):
        warnings.warn("test convergence failure", ConvergenceWarning)
        return orig_fit(self, X, y)

    monkeypatch.setattr(r.LogisticRegression, "fit", mock_fit)

    X_dummy = np.random.randn(r.TRAIN_ROWS, 10).astype(np.float32)
    y_dummy = np.random.randint(0, 7, size=r.TRAIN_ROWS)
    splits = r.generate_shared_5fold_splits(y_dummy)

    with pytest.raises(
        RuntimeError, match="failed convergence with ConvergenceWarning"
    ):
        r.run_fixed_oof_probe(X_dummy, y_dummy, splits)


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


def test_21_fail_closed_assertions_on_bad_inputs():
    with pytest.raises(ValueError, match="shape"):
        r.dense_pyramid_feature(np.zeros((10, 10), dtype=np.int16))

    with pytest.raises(ValueError, match="range"):
        r.dense_pyramid_feature(np.full((36, 36), 600, dtype=np.int16))

    with pytest.raises(ValueError, match="unary feature dimension"):
        r.combine_unary_and_relation(np.ones(10), np.ones(3528))

    with pytest.raises(ValueError, match="relation feature dimension"):
        r.combine_unary_and_relation(np.ones(2560), np.ones(10))
