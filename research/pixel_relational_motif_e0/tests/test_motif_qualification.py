import inspect
import json
import numpy as np
import pytest
from scipy import sparse

from pixel_relational_motif_e0 import motif_qualification as q
from pixel_relational_motif_e0 import motif_train_runner as r


def test_registered_constants_are_frozen():
    assert q.STABILITY_REPLICATES == 20
    assert q.FIT_FRACTION == 0.80
    assert q.SUPPORT_THRESHOLD == 288
    assert q.JACCARD_THRESHOLD == 0.75
    assert q.MAX_OCCURRENCES_PER_IMAGE == 64
    assert q.SPARSE_FEATURE_DIM == 2560
    assert r.SKM_N_INIT == 3
    assert r.SKM_MAX_ITER == 500
    assert r.SKM_TOL == 1e-6


@pytest.mark.parametrize("arm,replicate_id", [("M", 0), ("M", 19), ("C", 0), ("C", 19)])
def test_image_split_is_deterministic_disjoint_and_exact_80_20(arm, replicate_id):
    ids = np.arange(28_709, dtype=np.int32)
    fit_a, held_a = q.image_level_partition(ids, arm=arm, replicate_id=replicate_id)
    fit_b, held_b = q.image_level_partition(ids, arm=arm, replicate_id=replicate_id)
    assert np.array_equal(fit_a, fit_b)
    assert np.array_equal(held_a, held_b)
    assert len(fit_a) == 22_967
    assert len(held_a) == 5_742
    assert len(np.intersect1d(fit_a, held_a)) == 0
    assert np.array_equal(np.sort(np.concatenate([fit_a, held_a])), ids)


def test_seed_namespace_differs_by_arm_and_replicate():
    seeds = {
        q.stability_seed(arm, replicate)
        for arm in ("M", "C")
        for replicate in range(20)
    }
    assert len(seeds) == 40


def test_replicate_clustering_is_from_scratch_and_does_not_accept_original_centers():
    signature = inspect.signature(r._registered_replicate_skm)
    assert "original_centers" not in signature.parameters
    traces = []
    model = r._registered_replicate_skm(arm="M", replicate_id=3, diagnostics=traces)
    assert model.n_clusters == 512
    assert model.n_init == 3
    assert model.max_iter == 500
    assert model.random_state == q.stability_seed("M", 3)


def _basis_centers():
    centers = np.zeros((512, 1152), dtype=np.float32)
    centers[np.arange(512), np.arange(512)] = 1.0
    return centers


def test_hungarian_is_centroid_only_and_returns_permutations():
    original = _basis_centers()
    permutation = np.roll(np.arange(512), 17)
    replicate = original[permutation]
    match = q.centroid_only_hungarian_match(original, replicate)
    assert set(inspect.signature(q.centroid_only_hungarian_match).parameters) == {
        "original_centers",
        "replicate_centers",
    }
    assert np.array_equal(np.sort(match.original_to_replicate), np.arange(512))
    assert np.array_equal(np.sort(match.replicate_to_original), np.arange(512))
    assert np.allclose(match.matched_cosine, 1.0)


def test_exact_cosine_assignments_and_margins_match_dense_and_csr():
    rng = np.random.default_rng(1)
    centers = rng.normal(size=(7, 13)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    descriptors = rng.normal(size=(19, 13)).astype(np.float32)
    dense = q.exact_cosine_assignments_and_margins(descriptors, centers, batch_size=5)
    csr = q.exact_cosine_assignments_and_margins(
        sparse.csr_matrix(descriptors), centers, batch_size=5
    )
    assert np.array_equal(dense[0], csr[0])
    assert np.allclose(dense[1], csr[1], atol=1e-7)
    normalized = descriptors / np.linalg.norm(descriptors, axis=1, keepdims=True)
    similarity = normalized @ centers.T
    ordered = np.sort(similarity, axis=1)
    assert np.array_equal(dense[0], np.argmax(similarity, axis=1))
    assert np.allclose(dense[1], ordered[:, -1] - ordered[:, -2], atol=1e-7)


def test_heldout_dense_evaluator_consumes_all_1296_positions_and_no_sample_pool(
    monkeypatch,
):
    calls = []

    def fake_descriptors(_primitive_map, normalize=True):
        assert normalize
        return np.ones((1296, 1152), dtype=np.float32)

    def fake_assign(descriptors, _centers, batch_size=8192):
        calls.append(descriptors.shape[0])
        labels = np.arange(descriptors.shape[0], dtype=np.int32) % 7
        return labels, np.ones(descriptors.shape[0], dtype=np.float32)

    monkeypatch.setattr(q, "composition_descriptors", fake_descriptors)
    monkeypatch.setattr(q, "exact_cosine_assignments_and_margins", fake_assign)
    maps = np.zeros((2, 44, 44), dtype=np.int16)
    result = q.heldout_dense_jaccard(
        maps,
        np.array([2, 9], dtype=np.int32),
        arm="M",
        original_centers=_basis_centers(),
        replicate_centers=_basis_centers(),
        replicate_to_original=np.arange(512, dtype=np.int32),
    )
    assert calls == [1296, 1296, 1296, 1296]
    assert np.all(result[:7] == 1)
    assert np.all(result[7:] == 0)
    source = inspect.getsource(q.heldout_dense_jaccard)
    assert "sampled_training_descriptors" not in source
    assert "CRS_VALID_POSITIONS" in source


def test_control_heldout_evaluator_uses_train_keyed_permutation(monkeypatch):
    observed = []
    monkeypatch.setattr(
        q,
        "composition_descriptors",
        lambda *_args, **_kwargs: np.ones((1296, 1152), dtype=np.float32),
    )

    def fake_permute(value, **kwargs):
        observed.append(
            (kwargs["split_id"], kwargs["image_id"], len(kwargs["center_indices"]))
        )
        return value

    monkeypatch.setattr(q, "permute_cell_blocks", fake_permute)
    monkeypatch.setattr(
        q,
        "exact_cosine_assignments_and_margins",
        lambda descriptors, _centers: (
            np.zeros(descriptors.shape[0], dtype=np.int32),
            np.ones(descriptors.shape[0], dtype=np.float32),
        ),
    )
    q.heldout_dense_jaccard(
        np.zeros((1, 44, 44), dtype=np.int16),
        np.array([11], dtype=np.int32),
        arm="C",
        original_centers=_basis_centers(),
        replicate_centers=_basis_centers(),
        replicate_to_original=np.arange(512, dtype=np.int32),
    )
    assert observed == [("train", 11, 1296)]


def test_jaccard_correctness_and_empty_union_zero():
    actual = np.array([0, 0, 1, 1, 1], dtype=np.int32)
    replicate = np.array([0, 1, 1, 1, 2], dtype=np.int32)
    observed = q.cluster_jaccard(actual, replicate, n_types=4)
    assert np.allclose(observed, [0.5, 0.5, 0.0, 0.0])


def test_support_counts_distinct_images_not_positions():
    assignments = np.zeros((3, 36, 36), dtype=np.int16)
    assignments[0, :, :] = 7
    assignments[1, 0, 0] = 7
    assignments[1, 0, 1] = 8
    assignments[2, :, :] = 8
    support = q.distinct_image_support(assignments)
    assert support[7] == 2
    assert support[8] == 2
    assert support[0] == 1


def test_m_and_c_use_identical_qualification_rule_and_strict_thresholds():
    support = np.zeros(512, dtype=np.int32)
    stability = np.zeros(512, dtype=np.float64)
    support[:4] = [287, 288, 288, 1000]
    stability[:4] = [1.0, 0.749, 0.75, 0.9]
    expected = np.array([2, 3], dtype=np.int32)
    assert np.array_equal(q.qualified_type_ids(support, stability), expected)
    assert np.array_equal(
        q.qualified_type_ids(support.copy(), stability.copy()), expected
    )


def test_jaccard_summary_stores_all_registered_order_statistics():
    raw = np.repeat(np.linspace(0.0, 1.0, 20)[:, None], 512, axis=1)
    summary = q.summarize_jaccards(raw)
    assert set(summary) == {"median", "q1", "q3", "min", "max"}
    assert all(value.shape == (512,) for value in summary.values())
    assert np.allclose(summary["median"], 0.5)
    assert np.allclose(summary["min"], 0.0)
    assert np.allclose(summary["max"], 1.0)


def test_q_star_ranking_is_symmetric_and_never_introduces_unqualified_types():
    m_support = np.zeros(512, dtype=np.int32)
    c_support = np.zeros(512, dtype=np.int32)
    m_median = np.zeros(512)
    c_median = np.zeros(512)
    q_m = np.array([9, 4, 7], dtype=np.int32)
    q_c = np.array([12, 2], dtype=np.int32)
    m_median[q_m] = [0.9, 0.9, 0.8]
    m_support[q_m] = [400, 500, 900]
    c_median[q_c] = [0.81, 0.95]
    c_support[q_c] = [900, 300]
    m_match, c_match, q_star = q.matched_vocabularies(
        q_m,
        q_c,
        m_support=m_support,
        c_support=c_support,
        m_median=m_median,
        c_median=c_median,
    )
    assert q_star == 2
    assert np.array_equal(m_match, [4, 9])
    assert np.array_equal(c_match, [2, 12])
    assert set(c_match).issubset(set(q_c))


def test_eight_connected_components_and_max_margin_representative():
    assignments = np.zeros((36, 36), dtype=np.int32)
    assignments[4, 4] = assignments[5, 5] = 3
    margins = np.zeros((36, 36), dtype=np.float32)
    margins[4, 4] = 0.2
    margins[5, 5] = 0.8
    result = q.extract_occurrences(assignments, margins, np.array([3]))
    assert result.pre_cap_count == 1
    assert np.array_equal(result.component_sizes, [2])
    occurrence = result.retained[0]
    assert (occurrence.row, occurrence.col, occurrence.component_size) == (5, 5, 2)


def test_occurrence_tie_breaking_prefers_type_then_row_major():
    assignments = np.full((36, 36), 511, dtype=np.int32)
    assignments[0, 0] = assignments[0, 2] = 9
    assignments[3, 3] = 4
    margins = np.zeros((36, 36), dtype=np.float32)
    margins[0, 0] = margins[0, 2] = margins[3, 3] = 0.5
    result = q.extract_occurrences(assignments, margins, np.array([9, 4]))
    assert [(x.type_id, x.row, x.col) for x in result.retained] == [
        (4, 3, 3),
        (9, 0, 0),
        (9, 0, 2),
    ]


def test_global_top_64_cap_is_exact_and_deterministic():
    assignments = np.ones((36, 36), dtype=np.int32)
    margins = np.zeros((36, 36), dtype=np.float32)
    rank = 0
    for row in range(0, 36, 2):
        for col in range(0, 36, 2):
            assignments[row, col] = 0
            margins[row, col] = rank / 1000.0
            rank += 1
    first = q.extract_occurrences(assignments, margins, np.array([0]))
    second = q.extract_occurrences(assignments, margins, np.array([0]))
    assert first.pre_cap_count == 324
    assert len(first.retained) == 64
    assert first.retained == second.retained
    assert all(
        first.retained[i].margin >= first.retained[i + 1].margin for i in range(63)
    )


def test_occurrence_diagnostics_cover_component_and_cap_contracts():
    assignments = np.ones((36, 36), dtype=np.int32)
    assignments[0, 0] = 2
    margins = np.ones((36, 36), dtype=np.float32)
    one = q.extract_occurrences(assignments, margins, np.array([2]))
    zero = q.extract_occurrences(assignments, margins, np.array([], dtype=np.int32))
    observed = q.occurrence_diagnostics([one, zero])
    assert observed["component_summary"]["count"] == 1
    assert observed["component_summary"]["fraction_size_1"] == 1.0
    assert observed["zero_node_image_count"] == 1
    assert observed["cap_binding_fraction"] == 0.0


def test_sparse_pyramid_dimension_norm_and_zero_node_semantics():
    zero = q.sparse_occurrence_feature(())
    assert zero.shape == (2560,)
    assert np.count_nonzero(zero) == 0
    occurrences = (
        q.Occurrence(3, 0, 0, 0.4, 1),
        q.Occurrence(3, 35, 35, 0.3, 2),
        q.Occurrence(7, 0, 35, 0.2, 1),
    )
    feature = q.sparse_occurrence_feature(occurrences)
    assert feature.shape == (2560,)
    assert np.all(np.isfinite(feature))
    assert np.isclose(np.linalg.norm(feature), 1.0)
    assert np.count_nonzero(feature) == 5


def _all_parser_options(parser):
    options = set()
    for action in parser._actions:
        options.update(action.option_strings)
        if hasattr(action, "choices") and isinstance(action.choices, dict):
            for child in action.choices.values():
                options.update(_all_parser_options(child))
    return options


def test_cli_exposes_orchestration_only_and_no_public_or_private_input():
    options = _all_parser_options(r._build_parser())
    assert {"--arm", "--replicate-id", "--substrate-dir", "--replicates-dir"}.issubset(
        options
    )
    for forbidden in (
        "--k",
        "--seed",
        "--n-init",
        "--max-iter",
        "--support",
        "--jaccard-cutoff",
        "--public-csv",
        "--private-csv",
        "--test-csv",
    ):
        assert forbidden not in options


def test_sparse_probe_configuration_is_identical_for_all_arms(monkeypatch):
    captured = []

    class FakeClassifier:
        def __init__(self, **kwargs):
            captured.append(kwargs)
            self.classes_ = np.arange(7)
            self.coef_ = np.zeros((7, 2560))
            self.intercept_ = np.zeros(7)
            self.n_iter_ = np.array([1])

        def fit(self, _x, _y):
            return self

        def predict(self, x):
            return np.zeros(x.shape[0], dtype=np.int64)

    monkeypatch.setattr(r, "TRAIN_ROWS", 7)
    monkeypatch.setattr(r, "LogisticRegression", FakeClassifier)
    x = sparse.csr_matrix((7, 2560), dtype=np.float32)
    y = np.arange(7)
    for arm in ("S_M_all", "S_M_match", "S_C_match"):
        state = r._fit_sparse_probe(x, y, arm=arm)
        assert state.converged
    assert (
        captured[0]
        == captured[1]
        == captured[2]
        == {
            "C": 1.0,
            "solver": "lbfgs",
            "class_weight": "balanced",
            "max_iter": 5000,
            "tol": 1e-4,
            "random_state": 42,
        }
    )


def test_labels_are_accessed_after_sparse_features_and_diagnostics_are_frozen():
    source = inspect.getsource(r.finalize_train_stage)
    label_access = source.index("load_labels_downstream(")
    assert source.index("_build_sparse_features(") < label_access
    assert source.index("_atomic_sparse(path, matrix)") < label_access
    assert (
        source.index("_atomic_savez(out / OCCURRENCE_DIAGNOSTICS_FILE") < label_access
    )
    assert 'role="train"' in source


def test_atomic_artifact_roundtrip_hash_and_overwrite_protection(tmp_path):
    path = tmp_path / "artifact.npz"
    payload = {"x": np.arange(12, dtype=np.int32).reshape(3, 4)}
    r._atomic_savez(path, payload)
    digest = r.sha256_file(path)
    with np.load(path, allow_pickle=False) as restored:
        assert np.array_equal(restored["x"], payload["x"])
    assert digest == r.sha256_file(path)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        r._atomic_savez(path, payload)


def test_substrate_manifest_roundtrip_and_hash_validation(tmp_path):
    artifact = tmp_path / "tiny.bin"
    artifact.write_bytes(b"immutable")
    identity_payload = {
        "scientific_source_sha": "a" * 40,
        "preregistration_sha": q.PREREGISTRATION_SHA,
        "train_sha256": r.TRAIN_SHA256,
        "dictionary_sha256": "b" * 64,
        "accepted_train_model_sha256": r.ACCEPTED_TRAIN_MODEL_SHA256,
        "files": {"tiny.bin": r._file_record(artifact)},
    }
    manifest = {
        "issue": 86,
        **identity_payload,
        "substrate_identity_sha256": __import__("hashlib")
        .sha256(r._canonical_json_bytes(identity_payload))
        .hexdigest(),
    }
    (tmp_path / r.SUBSTRATE_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    assert (
        r.load_substrate_manifest(tmp_path)["substrate_identity_sha256"]
        == manifest["substrate_identity_sha256"]
    )
    artifact.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="artifact identity mismatch"):
        r.load_substrate_manifest(tmp_path)


def test_train_runner_source_has_explicit_isolation_flags():
    source = inspect.getsource(r)
    assert '"public_test_accessed": False' in source
    assert '"private_test_accessed": False' in source
    assert "test.csv" not in source
