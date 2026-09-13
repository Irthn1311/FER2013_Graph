import numpy as np
import pytest

from pixel_relational_motif_e0.e01_occurrence_runner import (
    FrozenDescriptorTransform,
    calibrate_tau_from_scores,
    occurrence_count_diagnostics,
    run_occurrence_calibration,
)
from pixel_relational_motif_e0.occurrence import Occurrence, apply_occurrence_policy, choose_tau_for_budget


def _occurrences(score_arrays):
    out = []
    for image_i, scores in enumerate(score_arrays):
        out.append(
            [
                Occurrence(motif=0, confidence=float(s), y=image_i, x=j)
                for j, s in enumerate(scores)
            ]
        )
    return out


def test_fast_tau_matches_locked_bruteforce_for_odd_image_count_with_ties():
    scores = [
        np.array([0.9, 0.8, 0.7, 0.6]),
        np.array([0.9, 0.8, 0.8, 0.4]),
        np.array([0.95, 0.2]),
        np.array([0.7, 0.6, 0.5, 0.4, 0.3]),
        np.array([0.99, 0.98, 0.1]),
    ]
    expected = choose_tau_for_budget(_occurrences(scores), median_budget=2)
    observed = calibrate_tau_from_scores(scores, median_budget=2)
    assert observed == expected


def test_fast_tau_rejects_even_count_instead_of_approximating_even_median():
    with pytest.raises(ValueError):
        calibrate_tau_from_scores([np.array([0.9]), np.array([0.8])], median_budget=0)


def test_occurrence_count_diagnostics_matches_policy_counts_and_flags():
    scores = [
        np.array([0.9, 0.8, 0.7, 0.6]),
        np.array([0.9]),
        np.array([0.95, 0.85, 0.1]),
    ]
    tau = 0.75
    diagnostics, final_counts, threshold_counts, candidate_counts, fallback_flags = (
        occurrence_count_diagnostics(scores, tau=tau, cap=2, minimum_nodes=2)
    )
    occurrences = _occurrences(scores)
    expected = [apply_occurrence_policy(xs, tau=tau, cap=2, minimum_nodes=2) for xs in occurrences]
    assert np.array_equal(final_counts, np.array([len(x[0]) for x in expected]))
    assert np.array_equal(fallback_flags, np.array([x[2] for x in expected]))
    assert np.array_equal(threshold_counts, np.array([2, 1, 2]))
    assert np.array_equal(candidate_counts, np.array([4, 1, 3]))
    assert diagnostics["median_nodes"] == float(np.median(final_counts))


def test_frozen_descriptor_transform_matches_pca_formula():
    rng = np.random.default_rng(4)
    s = rng.normal(size=(8, 24))
    mean = rng.normal(size=24)
    components = rng.normal(size=(11, 24))
    l = rng.normal(size=8)
    tr = FrozenDescriptorTransform(mean, components, log_sigma_mean=0.25, log_sigma_std=1.5)
    got = tr.transform(s, l)
    expected_p = (s - mean[None, :]) @ components.T
    expected_z = (l - 0.25) / 1.5
    expected = np.concatenate([expected_p, expected_z[:, None]], axis=1)
    assert np.allclose(got, expected)


def test_no_stable_components_still_writes_required_counts_artifact(tmp_path, monkeypatch):
    from pixel_relational_motif_e0.e01_occurrence_runner import DictionaryArtifact
    from pixel_relational_motif_e0.diag_gmm import DiagonalGaussianMixture
    from pixel_relational_motif_e0.e01_runner import TrainData, TRAIN_ROWS

    transform = FrozenDescriptorTransform(
        np.zeros(24), np.zeros((11, 24)), log_sigma_mean=0.0, log_sigma_std=1.0
    )
    artifact = DictionaryArtifact(
        "train-sha", transform, DiagonalGaussianMixture(2, 1e-6), np.empty(0, dtype=np.int64), 2
    )
    train = TrainData(np.empty((TRAIN_ROWS, 0), dtype=np.uint8), "train-sha")
    monkeypatch.setattr(
        "pixel_relational_motif_e0.e01_occurrence_runner.load_dictionary_artifact",
        lambda _: artifact,
    )
    monkeypatch.setattr(
        "pixel_relational_motif_e0.e01_occurrence_runner.load_official_train_images",
        lambda _: train,
    )

    summary = run_occurrence_calibration("train.csv", "dictionary.npz", tmp_path)
    counts_path = tmp_path / "e01_occurrence_counts.npz"
    assert summary["status"] == "NO_STABLE_COMPONENTS"
    assert summary["counts_artifact"] == counts_path.name
    with np.load(counts_path, allow_pickle=False) as counts:
        assert str(counts["status"].item()) == "NO_STABLE_COMPONENTS"
        assert counts["final_node_counts"].size == 0
