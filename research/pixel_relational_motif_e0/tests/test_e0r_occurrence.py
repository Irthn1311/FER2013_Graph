import inspect

import numpy as np
import pytest

from pixel_relational_motif_e0.e0r_frozen import CONTROL_ARTIFACT_SHA256
from pixel_relational_motif_e0.e0r_occurrence import (
    CompactOccurrences,
    apply_no_fallback,
    brute_force_tau,
    exact_tau,
    full_k_score_maps,
    occurrence_diagnostics,
    occurrence_histograms,
)
from pixel_relational_motif_e0.e0r_runner import registered_joint_verdict


def _occ(components, confidence=None):
    components = np.asarray(components, dtype=np.int16)
    if confidence is None:
        confidence = np.linspace(1.0, 0.1, len(components))
    return CompactOccurrences(
        components,
        np.asarray(confidence, dtype=np.float64),
        np.arange(len(components), dtype=np.int16),
        np.arange(len(components), dtype=np.int16),
    )


def test_frozen_seed_hash_mapping_is_exact():
    assert tuple(CONTROL_ARTIFACT_SHA256) == (42, 43, 44, 45, 46)
    assert CONTROL_ARTIFACT_SHA256[42] == "ace39fc9f8bac39523080002acab547c4c65cf6015fda8b09101003238d5c1e6"
    assert CONTROL_ARTIFACT_SHA256[46] == "0ddea5e79dbe512e3af91f5d2778082832b6a12fe7f63192ce9ecc7a15e8d415"


def test_full_k_argmax_and_max_use_original_128_posteriors():
    posterior = np.zeros((1936, 128), dtype=np.float64)
    posterior[:, 3] = 0.2
    posterior[:, 77] = 0.6
    posterior[:, 100] = 0.1
    confidence, component = full_k_score_maps(posterior)
    assert confidence.shape == component.shape == (44, 44)
    assert np.all(component == 77)
    assert np.all(confidence == 0.6)


def test_full_k_score_requires_exact_1936_by_128():
    with pytest.raises(ValueError):
        full_k_score_maps(np.ones((1935, 128)))
    with pytest.raises(ValueError):
        full_k_score_maps(np.ones((1936, 127)))


def test_exact_tau_matches_brute_force_randomized_odd_and_even_cases():
    rng = np.random.default_rng(42)
    for n_images in range(1, 12):
        for _ in range(30):
            scores = [
                np.round(rng.random(int(rng.integers(0, 15))), 4)
                for _ in range(n_images)
            ]
            budget = int(rng.integers(0, 10))
            assert exact_tau(scores, median_budget=budget) == brute_force_tau(
                scores, median_budget=budget
            )


def test_separate_conditions_can_have_separate_train_only_taus():
    actual = [np.array([0.9, 0.8]), np.array([0.7, 0.6]), np.array([0.5, 0.4])]
    control = [x / 10.0 for x in actual]
    assert exact_tau(actual, median_budget=1) != exact_tau(control, median_budget=1)


def test_no_fallback_preserves_zero_and_one_and_cap_is_deterministic():
    candidates = _occ(np.arange(100) % 128, np.linspace(1.0, 0.01, 100))
    zero, zero_cap = apply_no_fallback(candidates, tau=2.0, cap=80)
    one, one_cap = apply_no_fallback(candidates, tau=1.0, cap=80)
    capped_a, cap_a = apply_no_fallback(candidates, tau=0.0, cap=80)
    capped_b, cap_b = apply_no_fallback(candidates, tau=0.0, cap=80)
    assert len(zero) == 0 and not zero_cap
    assert len(one) == 1 and not one_cap
    assert len(capped_a) == 80 and cap_a and cap_b
    assert np.array_equal(capped_a.components, capped_b.components)


def test_occurrence_histogram_full_k_normalization_and_zero_contract():
    features = occurrence_histograms([_occ([1, 1, 7]), _occ([])])
    assert features.shape == (2, 128)
    assert np.isclose(features[0, 1], 2 / 3)
    assert np.isclose(features[0, 7], 1 / 3)
    assert np.isclose(features[0].sum(), 1.0)
    assert features[1].sum() == 0.0


def test_diagnostics_define_prevalence_and_support_without_selection():
    diagnostics = occurrence_diagnostics([_occ([1, 1]), _occ([1, 2]), _occ([])], np.array([0, 1, 0]))
    prevalence = np.asarray(diagnostics["component_prevalence"])
    support = np.asarray(diagnostics["component_image_support_rate"])
    assert np.isclose(prevalence[1], 0.75)
    assert np.isclose(prevalence[2], 0.25)
    assert np.isclose(support[1], 2 / 3)
    assert np.isclose(support[2], 1 / 3)
    assert diagnostics["zero_node_rate"] == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    "accuracy,f1,expected",
    [
        ([0.01, 0.2], [0.02, 0.3], "E0.R1 SUPPORTED"),
        ([0.01, 0.2], [0.0, 0.3], "E0.R1 MIXED — NOT SUPPORTED BY REGISTERED JOINT CRITERION"),
        ([0.0, 0.2], [-0.1, 0.3], "E0.R1 NOT SUPPORTED"),
    ],
)
def test_r1_registered_joint_verdict_is_strict(accuracy, f1, expected):
    assert registered_joint_verdict(np.array(accuracy), np.array(f1), stage="R1") == expected


def test_no_pca_or_gmm_refit_call_exists_in_frozen_loader():
    import pixel_relational_motif_e0.e0r_frozen as frozen

    source = inspect.getsource(frozen)
    assert ".fit(" not in source

