import numpy as np
from pixel_relational_motif_e0.diag_gmm import DiagonalGaussianMixture, variance_floor_from_data


def test_gmm_floor_posteriors_and_likelihood_are_valid():
    rng = np.random.default_rng(4)
    x = np.r_[rng.normal(-2, .2, (150, 3)), rng.normal(2, .3, (150, 3))]
    floor = variance_floor_from_data(x, fraction=.01)
    g = DiagonalGaussianMixture(2, floor, max_iter=50, n_init=2, random_state=5, batch_size=64).fit(x)
    assert np.all(g.variances_ >= floor[None, :] - 1e-12)
    p = g.predict_proba(x[:40])
    assert p.shape == (40, 2)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-10)
    assert np.all(np.isfinite(g.score_samples(x[:40])))
    assert np.isfinite(g.score(x))


def test_weighted_fit_never_reseeds_or_scales_from_zero_weight_rows():
    # The extreme row represents an image excluded by an image bootstrap.
    # It must not influence initialization variance or dead-component reseeding.
    x = np.array([[0.0], [1.0], [1_000_000.0]], dtype=np.float64)
    w = np.array([1.0, 1.0, 0.0], dtype=np.float64)
    g = DiagonalGaussianMixture(
        2,
        variance_floor=np.array([1e-6]),
        max_iter=3,
        n_init=1,
        random_state=7,
        batch_size=3,
        min_component_weight=0.6,
    ).fit(x, sample_weight=w)
    assert np.max(np.abs(g.means_)) <= 1.0
    assert np.max(g.variances_) < 10.0
