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
