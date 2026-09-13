import numpy as np
from pixel_relational_motif_e0.stability import symmetric_kl_diag, pairwise_symmetric_kl, hungarian_match, medoid_run, support_matched_null_statistic, benjamini_hochberg, effective_support


def test_symmetric_kl_properties():
    a = np.array([0., 1.]); va = np.array([1., 2.])
    b = np.array([1., -1.]); vb = np.array([2., 3.])
    assert symmetric_kl_diag(a, va, a, va) == 0.0
    ab = symmetric_kl_diag(a, va, b, vb)
    ba = symmetric_kl_diag(b, vb, a, va)
    assert ab >= 0 and np.isclose(ab, ba)


def test_hungarian_and_medoid_are_deterministic():
    means = [np.array([[0.], [5.]]), np.array([[.1], [5.1]]), np.array([[2.], [7.]])]
    vars_ = [np.ones((2,1)) for _ in means]
    cost = pairwise_symmetric_kl(means[0], vars_[0], means[1], vars_[1])
    rows, cols, _ = hungarian_match(cost)
    assert list(zip(rows, cols)) == [(0,0), (1,1)]
    assert medoid_run(means, vars_) == medoid_run(means, vars_)


def test_support_matched_null_keeps_requested_cardinalities_indirectly():
    ref = np.arange(10)
    sizes = np.array([5, 20, 30])
    out = support_matched_null_statistic(ref, sizes, universe_size=100, n_replicates=20, seed=8)
    assert out.shape == (20,)
    assert np.all((out >= 0) & (out <= 1))


def test_bh_is_deterministic_and_effective_support_behaves():
    p = np.array([.001, .01, .2, .9])
    r1, q1 = benjamini_hochberg(p)
    r2, q2 = benjamini_hochberg(p)
    assert np.array_equal(r1, r2) and np.array_equal(q1, q2)
    assert effective_support([1,1,1,1]) == 4.0
    assert effective_support([4,0,0,0]) == 1.0
