import numpy as np
from pixel_relational_motif_e0.occurrence import stable_score_map, local_max_nms, choose_tau_for_budget, apply_occurrence_policy, Occurrence
from pixel_relational_motif_e0.features import dense_posterior_histogram, geometry_pair_tensor
from pixel_relational_motif_e0.controls import shuffle_geometry_bins


def test_stable_score_does_not_renormalize():
    p = np.zeros((44*44, 3), dtype=float)
    p[:,0] = .8; p[:,1] = .15; p[:,2] = .05
    score, motif = stable_score_map(p, np.array([1,2]))
    assert np.allclose(score, .15)
    assert np.all(motif == 1)


def test_nms_tau_and_fallback_deterministic():
    score = np.zeros((44,44)); motif = np.zeros((44,44), dtype=int)
    score[2,2] = .9; score[2,3] = .8; score[20,20] = .7
    a = local_max_nms(score, motif, radius=2)
    b = local_max_nms(score, motif, radius=2)
    assert a == b
    assert any(o.confidence == .9 for o in a)
    lists = [[Occurrence(0, c, i, i) for i,c in enumerate([.9,.7,.5])], [Occurrence(0, c, i, i) for i,c in enumerate([.8,.6,.4])]]
    tau = choose_tau_for_budget(lists, median_budget=2)
    counts = [sum(o.confidence >= tau for o in xs) for xs in lists]
    assert np.median(counts) <= 2
    kept, _, fallback = apply_occurrence_policy(lists[0], tau=.95, minimum_nodes=2)
    assert fallback and len(kept) == 2


def test_histogram_and_geometry_shuffle_preserve_mass():
    p = np.array([[.2,.8],[.6,.4]])
    h = dense_posterior_histogram(p)
    assert np.isclose(h.sum(), 1)
    occ = [Occurrence(3,.9,2,2), Occurrence(7,.8,10,15), Occurrence(3,.7,20,5)]
    t = geometry_pair_tensor(occ, stable_components=np.array([3,7]), distance_median=.3)
    assert np.isclose(t.sum(), 1.0)
    s = shuffle_geometry_bins(t, image_index=12, seed=5)
    assert np.isclose(s.sum(), t.sum())
    assert np.allclose(s.sum(axis=-1), t.sum(axis=-1))
