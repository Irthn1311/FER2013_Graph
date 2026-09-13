import numpy as np
from pixel_relational_motif_e0.diag_gmm import variance_floor_from_data
from pixel_relational_motif_e0.e0 import bootstrap_stability, nondegenerate_mask, fit_canonical_dictionary


def test_small_bootstrap_stability_and_canonical_pipeline_runs():
    rng = np.random.default_rng(13)
    image_ids = np.repeat(np.arange(6), 30)
    x = np.vstack([rng.normal(-2, .15, (90, 2)), rng.normal(2, .15, (90, 2))])
    order = rng.permutation(len(x)); x = x[order]; image_ids = image_ids[order]
    anchor = np.vstack([rng.normal(-2,.15,(80,2)), rng.normal(2,.15,(80,2))])
    floor = variance_floor_from_data(x)
    st = bootstrap_stability(x, image_ids, np.arange(6), anchor, k=2, variance_floor=floor, n_bootstrap=4, n_null=40, seed=4, gmm_max_iter=30, gmm_batch_size=128)
    assert st.match_distances.shape == (2,4)
    assert st.stable_mask.shape == (2,)
    counts = np.array([[10,10,10,10,10,10],[10,10,10,10,10,10]])
    nd, _, _ = nondegenerate_mask(counts, n_null=30, seed=2)
    can = fit_canonical_dictionary(x, variance_floor=floor, stability=st, nondegenerate=nd, n_init=2, seed=5, gmm_max_iter=30, gmm_batch_size=128)
    assert can.model.means_.shape == (2,2)
    assert np.all((can.stable_components >= 0) & (can.stable_components < 2))
