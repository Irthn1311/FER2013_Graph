from lap_gnn.constants import EDGE_FEATURE_NAMES


def test_edge_features_are_locked():
    assert EDGE_FEATURE_NAMES == [
        "dx", "dy", "spatial_dist", "abs_intensity_diff",
        "abs_grad_mag_diff", "abs_laplacian_diff",
        "part_similarity", "same_dominant_part",
    ]
