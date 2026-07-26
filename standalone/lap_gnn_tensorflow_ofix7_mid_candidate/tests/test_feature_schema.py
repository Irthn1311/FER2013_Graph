from lap_gnn_tf.constants import NODE_FEATURE_NAMES


def test_feature_schema():
    assert len(NODE_FEATURE_NAMES) == 37
    assert NODE_FEATURE_NAMES[:6] == ["intensity", "gx", "gy", "x_norm", "y_norm", "face_mask"]
    assert NODE_FEATURE_NAMES[-5:] == ["grad_mag", "local_mean_3x3", "local_std_3x3", "laplacian_abs", "center_surround"]

