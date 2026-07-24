from lap_gnn_tf.constants import EDGE_FEATURE_NAMES


def test_edge_schema():
    assert len(EDGE_FEATURE_NAMES) == 8

