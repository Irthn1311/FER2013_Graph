from lap_gnn_tf.resources import ResourceControls


def test_resource_controls():
    controls = ResourceControls()
    assert controls.batch_size == 16
    assert controls.xla is False
    assert controls.graph_workers == 2
    assert controls.tf_data_prefetch == 2

