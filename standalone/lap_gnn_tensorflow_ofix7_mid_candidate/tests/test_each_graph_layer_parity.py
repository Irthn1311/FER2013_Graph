import numpy as np

from _helpers import GOLDEN, loaded


def test_each_graph_layer_parity():
    model, batch = loaded()
    actual = model(batch, training=False, collect_intermediates=True)["intermediates"]
    with np.load(GOLDEN / "layer_outputs.npz") as expected:
        for index in range(1, 4):
            name = f"gnn_layer_{index}"
            assert np.max(np.abs(actual[name].numpy() - expected[name])) <= 1e-5

