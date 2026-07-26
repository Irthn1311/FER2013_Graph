import numpy as np

from _helpers import GOLDEN, loaded


def test_layer_parity():
    model, batch = loaded()
    actual = model(batch, training=False, collect_intermediates=True)["intermediates"]
    with np.load(GOLDEN / "layer_outputs.npz", allow_pickle=False) as expected:
        for name in expected.files:
            if name in actual:
                assert np.max(np.abs(actual[name].numpy() - expected[name])) <= 1e-5, name

