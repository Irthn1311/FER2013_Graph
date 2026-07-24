import numpy as np

from _helpers import GOLDEN, loaded


def test_input_projection_parity():
    model, batch = loaded()
    actual = model(batch, training=False, collect_intermediates=True)["intermediates"]["input_projection"].numpy()
    with np.load(GOLDEN / "layer_outputs.npz") as expected:
        assert np.max(np.abs(actual - expected["input_projection"])) <= 1e-5

