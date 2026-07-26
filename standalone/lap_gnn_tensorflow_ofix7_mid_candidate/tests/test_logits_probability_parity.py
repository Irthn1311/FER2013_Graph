import numpy as np

from _helpers import golden, loaded


def test_logits_probability_parity():
    model, batch = loaded()
    output = model(batch, training=False)
    logits = output["logits"].numpy()
    expected = golden("logits.npy")
    assert np.array_equal(logits.argmax(1), expected.argmax(1))
    assert np.max(np.abs(logits - expected)) <= 1e-5
    assert np.max(np.abs(output["probabilities"].numpy() - golden("probabilities.npy"))) <= 2e-6
