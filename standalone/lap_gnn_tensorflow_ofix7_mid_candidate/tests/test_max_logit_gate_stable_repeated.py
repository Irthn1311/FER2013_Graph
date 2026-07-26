import numpy as np
import tensorflow as tf

from _helpers import golden, loaded


def test_max_logit_gate_stable_repeated():
    model, batch = loaded()
    expected = golden("logits.npy")
    eager = [model(batch, training=False)["logits"].numpy() for _ in range(10)]

    @tf.function(autograph=False)
    def compiled(value):
        return model(value, training=False)["logits"]

    compiled(batch)
    graph = [compiled(batch).numpy() for _ in range(10)]
    for actual in eager + graph:
        assert np.max(np.abs(actual - expected)) <= 1e-5
        assert np.array_equal(actual.argmax(axis=1), expected.argmax(axis=1))
