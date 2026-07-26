import numpy as np

from lap_gnn_tf.seed import seed_everything


def test_seed_determinism():
    seed_everything(42)
    first = np.random.random(8)
    seed_everything(42)
    second = np.random.random(8)
    assert np.array_equal(first, second)

