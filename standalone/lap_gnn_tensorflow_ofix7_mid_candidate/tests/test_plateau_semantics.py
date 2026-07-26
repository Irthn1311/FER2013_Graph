import tensorflow as tf
import numpy as np

from lap_gnn_tf.training.plateau import TorchCompatibleReduceLROnPlateau


def test_plateau_semantics():
    optimizer = tf.keras.optimizers.SGD(learning_rate=3e-4)
    scheduler = TorchCompatibleReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=3e-5)
    values = [1.0, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]
    lrs = [scheduler.step(value) for value in values]
    np.testing.assert_allclose(lrs[:6], [3e-4] * 6, rtol=0.0, atol=2e-11)
    np.testing.assert_allclose(lrs[6], 1.5e-4, rtol=0.0, atol=2e-11)
