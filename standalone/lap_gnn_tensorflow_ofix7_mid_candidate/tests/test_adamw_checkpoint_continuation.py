import numpy as np
import tensorflow as tf

from _adamw_reference import make_optimizer
from lap_gnn_tf.compat import save_model_with_optimizer


def _model():
    inputs = tf.keras.Input((2,))
    outputs = tf.keras.layers.Dense(1, use_bias=False)(inputs)
    return tf.keras.Model(inputs, outputs)


def test_adamw_checkpoint_continuation(tmp_path):
    model = _model()
    model.layers[-1].kernel.assign([[1.0], [-1.0]])
    optimizer = make_optimizer()
    model.compile(optimizer=optimizer, loss="mse", run_eagerly=True)
    gradient = tf.constant([[0.25], [-0.5]], tf.float32)
    optimizer.apply_gradients([(gradient, model.layers[-1].kernel)])
    path = tmp_path / "adamw_continuation.keras"
    save_model_with_optimizer(model, path)
    restored = tf.keras.models.load_model(path)
    optimizer.apply_gradients([(gradient, model.layers[-1].kernel)])
    restored.optimizer.apply_gradients(
        [(gradient, restored.layers[-1].kernel)]
    )
    np.testing.assert_array_equal(
        model.layers[-1].kernel.numpy(),
        restored.layers[-1].kernel.numpy(),
    )
    for left, right in zip(optimizer.variables, restored.optimizer.variables):
        np.testing.assert_array_equal(left.numpy(), right.numpy())
