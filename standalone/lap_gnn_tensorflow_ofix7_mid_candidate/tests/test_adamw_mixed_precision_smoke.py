import numpy as np
import tensorflow as tf

from lap_gnn_tf.training.optimizer import TorchCompatibleAdamW


def test_adamw_mixed_precision_smoke():
    previous = tf.keras.mixed_precision.global_policy()
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        variable = tf.Variable([1.0, -1.0], dtype=tf.float32)
        inner = TorchCompatibleAdamW()
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(inner)
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(tf.cast(variable, tf.float16) ** 2)
            scaled_loss = optimizer.scale_loss(loss)
        gradients = tape.gradient(scaled_loss, [variable])
        optimizer.apply_gradients(zip(gradients, [variable]))
        assert variable.dtype == tf.float32
        assert all(value.dtype == tf.float32 for value in inner._momentums)
        assert np.isfinite(variable.numpy()).all()
    finally:
        tf.keras.mixed_precision.set_global_policy(previous)
