import tensorflow as tf

from lap_gnn_tf.training.execution import apply_gradients_eager_exact


def test_h1_eager_optimizer_context():
    variable = tf.Variable([1.0], dtype=tf.float32)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    apply_gradients_eager_exact(
        optimizer,
        [tf.constant([0.5], tf.float32)],
        [variable],
        expected_count=1,
    )
    assert float(variable.numpy()[0]) < 1.0
