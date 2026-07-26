import pytest
import tensorflow as tf

from _adamw_reference import make_optimizer


def test_adamw_rejects_non_float32_variables():
    optimizer = make_optimizer()
    with pytest.raises(TypeError, match="float32"):
        optimizer.build([tf.Variable([1.0], dtype=tf.float16)])


def test_adamw_rejects_sparse_gradients():
    variable = tf.Variable([1.0, 2.0], dtype=tf.float32)
    optimizer = make_optimizer()
    gradient = tf.IndexedSlices(
        values=tf.constant([1.0], tf.float32),
        indices=tf.constant([0]),
        dense_shape=tf.constant([2]),
    )
    with pytest.raises(TypeError, match="sparse"):
        optimizer.apply_gradients([(gradient, variable)])
