"""Locked CE-only objective."""

import tensorflow as tf


def sparse_cross_entropy(labels, logits):
    logits = tf.cast(logits, tf.float32)
    losses = tf.keras.losses.sparse_categorical_crossentropy(
        labels, logits, from_logits=True,
    )
    return tf.reduce_mean(losses)
