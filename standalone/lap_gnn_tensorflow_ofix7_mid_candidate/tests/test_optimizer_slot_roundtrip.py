import numpy as np
import tensorflow as tf

from _helpers import loaded
from lap_gnn_tf.training.optimizer import TorchCompatibleAdamW


def test_optimizer_slot_roundtrip(tmp_path):
    model, batch = loaded()
    optimizer = TorchCompatibleAdamW()
    optimizer.build(model.trainable_variables)
    model.compile(optimizer=optimizer, run_eagerly=True)
    for index, variable in enumerate(optimizer.variables):
        if variable.dtype == tf.int64:
            variable.assign(index)
        else:
            variable.assign(tf.fill(variable.shape, tf.cast(index + 1, variable.dtype) * 1e-7))
    expected = [value.numpy().copy() for value in optimizer.variables]
    path = tmp_path / "optimizer_slots.keras"
    model.save(path, include_optimizer=True)
    restored = tf.keras.models.load_model(path)
    actual = [value.numpy() for value in restored.optimizer.variables]
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        assert np.array_equal(left, right)
    assert np.isfinite(restored(batch, training=False)["logits"].numpy()).all()
