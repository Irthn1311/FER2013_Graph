import warnings

import numpy as np
import tensorflow as tf

from _helpers import loaded
from lap_gnn_tf.compat import save_model_with_optimizer
from lap_gnn_tf.training.optimizer import TorchCompatibleAdamW


def test_keras_clean_roundtrip(tmp_path):
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model, batch = loaded()
        expected = model(batch, training=False)["logits"].numpy()
        optimizer = TorchCompatibleAdamW()
        optimizer.build(model.trainable_variables)
        model.compile(optimizer=optimizer, run_eagerly=True)
        path = tmp_path / "clean.keras"
        save_model_with_optimizer(model, path)
        restored = tf.keras.models.load_model(path)
        actual = restored(batch, training=False)["logits"].numpy()
    messages = [str(item.message) for item in captured]
    assert not any("does not have a `build()` method" in message for message in messages)
    assert np.array_equal(actual, expected)
    assert restored.get_config() == model.get_config()
    assert len(restored.trainable_variables) == 127
    assert len(restored.optimizer.variables) == len(optimizer.variables)
