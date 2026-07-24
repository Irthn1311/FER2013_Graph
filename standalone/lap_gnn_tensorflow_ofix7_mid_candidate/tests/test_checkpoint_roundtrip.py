import numpy as np
import tensorflow as tf
from pathlib import Path

from _helpers import loaded
from lap_gnn_tf.config import load_config
from lap_gnn_tf.training.optimizer import build_optimizer


def test_checkpoint_roundtrip(tmp_path):
    model, batch = loaded()
    expected = model(batch, training=False)["logits"].numpy()
    path = tmp_path / "roundtrip.weights.h5"
    model.save_weights(path)
    restored, _ = loaded()
    restored.load_weights(path)
    actual = restored(batch, training=False)["logits"].numpy()
    assert np.array_equal(actual, expected)


def test_full_keras_checkpoint_roundtrip(tmp_path):
    model, batch = loaded()
    expected = model(batch, training=False)["logits"].numpy()
    config_path = Path(__file__).resolve().parents[1] / "configs" / "fer2013_ofix7_mid_tensorflow_seed42.yaml"
    optimizer = build_optimizer(load_config(config_path))
    optimizer.build(model.trainable_variables)
    model.compile(optimizer=optimizer, run_eagerly=True)
    path = tmp_path / "roundtrip.keras"
    model.save(path, include_optimizer=True)
    restored = tf.keras.models.load_model(path)
    actual = restored(batch, training=False)["logits"].numpy()
    assert np.array_equal(actual, expected)
    assert type(restored.optimizer).__name__ == type(optimizer).__name__
    assert len(restored.optimizer.variables) == len(optimizer.variables)
