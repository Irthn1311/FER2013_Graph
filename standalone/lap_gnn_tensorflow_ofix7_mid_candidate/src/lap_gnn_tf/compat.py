"""Small compatibility helpers for the validated Keras 3 and legacy Keras 2 paths."""

from __future__ import annotations

import importlib.metadata

import tensorflow as tf


def keras_version() -> str:
    value = getattr(tf.keras, "__version__", None)
    if value:
        return str(value)
    try:
        return importlib.metadata.version("keras")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def save_model_with_optimizer(model: tf.keras.Model, path) -> None:
    if hasattr(tf.keras.optimizers.Optimizer, "_backend_apply_gradients"):
        model.save(path, include_optimizer=True)
    else:
        model.save(path)
