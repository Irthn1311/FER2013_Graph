import numpy as np
import tensorflow as tf

from _helpers import GOLDEN, loaded
from lap_gnn_tf.training.losses import sparse_cross_entropy


def test_gradient_parity():
    model, batch = loaded()
    with tf.GradientTape() as tape:
        loss = sparse_cross_entropy(batch["labels"], model(batch, training=False)["logits"])
    gradients = tape.gradient(loss, model.trainable_variables)
    by_id = {id(variable): gradient for variable, gradient in zip(model.trainable_variables, gradients)}
    actual, expected = [], []
    with np.load(GOLDEN / "pytorch_gradients_eval_ce.npz", allow_pickle=False) as reference:
        for binding in model.state_bindings():
            gradient = by_id[id(binding.variable)].numpy()
            if binding.transform == "transpose":
                gradient = gradient.T
            actual.append(gradient.reshape(-1))
            expected.append(reference[binding.source_key].reshape(-1))
    actual = np.concatenate(actual).astype(np.float64)
    expected = np.concatenate(expected).astype(np.float64)
    cosine = np.dot(actual, expected) / (np.linalg.norm(actual) * np.linalg.norm(expected))
    assert cosine >= 0.99999

