import numpy as np
import tensorflow as tf

from _helpers import GOLDEN, golden, loaded


def test_classifier_isolation():
    model, _ = loaded()
    with np.load(GOLDEN / "layer_outputs.npz", allow_pickle=False) as layers:
        classifier_input = tf.convert_to_tensor(layers["classifier_input"])
    actual = model.classifier(classifier_input, training=False).numpy()
    expected = golden("logits.npy")
    assert np.max(np.abs(actual - expected)) <= 1e-6
