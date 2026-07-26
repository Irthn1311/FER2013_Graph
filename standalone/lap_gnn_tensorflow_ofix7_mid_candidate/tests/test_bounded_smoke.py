import tensorflow as tf

from _helpers import loaded
from lap_gnn_tf.training.losses import sparse_cross_entropy


def test_bounded_smoke():
    model, batch = loaded()
    output = model(batch, training=False)
    loss = sparse_cross_entropy(batch["labels"], output["logits"])
    assert output["logits"].shape == (8, 7)
    assert bool(tf.math.is_finite(loss))

