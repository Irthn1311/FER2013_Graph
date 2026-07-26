import tensorflow as tf

from lap_gnn_tf.training.execution import build_compiled_gradient_function


class TinyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.kernel = self.add_weight(
            shape=(2, 2), initializer="ones", dtype=tf.float32,
        )

    def call(self, batch, training=False):
        return {"logits": tf.matmul(batch["features"], self.kernel)}


def test_h1_compiled_gradients_do_not_update_state():
    model = TinyModel()
    batch = {
        "features": tf.constant([[1.0, 2.0]], tf.float32),
        "labels": tf.constant([1], tf.int64),
    }
    before = model.kernel.numpy().copy()
    compute = build_compiled_gradient_function(model, training=True)
    loss, logits, gradients, finite = compute(batch)
    assert loss.dtype == tf.float32
    assert logits.shape == (1, 2)
    assert len(gradients) == 1
    assert bool(finite.numpy())
    assert (model.kernel.numpy() == before).all()
