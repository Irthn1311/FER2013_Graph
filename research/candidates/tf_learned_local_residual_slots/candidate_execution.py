"""Selected G1-A execution extension for the 128-variable Step-10 candidate."""

from __future__ import annotations

import tensorflow as tf

from lap_gnn_tf.training.execution import validate_gradient_contract
from lap_gnn_tf.training.losses import sparse_cross_entropy


EXPECTED_CANDIDATE_TRAINABLE_VARIABLE_COUNT = 128


def build_candidate_restricted_graph_train_step(
    model,
    optimizer,
    input_signature: dict[str, tf.TensorSpec] | None = None,
):
    """Build the selected G1-A step with the amended 128-variable contract."""

    signature = None if input_signature is None else [input_signature]

    @tf.function(
        autograph=False,
        jit_compile=False,
        input_signature=signature,
        reduce_retracing=True,
    )
    def train_step(batch):
        with tf.GradientTape() as tape:
            output = model(batch, training=True)
            loss = sparse_cross_entropy(batch["labels"], output["logits"])
            if hasattr(optimizer, "scale_loss"):
                scaled_loss = optimizer.scale_loss(loss)
            elif hasattr(optimizer, "get_scaled_loss"):
                scaled_loss = optimizer.get_scaled_loss(loss)
            else:
                scaled_loss = loss
        gradients = tape.gradient(scaled_loss, model.trainable_variables)
        if (
            not hasattr(optimizer, "scale_loss")
            and hasattr(optimizer, "get_unscaled_gradients")
        ):
            gradients = optimizer.get_unscaled_gradients(gradients)
        validate_gradient_contract(
            gradients,
            model.trainable_variables,
            expected_count=EXPECTED_CANDIDATE_TRAINABLE_VARIABLE_COUNT,
        )
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    return train_step
