"""Registered TensorFlow execution modes for the locked OFIX7-mid trainer."""

from __future__ import annotations

from collections.abc import Sequence

import tensorflow as tf

from lap_gnn_tf.training.losses import sparse_cross_entropy


EXPECTED_TRAINABLE_VARIABLE_COUNT = 127
MAX_REGISTERED_TRAIN_STEP_TRACES = 1
EXECUTION_CONTRACT_SHA256 = (
    "14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22"
)
G1_A_OPTIONS = {
    "arithmetic_optimization": False,
    "remapping": False,
}
REGISTERED_OPTIMIZER_EXECUTION_MODES = {
    "restricted_tf_function",
    "eager_exact",
}
REGISTERED_GRADIENT_EXECUTION_MODES = {"tf_function"}


def configure_restricted_grappler() -> dict:
    """Register G1-A before tracing the optimizer function."""
    tf.config.optimizer.set_experimental_options(G1_A_OPTIONS)
    return tf.config.optimizer.get_experimental_options()


def validate_execution_config(training: dict) -> dict:
    optimizer_mode = str(training.get("optimizer_execution_mode", ""))
    gradient_mode = str(training.get("gradient_execution_mode", ""))
    if optimizer_mode not in REGISTERED_OPTIMIZER_EXECUTION_MODES:
        raise ValueError(
            f"Unregistered optimizer_execution_mode: {optimizer_mode!r}"
        )
    if gradient_mode not in REGISTERED_GRADIENT_EXECUTION_MODES:
        raise ValueError(
            f"Unregistered gradient_execution_mode: {gradient_mode!r}"
        )
    if optimizer_mode == "restricted_tf_function":
        if training.get("grappler_profile") != "G1-A":
            raise ValueError(
                "restricted_tf_function requires registered grappler_profile G1-A"
            )
        effective = configure_restricted_grappler()
    else:
        effective = tf.config.optimizer.get_experimental_options()
    return {
        "optimizer_execution_mode": optimizer_mode,
        "gradient_execution_mode": gradient_mode,
        "grappler_profile": (
            "G1-A" if optimizer_mode == "restricted_tf_function" else None
        ),
        "effective_grappler_options": effective,
    }


def validate_gradient_contract(
    gradients: Sequence[tf.Tensor | None],
    variables: Sequence[tf.Variable],
    expected_count: int = EXPECTED_TRAINABLE_VARIABLE_COUNT,
) -> None:
    if len(variables) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} trainable variables, got {len(variables)}"
        )
    if len(gradients) != len(variables):
        raise RuntimeError(
            f"Gradient-variable order mismatch: {len(gradients)} != {len(variables)}"
        )
    for index, (gradient, variable) in enumerate(zip(gradients, variables)):
        if gradient is None:
            raise RuntimeError(f"Missing gradient at ordered index {index}")
        if isinstance(gradient, tf.IndexedSlices):
            raise TypeError(
                f"Sparse gradient is unsupported at ordered index {index}"
            )
        if gradient.dtype != tf.float32:
            raise TypeError(
                f"Gradient {index} must be float32, got {gradient.dtype}"
            )
        if variable.dtype != tf.float32:
            raise TypeError(
                f"Variable {index} must be float32, got {variable.dtype}"
            )
        if gradient.shape != variable.shape:
            raise RuntimeError(
                f"Gradient shape mismatch at {index}: "
                f"{gradient.shape} != {variable.shape}"
            )


def apply_gradients_eager_exact(
    optimizer,
    gradients: Sequence[tf.Tensor | None],
    variables: Sequence[tf.Variable],
    expected_count: int = EXPECTED_TRAINABLE_VARIABLE_COUNT,
) -> None:
    """Apply H1 gradients while failing closed outside eager execution."""
    if not tf.executing_eagerly():
        raise RuntimeError(
            "eager_exact optimizer application is forbidden in graph context"
        )
    validate_gradient_contract(gradients, variables, expected_count)
    optimizer.apply_gradients(zip(gradients, variables))


def build_compiled_gradient_function(model, training: bool):
    """Build H1 compute-only forward/loss/backward without state updates."""

    @tf.function(autograph=False, jit_compile=False)
    def compute_loss_and_gradients(batch):
        with tf.GradientTape() as tape:
            output = model(batch, training=training)
            logits = output["logits"]
            loss = sparse_cross_entropy(batch["labels"], logits)
        gradients = tape.gradient(loss, model.trainable_variables)
        finite = tf.reduce_all(
            tf.stack([
                tf.reduce_all(tf.math.is_finite(gradient))
                for gradient in gradients
            ])
        )
        return loss, logits, tuple(gradients), finite

    return compute_loss_and_gradients


def build_restricted_graph_train_step(
    model,
    optimizer,
    input_signature: dict[str, tf.TensorSpec] | None = None,
):
    """Build the selected G1-A full compute/update training function."""

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
        validate_gradient_contract(gradients, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    return train_step
