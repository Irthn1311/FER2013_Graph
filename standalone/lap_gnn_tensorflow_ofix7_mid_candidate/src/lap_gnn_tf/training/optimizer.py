"""Locked AdamW factory with configuration-only extension points."""

from __future__ import annotations

import tensorflow as tf


def torch_adamw_first_step_numpy(parameter, gradient, lr, weight_decay, epsilon):
    """Reference first-step formula used by the bounded semantics test."""
    import numpy as np

    parameter = np.asarray(parameter, dtype=np.float32)
    gradient = np.asarray(gradient, dtype=np.float32)
    decayed = parameter * np.float32(1.0 - float(lr) * float(weight_decay))
    exp_avg = np.float32(0.1) * gradient
    exp_avg_sq = np.float32(0.001) * np.square(gradient)
    denominator = np.sqrt(exp_avg_sq) / np.float32((1.0 - 0.999) ** 0.5) + np.float32(epsilon)
    step_size = np.float32(float(lr) / (1.0 - 0.9))
    return decayed - step_size * exp_avg / denominator


@tf.keras.utils.register_keras_serializable(package="lap_gnn_tf")
class TorchCompatibleAdamW(tf.keras.optimizers.Optimizer):
    """Keras optimizer with PyTorch AdamW epsilon and bias-correction order."""

    def __init__(
        self,
        learning_rate=3e-4,
        weight_decay=1e-3,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        clipnorm=None,
        name="torch_compatible_adamw",
        **kwargs,
    ):
        super().__init__(learning_rate=learning_rate, name=name, clipnorm=clipnorm, **kwargs)
        self.weight_decay_rate = float(weight_decay)
        self.beta_1 = float(beta_1)
        self.beta_2 = float(beta_2)
        self.epsilon = float(epsilon)
        self._momentums = []
        self._velocities = []

    def build(self, variables):
        super().build(variables)
        self._momentums = [
            self.add_variable_from_reference(variable, name="momentum")
            for variable in variables
        ]
        self._velocities = [
            self.add_variable_from_reference(variable, name="velocity")
            for variable in variables
        ]

    def update_step(self, gradient, variable, learning_rate):
        if isinstance(gradient, tf.IndexedSlices):
            gradient = tf.convert_to_tensor(gradient)
        learning_rate = tf.cast(learning_rate, variable.dtype)
        gradient = tf.cast(gradient, variable.dtype)
        beta1 = tf.cast(self.beta_1, variable.dtype)
        beta2 = tf.cast(self.beta_2, variable.dtype)
        epsilon = tf.cast(self.epsilon, variable.dtype)
        weight_decay = tf.cast(self.weight_decay_rate, variable.dtype)
        index = self._get_variable_index(variable)
        momentum = self._momentums[index]
        velocity = self._velocities[index]
        variable.assign(variable * (tf.cast(1.0, variable.dtype) - learning_rate * weight_decay))
        momentum.assign(beta1 * momentum + (tf.cast(1.0, variable.dtype) - beta1) * gradient)
        velocity.assign(beta2 * velocity + (tf.cast(1.0, variable.dtype) - beta2) * tf.square(gradient))
        step = tf.cast(self.iterations + 1, variable.dtype)
        bias_correction1 = tf.cast(1.0, variable.dtype) - tf.pow(beta1, step)
        bias_correction2 = tf.cast(1.0, variable.dtype) - tf.pow(beta2, step)
        denominator = tf.sqrt(velocity) / tf.sqrt(bias_correction2) + epsilon
        variable.assign_sub((learning_rate / bias_correction1) * momentum / denominator)

    def get_config(self):
        config = super().get_config()
        config.update({
            "weight_decay": self.weight_decay_rate,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "epsilon": self.epsilon,
        })
        return config


def build_optimizer(config: dict) -> tf.keras.optimizers.Optimizer:
    training = config["training"]
    optimizer_cfg = training.get("optimizer", {})
    optimizer_type = str(optimizer_cfg.get("type", "adamw")).lower()
    if optimizer_type != "adamw":
        raise ValueError(f"Unsupported executable TensorFlow optimizer: {optimizer_type}")
    implementation = str(optimizer_cfg.get("implementation", "torch_compatible")).lower()
    optimizer_class = TorchCompatibleAdamW if implementation == "torch_compatible" else tf.keras.optimizers.AdamW
    optimizer = optimizer_class(
        learning_rate=float(training["lr"]),
        weight_decay=float(training["weight_decay"]),
        beta_1=float(optimizer_cfg.get("beta1", 0.9)),
        beta_2=float(optimizer_cfg.get("beta2", 0.999)),
        epsilon=float(optimizer_cfg.get("epsilon", 1e-8)),
        clipnorm=optimizer_cfg.get("clipnorm"),
        name="adamw",
    )
    if tf.keras.mixed_precision.global_policy().name == "mixed_float16":
        return tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    return optimizer
