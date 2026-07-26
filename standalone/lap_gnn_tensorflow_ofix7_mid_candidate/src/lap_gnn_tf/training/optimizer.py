"""Locked AdamW factory with configuration-only extension points."""

from __future__ import annotations

import struct

import tensorflow as tf


_FLOAT32 = tf.float32
_ZERO = tf.constant(0.0, _FLOAT32)
_ONE = tf.constant(1.0, _FLOAT32)
_TWO = tf.constant(2.0, _FLOAT32)
_SPLITTER = tf.constant(4097.0, _FLOAT32)


def _float32_parts(value: float) -> tuple[float, float]:
    high = struct.unpack("f", struct.pack("f", float(value)))[0]
    low = struct.unpack("f", struct.pack("f", float(value) - high))[0]
    return high, low


def _two_sum(left: tf.Tensor, right: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    total = left + right
    recovered = total - left
    error = (left - (total - recovered)) + (right - recovered)
    return total, error


def _split(value: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    scaled = _SPLITTER * value
    high = scaled - (scaled - value)
    return high, value - high


def _two_product(left: tf.Tensor, right: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    product = left * right
    left_high, left_low = _split(left)
    right_high, right_low = _split(right)
    error = (
        (left_high * right_high - product)
        + left_high * right_low
        + left_low * right_high
        + left_low * right_low
    )
    return product, error


def _software_fma(
    left: tf.Tensor, right: tf.Tensor, addend: tf.Tensor
) -> tf.Tensor:
    product, product_error = _two_product(left, right)
    total, sum_error = _two_sum(product, addend)
    return total + (product_error + sum_error)


def _ds_add(
    left: tuple[tf.Tensor, tf.Tensor],
    right: tuple[tf.Tensor, tf.Tensor],
) -> tuple[tf.Tensor, tf.Tensor]:
    total, error = _two_sum(left[0], right[0])
    return _two_sum(total, error + left[1] + right[1])


def _ds_mul(
    left: tuple[tf.Tensor, tf.Tensor],
    right: tuple[tf.Tensor, tf.Tensor],
) -> tuple[tf.Tensor, tf.Tensor]:
    product, error = _two_product(left[0], right[0])
    error = (
        error
        + left[0] * right[1]
        + left[1] * right[0]
        + left[1] * right[1]
    )
    return _two_sum(product, error)


def _ds_pow(
    base: tuple[tf.Tensor, tf.Tensor], exponent: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    exponent = tf.cast(exponent, tf.int64)

    def condition(remaining, _result_high, _result_low, _factor_high, _factor_low):
        return remaining > 0

    def body(remaining, result_high, result_low, factor_high, factor_low):
        multiplied = _ds_mul(
            (result_high, result_low), (factor_high, factor_low)
        )
        odd = tf.equal(tf.bitwise.bitwise_and(remaining, 1), 1)
        result_high = tf.where(odd, multiplied[0], result_high)
        result_low = tf.where(odd, multiplied[1], result_low)
        factor_high, factor_low = _ds_mul(
            (factor_high, factor_low), (factor_high, factor_low)
        )
        return (
            tf.bitwise.right_shift(remaining, 1),
            result_high,
            result_low,
            factor_high,
            factor_low,
        )

    _, result_high, result_low, _, _ = tf.while_loop(
        condition,
        body,
        (exponent, _ONE, _ZERO, base[0], base[1]),
        parallel_iterations=1,
    )
    return result_high, result_low


def _ds_div_float(
    numerator: tuple[tf.Tensor, tf.Tensor],
    denominator: tuple[tf.Tensor, tf.Tensor],
) -> tf.Tensor:
    quotient = numerator[0] / denominator[0]
    product = _ds_mul((quotient, _ZERO), denominator)
    remainder = _ds_add(numerator, (-product[0], -product[1]))
    return quotient + (remainder[0] + remainder[1]) / denominator[0]


def _ds_sqrt_float(value: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
    root = tf.sqrt(value[0])
    square = _ds_mul((root, _ZERO), (root, _ZERO))
    remainder = _ds_add(value, (-square[0], -square[1]))
    return root + (remainder[0] + remainder[1]) / (_TWO * root)


def _torch_cpu_avx2_norm(tensor: tf.Tensor) -> tf.Tensor:
    """Match PyTorch 2.11 CPU float32 vector_norm's AVX2 lane order."""
    flat = tf.reshape(tensor, (-1,))
    size = tf.size(flat)
    vectorized_size = size - tf.math.floormod(size, 8)
    lanes = tf.reshape(flat[:vectorized_size], (-1, 8))
    lane_squares = tf.square(lanes)
    lane_totals = tf.math.cumsum(
        tf.concat([tf.zeros((1, 8), _FLOAT32), lane_squares], axis=0),
        axis=0,
    )[-1]
    total = lane_totals[0]
    for lane in range(1, 8):
        total = total + lane_totals[lane]
    tail_squares = tf.concat(
        [tf.square(flat[vectorized_size:]), tf.zeros((7,), _FLOAT32)],
        axis=0,
    )[:7]
    for tail_index in range(7):
        total = total + tail_squares[tail_index]
    return tf.sqrt(total)


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
        global_clipnorm=5.0,
        amsgrad=False,
        maximize=False,
        capturable=False,
        differentiable=False,
        name="torch_compatible_adamw",
        **kwargs,
    ):
        if clipnorm is not None:
            raise ValueError("TorchCompatibleAdamW supports global clipping only")
        unsupported = {
            "amsgrad": amsgrad,
            "maximize": maximize,
            "capturable": capturable,
            "differentiable": differentiable,
        }
        enabled = [key for key, value in unsupported.items() if value]
        if enabled:
            raise ValueError(f"Unsupported TorchCompatibleAdamW options: {enabled}")
        if global_clipnorm is None or float(global_clipnorm) != 5.0:
            raise ValueError("TorchCompatibleAdamW requires global_clipnorm=5.0")
        if not isinstance(learning_rate, (int, float)):
            raise TypeError("TorchCompatibleAdamW requires a scalar learning rate")
        super().__init__(
            learning_rate=learning_rate,
            name=name,
            global_clipnorm=global_clipnorm,
            **kwargs,
        )
        self.weight_decay_rate = float(weight_decay)
        self.beta_1 = float(beta_1)
        self.beta_2 = float(beta_2)
        self.epsilon = float(epsilon)
        self.amsgrad = False
        self.maximize = False
        self.capturable = False
        self.differentiable = False
        self._learning_rate_parts = _float32_parts(float(learning_rate))
        self._weight_decay_parts = _float32_parts(self.weight_decay_rate)
        self._beta_1_parts = _float32_parts(self.beta_1)
        self._beta_2_parts = _float32_parts(self.beta_2)
        self._one_minus_beta_1 = tf.constant(
            _float32_parts(1.0 - self.beta_1)[0], _FLOAT32
        )
        self._one_minus_beta_2 = tf.constant(
            _float32_parts(1.0 - self.beta_2)[0], _FLOAT32
        )
        self._beta_2_tensor = tf.constant(self._beta_2_parts[0], _FLOAT32)
        self._epsilon_tensor = tf.constant(
            _float32_parts(self.epsilon)[0], _FLOAT32
        )
        self._momentums = []
        self._velocities = []
        self._current_step_size = None
        self._current_bias_correction2_sqrt = None
        self._current_decay_factor = None
        self.last_global_gradient_norm = None
        self.last_clip_coefficient = None
        self._clip_variables = None

    def build(self, variables):
        for variable in variables:
            if variable.dtype != _FLOAT32:
                raise TypeError(
                    "TorchCompatibleAdamW requires float32 trainable variables; "
                    f"received {variable.dtype}"
                )
        super().build(variables)
        self._momentums = [
            self.add_variable_from_reference(variable, name="momentum")
            for variable in variables
        ]
        self._velocities = [
            self.add_variable_from_reference(variable, name="velocity")
            for variable in variables
        ]

    def _clip_gradients(self, gradients):
        present = [gradient for gradient in gradients if gradient is not None]
        if not present:
            return gradients
        if self._clip_variables is None:
            raise RuntimeError(
                "TorchCompatibleAdamW clipping requires the paired variables"
            )
        if len(gradients) != len(self._clip_variables):
            raise RuntimeError("Gradient-variable count mismatch during clipping")
        norm_gradients = []
        for gradient, variable in zip(gradients, self._clip_variables):
            if gradient is None:
                continue
            if isinstance(gradient, tf.IndexedSlices):
                raise TypeError(
                    "TorchCompatibleAdamW does not support sparse gradients"
                )
            if gradient.dtype != _FLOAT32:
                raise TypeError(
                    "TorchCompatibleAdamW requires float32 gradients; "
                    f"received {gradient.dtype}"
                )
            path = str(getattr(variable, "path", variable.name))
            leaf_name = path.rsplit("/", 1)[-1].split(":", 1)[0]
            if len(variable.shape) == 2 and leaf_name in {
                "kernel",
                "in_proj_kernel",
                "out_kernel",
            }:
                gradient = tf.transpose(gradient)
            norm_gradients.append(gradient)
        norms = [_torch_cpu_avx2_norm(gradient) for gradient in norm_gradients]
        total_norm = _torch_cpu_avx2_norm(tf.stack(norms))
        coefficient = tf.minimum(
            _ONE,
            tf.constant(5.0, _FLOAT32)
            / (total_norm + tf.constant(1e-6, _FLOAT32)),
        )
        self.last_global_gradient_norm = total_norm
        self.last_clip_coefficient = coefficient
        return [
            gradient if gradient is None else gradient * coefficient
            for gradient in gradients
        ]

    def _backend_apply_gradients(self, gradients, trainable_variables):
        self.__dict__["_clip_variables"] = list(trainable_variables)
        try:
            return super()._backend_apply_gradients(
                gradients, trainable_variables
            )
        finally:
            self.__dict__["_clip_variables"] = None

    def _exact_step_scalars(
        self, learning_rate: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        learning_rate = tf.cast(learning_rate, _FLOAT32)
        initial_lr = tf.constant(self._learning_rate_parts[0], _FLOAT32)
        lr_scale = learning_rate / initial_lr
        lr_parts = _ds_mul(
            (
                tf.constant(self._learning_rate_parts[0], _FLOAT32),
                tf.constant(self._learning_rate_parts[1], _FLOAT32),
            ),
            (lr_scale, _ZERO),
        )
        step = self.iterations + 1
        beta1_power = _ds_pow(
            (
                tf.constant(self._beta_1_parts[0], _FLOAT32),
                tf.constant(self._beta_1_parts[1], _FLOAT32),
            ),
            step,
        )
        beta2_power = _ds_pow(
            (
                tf.constant(self._beta_2_parts[0], _FLOAT32),
                tf.constant(self._beta_2_parts[1], _FLOAT32),
            ),
            step,
        )
        correction1 = _ds_add(
            (_ONE, _ZERO), (-beta1_power[0], -beta1_power[1])
        )
        correction2 = _ds_add(
            (_ONE, _ZERO), (-beta2_power[0], -beta2_power[1])
        )
        step_size = _ds_div_float(lr_parts, correction1)
        correction2_sqrt = _ds_sqrt_float(correction2)
        decay_delta = _ds_mul(
            lr_parts,
            (
                tf.constant(self._weight_decay_parts[0], _FLOAT32),
                tf.constant(self._weight_decay_parts[1], _FLOAT32),
            ),
        )
        decay = _ds_add(
            (_ONE, _ZERO), (-decay_delta[0], -decay_delta[1])
        )
        decay_factor = decay[0] + decay[1]
        return step_size, correction2_sqrt, decay_factor

    def _backend_update_step(self, gradients, trainable_variables, learning_rate):
        (
            self._current_step_size,
            self._current_bias_correction2_sqrt,
            self._current_decay_factor,
        ) = self._exact_step_scalars(learning_rate)
        return super()._backend_update_step(
            gradients, trainable_variables, learning_rate
        )

    def update_step(self, gradient, variable, learning_rate):
        if isinstance(gradient, tf.IndexedSlices):
            raise TypeError("TorchCompatibleAdamW does not support sparse gradients")
        if variable.dtype != _FLOAT32 or gradient.dtype != _FLOAT32:
            raise TypeError(
                "TorchCompatibleAdamW update requires float32 variable and gradient"
            )
        index = self._get_variable_index(variable)
        momentum = self._momentums[index]
        velocity = self._velocities[index]
        variable.assign(variable * self._current_decay_factor)
        momentum_before = tf.identity(momentum)
        momentum.assign(
            _software_fma(
                self._one_minus_beta_1,
                gradient - momentum_before,
                momentum_before,
            )
        )
        velocity.assign(velocity * self._beta_2_tensor)
        velocity.assign_add(
            (self._one_minus_beta_2 * gradient) * gradient
        )
        denominator = (
            tf.sqrt(velocity) / self._current_bias_correction2_sqrt
            + self._epsilon_tensor
        )
        variable.assign_add(
            ((-self._current_step_size) * momentum) / denominator
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "weight_decay": self.weight_decay_rate,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "epsilon": self.epsilon,
            "amsgrad": self.amsgrad,
            "maximize": self.maximize,
            "capturable": self.capturable,
            "differentiable": self.differentiable,
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
        global_clipnorm=float(optimizer_cfg.get("global_clipnorm", 5.0)),
        name="adamw",
    )
    if tf.keras.mixed_precision.global_policy().name == "mixed_float16":
        return tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    return optimizer
