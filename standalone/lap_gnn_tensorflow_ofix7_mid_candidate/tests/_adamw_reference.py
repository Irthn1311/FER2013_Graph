import math

import numpy as np
import tensorflow as tf

from lap_gnn_tf.training.optimizer import TorchCompatibleAdamW


LR = 3e-4
WEIGHT_DECAY = 1e-3
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8


def lane8_norm(values):
    flat = np.asarray(values, np.float32).reshape(-1)
    vectorized_size = (flat.size // 8) * 8
    lanes = np.zeros(8, np.float32)
    for offset in range(0, vectorized_size, 8):
        chunk = flat[offset : offset + 8]
        lanes = np.float32(lanes + np.float32(chunk * chunk))
    total = lanes[0]
    for lane in range(1, 8):
        total = np.float32(total + lanes[lane])
    for value in flat[vectorized_size:]:
        total = np.float32(total + np.float32(value * value))
    return np.float32(np.sqrt(total))


def clipped_gradients(gradients):
    norms = np.array([lane8_norm(gradient) for gradient in gradients], np.float32)
    total_norm = lane8_norm(norms)
    coefficient = np.minimum(
        np.float32(1.0),
        np.float32(5.0)
        / np.float32(total_norm + np.float32(1e-6)),
    )
    return [np.float32(gradient * coefficient) for gradient in gradients]


def reference_step(parameters, gradients, momentums, velocities, step):
    gradients = clipped_gradients(gradients)
    outputs = []
    for parameter, gradient, momentum, velocity in zip(
        parameters, gradients, momentums, velocities
    ):
        parameter = np.float32(
            parameter * np.float32(1.0 - LR * WEIGHT_DECAY)
        )
        momentum = np.float32(
            np.float64(np.float32(1.0 - BETA1))
            * (gradient - momentum).astype(np.float64)
            + momentum.astype(np.float64)
        )
        velocity = np.float32(velocity * np.float32(BETA2))
        velocity = np.float32(
            velocity
            + np.float32(np.float32(1.0 - BETA2) * gradient) * gradient
        )
        denominator = np.float32(
            np.sqrt(velocity)
            / np.float32(math.sqrt(1.0 - BETA2**step))
            + np.float32(EPSILON)
        )
        parameter = np.float32(
            parameter
            + np.float32(
                np.float32(-LR / (1.0 - BETA1**step)) * momentum
            )
            / denominator
        )
        outputs.append((parameter, momentum, velocity))
    return outputs


def make_optimizer():
    return TorchCompatibleAdamW(
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        beta_1=BETA1,
        beta_2=BETA2,
        epsilon=EPSILON,
        global_clipnorm=5.0,
    )


def run_steps(parameters, gradients, steps=1, graph=False):
    variables = [tf.Variable(value, dtype=tf.float32) for value in parameters]
    tensors = [tf.constant(value, tf.float32) for value in gradients]
    optimizer = make_optimizer()

    def apply():
        optimizer.apply_gradients(zip(tensors, variables))

    apply_fn = tf.function(apply) if graph else apply
    for _ in range(steps):
        apply_fn()
    return variables, optimizer


def assert_optimizer_state(
    variables, optimizer, expected, atol=2e-8
):
    for index, (parameter, momentum, velocity) in enumerate(expected):
        np.testing.assert_allclose(
            variables[index].numpy(), parameter, rtol=0, atol=atol
        )
        np.testing.assert_allclose(
            optimizer._momentums[index].numpy(), momentum, rtol=0, atol=atol
        )
        np.testing.assert_allclose(
            optimizer._velocities[index].numpy(), velocity, rtol=0, atol=atol
        )
