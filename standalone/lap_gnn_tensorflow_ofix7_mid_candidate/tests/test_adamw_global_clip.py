import numpy as np

from _adamw_reference import assert_optimizer_state, reference_step, run_steps


def test_adamw_global_clip():
    parameters = [np.array([1.0, -1.0], np.float32)]
    gradients = [np.array([6.0, 8.0], np.float32)]
    expected = reference_step(
        parameters, gradients, [np.zeros(2, np.float32)], [np.zeros(2, np.float32)], 1
    )
    variables, optimizer = run_steps(parameters, gradients)
    assert float(optimizer.last_global_gradient_norm.numpy()) == 10.0
    assert_optimizer_state(variables, optimizer, expected)
