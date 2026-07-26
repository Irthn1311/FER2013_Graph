import numpy as np

from _adamw_reference import assert_optimizer_state, reference_step, run_steps


def test_adamw_vector():
    parameters = [np.array([1.0, -2.0, 0.125, 8.0], np.float32)]
    gradients = [np.array([0.25, -0.5, 0.0, 1.25], np.float32)]
    expected = reference_step(
        parameters, gradients, [np.zeros(4, np.float32)], [np.zeros(4, np.float32)], 1
    )
    assert_optimizer_state(*run_steps(parameters, gradients), expected)
