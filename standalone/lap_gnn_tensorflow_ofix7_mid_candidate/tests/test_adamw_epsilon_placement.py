import numpy as np

from _adamw_reference import assert_optimizer_state, reference_step, run_steps


def test_adamw_epsilon_placement():
    parameters = [np.array([1.0, -1.0], np.float32)]
    gradients = [np.array([1e-10, -1e-12], np.float32)]
    expected = reference_step(
        parameters, gradients, [np.zeros(2, np.float32)], [np.zeros(2, np.float32)], 1
    )
    assert_optimizer_state(*run_steps(parameters, gradients), expected)
