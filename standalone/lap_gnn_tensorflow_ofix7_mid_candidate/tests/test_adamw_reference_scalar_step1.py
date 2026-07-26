import numpy as np

from _adamw_reference import assert_optimizer_state, reference_step, run_steps


def test_adamw_reference_scalar_step1():
    parameters = [np.array(1.25, np.float32)]
    gradients = [np.array(0.25, np.float32)]
    expected = reference_step(
        parameters, gradients, [np.array(0.0, np.float32)], [np.array(0.0, np.float32)], 1
    )
    variables, optimizer = run_steps(parameters, gradients)
    assert_optimizer_state(variables, optimizer, expected)
