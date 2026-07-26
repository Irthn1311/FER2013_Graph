import numpy as np

from _adamw_reference import reference_step, run_steps


def test_adamw_weight_decay_order():
    parameters = [np.array([1.0], np.float32)]
    gradients = [np.array([0.25], np.float32)]
    expected = reference_step(
        parameters, gradients, [np.zeros(1, np.float32)], [np.zeros(1, np.float32)], 1
    )[0][0]
    variables, _ = run_steps(parameters, gradients)
    np.testing.assert_array_equal(variables[0].numpy(), expected)
