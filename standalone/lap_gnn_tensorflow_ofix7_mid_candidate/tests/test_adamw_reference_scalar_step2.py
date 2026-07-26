import numpy as np

from _adamw_reference import assert_optimizer_state, reference_step, run_steps


def test_adamw_reference_scalar_step2():
    parameters = [np.array(1.25, np.float32)]
    gradients = [np.array(0.25, np.float32)]
    first = reference_step(
        parameters, gradients, [np.array(0.0, np.float32)], [np.array(0.0, np.float32)], 1
    )
    expected = reference_step(
        [first[0][0]], gradients, [first[0][1]], [first[0][2]], 2
    )
    variables, optimizer = run_steps(parameters, gradients, steps=2)
    assert_optimizer_state(variables, optimizer, expected)
