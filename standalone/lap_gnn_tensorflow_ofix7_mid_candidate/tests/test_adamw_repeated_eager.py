import numpy as np

from _adamw_reference import run_steps


def test_adamw_repeated_eager():
    outputs = []
    for _ in range(10):
        variables, optimizer = run_steps(
            [np.array([1.0, -1.0], np.float32)],
            [np.array([6.0, 8.0], np.float32)],
            steps=2,
        )
        outputs.append(
            (
                variables[0].numpy(),
                optimizer._momentums[0].numpy(),
                optimizer._velocities[0].numpy(),
            )
        )
    for actual in outputs[1:]:
        for left, right in zip(outputs[0], actual):
            np.testing.assert_array_equal(left, right)
