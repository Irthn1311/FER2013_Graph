import numpy as np

from _adamw_reference import run_steps


def test_adamw_iteration_counter():
    _, optimizer = run_steps(
        [np.array([1.0], np.float32)],
        [np.array([0.25], np.float32)],
        steps=2,
    )
    assert int(optimizer.iterations.numpy()) == 2
