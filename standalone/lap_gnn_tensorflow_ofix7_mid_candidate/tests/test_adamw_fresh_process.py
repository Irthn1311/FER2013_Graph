import json
import os
import subprocess
import sys


def test_adamw_fresh_process():
    script = """
import json, numpy as np
from _adamw_reference import run_steps
v, o = run_steps([np.array([1.0, -1.0], np.float32)], [np.array([6.0, 8.0], np.float32)], steps=2)
print(json.dumps([v[0].numpy().tolist(), o._momentums[0].numpy().tolist(), o._velocities[0].numpy().tolist()]))
"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    outputs = [
        subprocess.check_output(
            [sys.executable, "-B", "-c", script],
            cwd=os.path.dirname(__file__),
            env=env,
            text=True,
        ).strip().splitlines()[-1]
        for _ in range(5)
    ]
    assert all(json.loads(value) == json.loads(outputs[0]) for value in outputs)
