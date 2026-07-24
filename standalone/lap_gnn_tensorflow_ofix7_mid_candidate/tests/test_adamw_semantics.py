import numpy as np

from _helpers import GOLDEN
from lap_gnn_tf.training.optimizer import torch_adamw_first_step_numpy


def test_adamw_semantics():
    maximum = 0.0
    with np.load(GOLDEN / "model_state.npz") as state, np.load(GOLDEN / "pytorch_gradients_eval_ce.npz") as gradients, np.load(GOLDEN / "pytorch_adamw_step1_eval_ce.npz") as updated:
        for key in state.files:
            actual = torch_adamw_first_step_numpy(state[key], gradients[key], 3e-4, 1e-3, 1e-8)
            maximum = max(maximum, float(np.max(np.abs(actual - updated[key]))))
    assert maximum <= 2e-8

