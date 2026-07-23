from pathlib import Path

import numpy as np
import torch

from lap_gnn.config import load_config
from lap_gnn.model.d16_model import D16Model
from lap_gnn.validation import load_golden_batch, load_portable_model_state


ROOT = Path(__file__).resolve().parents[1]


def loaded_model(eval_mode=True):
    cfg = load_config(ROOT / "configs/fer2013_ofix7_mid_seed42.yaml")
    model = D16Model.from_config(cfg, input_dim=37)
    model.load_state_dict(load_portable_model_state(ROOT), strict=True)
    model.eval() if eval_mode else model.train()
    return model


def golden_batch():
    return load_golden_batch(ROOT)


def golden_array(name):
    return np.load(ROOT / "validation_assets/golden" / name, allow_pickle=False)
