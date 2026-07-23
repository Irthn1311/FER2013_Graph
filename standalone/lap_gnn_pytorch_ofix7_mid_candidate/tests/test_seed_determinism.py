import torch

from lap_gnn.config import load_config
from lap_gnn.model.d16_model import D16Model
from lap_gnn.seed import set_seed

from _helpers import ROOT


def test_initial_state_is_seed_deterministic():
    cfg = load_config(ROOT / "configs/fer2013_ofix7_mid_seed42.yaml")
    set_seed(42)
    left = D16Model.from_config(cfg, input_dim=37)
    set_seed(42)
    right = D16Model.from_config(cfg, input_dim=37)
    assert all(torch.equal(value, right.state_dict()[key]) for key, value in left.state_dict().items())
