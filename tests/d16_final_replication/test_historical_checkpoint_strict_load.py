import torch
from d16.models.d16_model import D16Model
from d16.scripts.prepare_ofix7_mid_final_replication import load_yaml, verify_lock


def test_historical_checkpoint_strict_load():
    lock, paths = verify_lock(); cfg = load_yaml(paths["config"])
    state = torch.load(paths["best"], map_location="cpu", weights_only=False)["model_state_dict"]
    model = D16Model.from_config(cfg, input_dim=37)
    model.load_state_dict(state, strict=True)
    assert sum(p.numel() for p in model.parameters()) == lock["parameter_count"]
