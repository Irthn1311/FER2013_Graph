from lap_gnn.config import load_config, validate_locked_config

from _helpers import ROOT


def test_all_seed_configs_are_locked():
    for seed in (42, 1009, 1337, 777, 3407):
        cfg = load_config(ROOT / f"configs/fer2013_ofix7_mid_seed{seed}.yaml")
        assert not validate_locked_config(cfg)
        assert cfg["graph"]["prior_corruption"]["seed"] == seed + 7699
