from d16.scripts.prepare_ofix7_mid_final_replication import REGISTERED_SEEDS, make_replication_config, verify_lock, load_yaml


def test_seed_propagation():
    _, paths = verify_lock(); source = load_yaml(paths["config"])
    for seed in REGISTERED_SEEDS:
        cfg = make_replication_config(source, seed)
        assert cfg["seed"] == cfg["training"]["seed"] == seed
        assert cfg["graph"]["prior_corruption"]["seed"] == seed + 7699
