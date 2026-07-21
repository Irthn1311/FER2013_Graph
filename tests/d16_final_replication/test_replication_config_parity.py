from d16.scripts.prepare_ofix7_mid_final_replication import CONFIG_DIR, REGISTERED_SEEDS, validate_replication_config


def test_replication_config_parity():
    for seed in REGISTERED_SEEDS:
        result = validate_replication_config(CONFIG_DIR / f"ofix7_mid_seed{seed}.yaml")
        assert result["valid"]
        assert not result["unauthorized"]
