import pytest
from d16.scripts.prepare_ofix7_mid_final_replication import load_yaml, make_replication_config, verify_lock


def test_runner_refuses_unknown_seed():
    _, paths = verify_lock()
    with pytest.raises(ValueError): make_replication_config(load_yaml(paths["config"]), 999)
