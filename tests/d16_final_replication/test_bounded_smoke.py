import torch
from d16.scripts.prepare_ofix7_mid_final_replication import load_yaml, verify_lock
from d16.scripts.validate_ofix7_mid_final_replication import DEFAULT_PRIOR_DIR, smoke


def test_bounded_smoke():
    _, paths = verify_lock()
    passed, details = smoke(load_yaml(paths["config"]), DEFAULT_PRIOR_DIR, torch.device("cpu"))
    assert passed, details
    assert details["samples_constructed"] <= 8 and details["batches_consumed"] <= 2 and details["forward_backward_steps"] == 1
