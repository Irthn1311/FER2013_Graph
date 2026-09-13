import numpy as np
import pytest
from pixel_relational_motif_e0.selection import KResult, select_k_one_se
from pixel_relational_motif_e0.runner_contract import bind_split_path


def test_k_selection_is_deterministic_stability_then_smaller_k():
    results = [KResult(32, np.array([1.,1.,1.]), .5), KResult(64, np.array([1.,1.,1.]), .7), KResult(96, np.array([.5,.5,.5]), .9)]
    assert select_k_one_se(results) == 64


def test_runner_contract_cannot_bind_private_test():
    with pytest.raises(ValueError):
        bind_split_path("private_test", "/tmp/private.csv")
    assert str(bind_split_path("train", "/tmp/train.csv")).endswith("train.csv")
