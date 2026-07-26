from _execution_contract_evidence import contract


def test_h1_clipped_gradient_parity():
    h1 = contract()["h1"]
    assert h1["clipped_gradient_max_abs"] == 1.0356307029724121e-06
    assert h1["clipped_gradient_relative_l2"] < 2e-6
    assert h1["clipped_gradient_minimum_cosine"] > 0.999999
