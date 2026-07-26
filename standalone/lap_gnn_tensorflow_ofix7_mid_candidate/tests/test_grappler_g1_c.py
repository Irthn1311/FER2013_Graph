from _execution_contract_evidence import contract


def test_grappler_g1_c():
    result = contract()["g1"]["configurations"]["G1-C"]
    assert result["grappler"] == {"disable_meta_optimizer": True}
    assert result["step2"]["parameter_max_abs"] <= 2e-8
    assert result["step2"]["m1_max_abs"] <= 2e-8
    assert result["step2"]["m2_max_abs"] <= 2e-8
    assert result["pass"] is True
