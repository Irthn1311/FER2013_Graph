from _execution_contract_evidence import contract


def test_grappler_g1_a():
    result = contract()["g1"]["configurations"]["G1-A"]
    assert result["grappler"] == {
        "arithmetic_optimization": False,
        "remapping": False,
    }
    assert result["step1"]["parameter_max_abs"] == 0.0
    assert result["step2"]["parameter_max_abs"] == 0.0
    assert result["step2"]["m1_max_abs"] <= 2e-8
    assert result["pass"] is True
