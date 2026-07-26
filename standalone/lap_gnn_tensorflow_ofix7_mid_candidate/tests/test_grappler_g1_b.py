from _execution_contract_evidence import contract


def test_grappler_g1_b():
    result = contract()["g1"]["configurations"]["G1-B"]
    assert result["grappler"] == {
        "arithmetic_optimization": False,
        "remapping": False,
        "function_optimization": False,
        "dependency_optimization": False,
    }
    assert result["step2"]["parameter_max_abs"] <= 2e-8
    assert result["step2"]["m1_max_abs"] <= 2e-8
    assert result["step2"]["m2_max_abs"] <= 2e-8
    assert result["pass"] is True
