from _execution_contract_evidence import contract


def test_h1_live_step1_is_retained_as_rejected_evidence():
    step = contract()["h1"]["step1"]
    assert step["parameter_max_abs"] == 2.4257227778434753e-05
    assert step["m1_max_abs"] == 1.0337680578231812e-07
    assert step["m2_max_abs"] == 1.5133991837501526e-09
    assert step["pass"] is False
