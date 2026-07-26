from _execution_contract_evidence import contract


def test_h1_live_step2_is_retained_as_rejected_evidence():
    step = contract()["h1"]["step2"]
    assert step["parameter_max_abs"] == 4.851445555686951e-05
    assert step["m1_max_abs"] == 1.9744038581848145e-07
    assert step["m2_max_abs"] == 3.259629011154175e-09
    assert step["pass"] is False
