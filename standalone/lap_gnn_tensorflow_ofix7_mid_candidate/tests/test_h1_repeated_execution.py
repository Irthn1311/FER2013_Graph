from _execution_contract_evidence import contract


def test_h1_repeated_execution_is_not_promoted():
    h1 = contract()["h1"]
    assert h1["repeated_pass"] is False
    assert h1["pass"] is False
