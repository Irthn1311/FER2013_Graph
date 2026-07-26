from _execution_contract_evidence import contract


def test_g1_repeated_fresh_process():
    g1 = contract()["g1"]
    assert g1["decision"] == "G1_PASS"
    for result in g1["configurations"].values():
        assert result["repetitions"] == 15
        assert result["iterations_exact"] is True
        assert result["variables_accounted_for"] == 127
        assert result["pass"] is True
