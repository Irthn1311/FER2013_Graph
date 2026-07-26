from _execution_contract_evidence import contract


def test_h1_raw_gradient_order():
    evidence = contract()
    assert evidence["g1"]["configurations"]["G1-A"][
        "variables_accounted_for"
    ] == 127
    assert "127 ordered TensorFlow gradients" in evidence["h1"]["design"]
