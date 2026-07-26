from _execution_contract_evidence import contract


def test_h1_mixed_precision_order_is_not_registered_after_h1_gate_failure():
    evidence = contract()
    assert evidence["h1"]["pass"] is False
    assert evidence["selected_execution"]["strategy"] != (
        "SELECT_H1_COMPILED_GRADIENTS_EAGER_OPTIMIZER"
    )
    assert evidence["mixed_precision_validation"]["pass"] is True
