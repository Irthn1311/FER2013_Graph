from _adamw_closure_evidence import primitive


def test_adamw_addcdiv_primitive():
    row = primitive(
        2, "parameter_addcdiv", "scale_numerator_then_div_add"
    )
    assert float(row["max_abs"]) == 0.0
    assert int(row["array_exact_tensors"]) == 127
