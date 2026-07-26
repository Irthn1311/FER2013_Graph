from _adamw_closure_evidence import primitive


def test_adamw_denominator_primitive():
    row = primitive(2, "denominator", "sqrt_div_add")
    assert float(row["max_abs"]) == 0.0
    assert int(row["array_exact_tensors"]) == 127
