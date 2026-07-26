from _adamw_closure_evidence import primitive


def test_adamw_addcmul_primitive():
    row = primitive(2, "velocity_addcmul", "scale_then_product")
    assert float(row["max_abs"]) == 0.0
    assert int(row["array_exact_tensors"]) == 127
