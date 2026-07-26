from _adamw_closure_evidence import primitive


def test_adamw_weight_decay_primitive():
    row = primitive(2, "weight_decay_mul", "mul_expression")
    assert float(row["max_abs"]) == 0.0
    assert int(row["array_exact_tensors"]) == 127
