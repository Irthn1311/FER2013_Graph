from _adamw_closure_evidence import GATE, load_json


def test_live_recertification_step2():
    step = load_json("fresh_live_adamw_comparison.json")["steps"][1]
    assert step["pass_2e_8"]
    assert step["parameter_max_abs"] <= GATE
    assert step["momentum_max_abs"] <= GATE
