from _adamw_closure_evidence import load_json


def test_live_recertification_step1():
    step = load_json("fresh_live_adamw_comparison.json")["steps"][0]
    assert step["pass_2e_8"]
    assert step["parameter_max_abs"] == 0.0
