from _adamw_closure_evidence import GATE, load_json


def test_offline_real_model_step1_closure():
    step = load_json("production_optimizer_offline_closure.json")["steps"][0]
    assert step["pass"]
    assert step["parameter"]["max_abs"] <= GATE
    assert step["momentum"]["max_abs"] <= GATE
    assert step["velocity"]["max_abs"] <= GATE
