from _adamw_closure_evidence import load_json


def test_live_slot_state_step1():
    step = load_json("fresh_live_adamw_comparison.json")["steps"][0]
    assert step["momentum_max_abs"] == 0.0
    assert step["velocity_max_abs"] == 0.0
