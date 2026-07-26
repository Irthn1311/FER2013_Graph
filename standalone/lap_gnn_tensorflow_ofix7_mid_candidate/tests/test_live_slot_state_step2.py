from _adamw_closure_evidence import GATE, load_json


def test_live_slot_state_step2():
    step = load_json("fresh_live_adamw_comparison.json")["steps"][1]
    assert step["momentum_max_abs"] <= GATE
    assert step["velocity_max_abs"] == 0.0
