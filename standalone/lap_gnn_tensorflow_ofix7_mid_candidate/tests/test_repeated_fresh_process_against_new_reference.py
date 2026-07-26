from _adamw_closure_evidence import load_json


def test_repeated_fresh_process_against_new_reference():
    assert load_json("repeated_determinism.json")["fresh_process_pass"]
