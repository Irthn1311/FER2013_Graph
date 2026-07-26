from _adamw_closure_evidence import load_json


def test_repeated_eager_against_new_reference():
    assert load_json("repeated_determinism.json")["eager_pass"]
