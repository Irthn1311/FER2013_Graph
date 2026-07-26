from _adamw_closure_evidence import load_json


def test_optimizer_restore_continuation():
    result = load_json("checkpoint_continuation.json")
    assert result["pass"]
    assert result["fresh_process_restore"]["max_abs"] == 0.0
    assert result["fresh_process_continuation"]["max_abs"] == 0.0
