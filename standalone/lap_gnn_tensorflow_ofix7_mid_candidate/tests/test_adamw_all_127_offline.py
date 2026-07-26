from _adamw_closure_evidence import load_json


def test_adamw_all_127_offline():
    result = load_json("production_optimizer_offline_closure.json")
    assert result["offline_closure_pass"]
    assert all(
        step[kind]["tensor_count"] == 127
        for step in result["steps"]
        for kind in ("parameter", "momentum", "velocity")
    )
