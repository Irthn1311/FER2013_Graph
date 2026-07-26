from _adamw_closure_evidence import GATE, PACKAGE_ROOT, load_json


def test_adamw_lerp_primitive():
    source = (
        PACKAGE_ROOT / "src" / "lap_gnn_tf" / "training" / "optimizer.py"
    ).read_text(encoding="utf-8")
    step = load_json("production_optimizer_offline_closure.json")["steps"][1]
    assert "_software_fma(" in source
    assert step["momentum"]["max_abs"] <= GATE
