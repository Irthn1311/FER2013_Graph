import json

from _helpers import ROOT


def test_all_recorded_layers_match_parent():
    result = json.loads((ROOT / "validation_assets/parity_results.json").read_text())
    assert result["forward_parity_pass"]
    assert all(item["max_abs"] <= 1e-6 for item in result["layer_metrics"].values())
