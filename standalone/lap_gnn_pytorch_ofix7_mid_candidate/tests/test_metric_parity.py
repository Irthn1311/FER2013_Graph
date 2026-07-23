import json

from _helpers import ROOT


def test_fixed_array_metrics_match_parent():
    result = json.loads((ROOT / "validation_assets/parity_results.json").read_text())
    assert result["metric_parity_pass"]
    assert max(result["metric_differences"].values()) <= 1e-12
