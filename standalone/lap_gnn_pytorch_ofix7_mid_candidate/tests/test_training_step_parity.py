import json

from _helpers import ROOT


def test_two_step_parent_parity_passed():
    result = json.loads((ROOT / "validation_assets/parity_results.json").read_text())
    assert result["optimizer_steps"] == 2
    assert result["completed_epoch"] is False
    assert result["training_step_parity_pass"]
