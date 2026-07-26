import json
from pathlib import Path


def test_torch_compatible_adamw_live_step1():
    path = Path(__file__).resolve().parents[1] / "validation_assets" / "repair" / "live_adamw_comparison.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["total_optimizer_updates"] == 4
    assert result["steps"][0]["pass_2e_8"], result["steps"][0]
