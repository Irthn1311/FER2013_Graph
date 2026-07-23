import json

from _helpers import ROOT


def test_parent_graph_parity_was_exact_for_32_samples():
    result = json.loads((ROOT / "validation_assets/parity_results.json").read_text())
    assert result["sample_count"] == 32
    assert result["graph_parity_pass"]
    assert result["graph_max_node_abs"] == 0.0
    assert result["graph_max_edge_abs"] == 0.0
