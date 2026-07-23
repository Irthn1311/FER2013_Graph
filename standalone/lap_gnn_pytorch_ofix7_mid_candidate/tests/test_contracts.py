import json

from _helpers import ROOT


def test_contracts_are_machine_readable():
    for name in (
        "feature_schema.json", "edge_schema.json", "node_schema.json",
        "graph_batch_schema.json", "checkpoint_policy.json",
        "class_mapping.json", "preprocessing_contract.json",
    ):
        assert json.loads((ROOT / "contracts" / name).read_text(encoding="utf-8"))
