import json

from _helpers import ROOT


def test_anchor_and_readout_token_semantics():
    schema = json.loads((ROOT / "contracts/node_schema.json").read_text())
    assert schema["semantic_anchor_nodes"]["message_passing"] is True
    assert schema["semantic_anchor_nodes"]["count"] == 5
    assert schema["readout_cls_and_motif_tokens"]["message_passing"] is False
