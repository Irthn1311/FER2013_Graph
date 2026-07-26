from pathlib import Path


def test_checkpoint_policy_contract():
    text = (Path(__file__).resolve().parents[1] / "contracts" / "training_contract.json").read_text()
    assert '"checkpoint_primary": "val_macro_f1"' in text
    assert '"checkpoint_secondary": "val_accuracy"' in text
    assert '"test_used_for_selection": false' in text

