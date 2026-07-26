import json

from _execution_contract_evidence import CONTRACT_SHA, ROOT


def test_kaggle_notebook_selected_mode():
    notebook_path = ROOT.parents[1] / "notebooks" / "kaggle-end-to-end.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert CONTRACT_SHA in source
    assert "restricted_tf_function" in source
    assert "G1-A" in source
    assert "arithmetic_optimization" in source
    assert "remapping" in source
    assert "SELECT_G1_RESTRICTED_GRAPH_OPTIMIZER" in source
    assert "grappler_arithmetic_optimization" in source
    assert "grappler_remapping" in source
    assert "checkpoint_continuation" in source
    assert "Fresh output is contaminated" in source
    assert source.index("Fresh output is contaminated") < source.index(
        'print("READY_FOR_TENSORFLOW_KAGGLE_SEED42")'
    )
    assert "checkpoint_inventory" in source
    assert "tensorflow_execution_contract_v2.sha256" in source
    assert "--no-resume" in source
    assert "--no-xla" in source
    assert "Kaggle Save Version" in source
    assert "importlib.metadata.version" in source
    assert "tensorflow imported before bootstrap" in source
    assert "fresh_process_environment" in source
    assert "Restart the Kaggle session" not in source
