import ast
import json
import re
from pathlib import Path


SOURCE_SHA = "671e3c2f69607778f08a923026e76744561e13b2"
DICTIONARY_SHA = "68154a054f712bb07692146904bcba57f10e079c7efc92723aa0bccba9f6273b"


def test_e0r_kaggle_notebook_contract():
    notebook = Path(__file__).resolve().parents[3] / "notebooks" / "pixel-relational-motif-e0r-kaggle.ipynb"
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    assert nb["metadata"]["kernelspec"]["name"] == "python3"
    code = ""
    for index, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            compile(source, filename=f"cell_{index}", mode="exec")
            code += "\n" + source

    source_match = re.search(r"(?m)^SOURCE_SHA = (.+)$", code)
    dictionary_match = re.search(r"(?m)^EXPECTED_DICTIONARY_SHA256 = (.+)$", code)
    assert source_match and ast.literal_eval(source_match.group(1)) == SOURCE_SHA
    assert dictionary_match and ast.literal_eval(dictionary_match.group(1)) == DICTIONARY_SHA
    assert "RUN_TESTS = True" in code
    assert "RUN_E0R = True" in code
    assert "run_e0r" in code
    assert "pgm-e01-v533-dictionary" in code
    assert "pgm-e02-v535-artifacts" in code
    assert "EXPECTED_CONTROL_SHA256" in code
    assert "e0r_r1_results.npz" in code
    assert "e0r_r2_status.json" in code
    assert "e0r_r2_results.npz" in code
    assert "source lock mismatch" in code
    assert "PrivateTest" in code
    assert "test.csv" not in code
    assert code.index("pytest") < code.index("TRAIN_CSV.is_file")
    assert "heartbeat" in code
