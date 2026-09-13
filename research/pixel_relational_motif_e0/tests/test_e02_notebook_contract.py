import ast
import json
import re
from pathlib import Path


SOURCE_SHA = "d980037fe5d004200ffd7cbb2f1b86176529e46a"
DICTIONARY_SHA = "68154a054f712bb07692146904bcba57f10e079c7efc92723aa0bccba9f6273b"


def test_e02_kaggle_notebook_contract():
    notebook = Path(__file__).resolve().parents[3] / "notebooks" / "pixel-relational-motif-e02-kaggle.ipynb"
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
    assert "RUN_E02 = True" in code
    assert "run_e02" in code
    assert "e02_summary.json" in code
    assert "e02_results.npz" in code
    assert "source lock mismatch" in code
    assert "PrivateTest" in code
    assert "test.csv" not in code
    assert code.index("pytest") < code.index("TRAIN_CSV.is_file")
    assert "allow_nan=False" in code
