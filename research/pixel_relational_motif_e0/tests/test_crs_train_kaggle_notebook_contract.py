import ast
import json
import re
from pathlib import Path


SOURCE_SHA = "f2ff796ce01a92b1d4a010113d0c5160f1dc2c3d"
EXPECTED_DICTIONARY_SHA256 = "68154a054f712bb07692146904bcba57f10e079c7efc92723aa0bccba9f6273b"


def test_crs_train_notebook_is_source_locked_train_only_and_runs_tests_first():
    path = (
        Path(__file__).resolve().parents[3]
        / "notebooks"
        / "pixel-relational-composition-crs-train-kaggle.ipynb"
    )
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    compile(code, str(path), "exec")

    source_match = re.search(r"(?m)^SOURCE_SHA = (.+)$", code)
    assert source_match and ast.literal_eval(source_match.group(1)) == SOURCE_SHA
    dict_match = re.search(r"(?m)^EXPECTED_DICTIONARY_SHA256 = (.+)$", code)
    assert dict_match and ast.literal_eval(dict_match.group(1)) == EXPECTED_DICTIONARY_SHA256

    assert "research/pixel-relational-composition-crs" in code
    assert "train.csv" in code
    assert "crs-train-entry.py" in code
    assert "pytest" in code
    assert code.index("pytest") < code.index("crs-train-entry.py")

    # No validation/final split is configured by this Train-only notebook.
    forbidden = (
        "val.csv",
        "test.csv",
        "PUBLIC_CSV",
        "PRIVATE_CSV",
        "--public-csv",
        "--private-csv",
        "--test-csv",
    )
    for token in forbidden:
        assert token not in code

    assert "e01_dictionary.npz" in code
    assert "crs_train_execution_manifest.json" in code
