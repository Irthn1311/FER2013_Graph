import ast
import json
import re
from pathlib import Path


SOURCE_SHA = "175320c4a7215976891ba7d56c0c88bab168e873"
EXPECTED_PUBLIC_SHA256 = "412036d077c6ec203047b2935ab14bc858d8136ee26e8db3e23023f1fc9dee08"
EXPECTED_DICTIONARY_SHA256 = "68154a054f712bb07692146904bcba57f10e079c7efc92723aa0bccba9f6273b"
EXPECTED_TRAIN_MODEL_SHA256 = "77b8a41d4a7de79b2216b4b9c7ad2d19e326cac25a46ec2a7a3dcaaa1f95607a"


def _constant(code: str, name: str):
    match = re.search(rf"(?m)^{name} = (.+)$", code)
    assert match
    return ast.literal_eval(match.group(1))


def test_public_notebook_is_source_locked_inference_only_and_tests_before_data():
    path = (
        Path(__file__).resolve().parents[3]
        / "notebooks"
        / "pixel-relational-composition-crs-public-kaggle.ipynb"
    )
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    compile(code, str(path), "exec")

    assert _constant(code, "SOURCE_SHA") == SOURCE_SHA
    assert _constant(code, "EXPECTED_PUBLIC_SHA256") == EXPECTED_PUBLIC_SHA256
    assert _constant(code, "EXPECTED_DICTIONARY_SHA256") == EXPECTED_DICTIONARY_SHA256
    assert _constant(code, "EXPECTED_TRAIN_MODEL_SHA256") == EXPECTED_TRAIN_MODEL_SHA256
    assert "research/pixel-relational-composition-crs" in code
    assert "val.csv" in code
    assert "e01_dictionary.npz" in code
    assert "crs_train_model.npz" in code
    assert "crs-public-entry.py" in code
    assert "pytest" in code
    assert code.index("pytest") < code.index("locked_inputs")
    assert code.index("locked_inputs") < code.index("crs-public-entry.py")
    assert ".fit(" not in code

    assert "CRS_PUBLIC_SOURCE_SHA" in code
    assert "nuyntai/pgm-crs-public-issue-84" in code
    assert "--public-csv" in code
    assert "--dictionary-npz" in code
    assert "--train-model-npz" in code
    assert "--output-dir" in code
    assert "--train-csv" not in code
    assert "--private-csv" not in code
    assert "--test-csv" not in code
    assert "test.csv" not in code.lower()

    assert "crs_public_predictions.npz" in code
    assert "crs_public_summary.json" in code
    assert "crs_public_execution_manifest.json" in code
