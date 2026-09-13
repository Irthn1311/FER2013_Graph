import ast
import json
import re
from pathlib import Path


SOURCE_SHA = "3858b3e17b9ea6cc591381e3096be53c27581c02"
DICTIONARY_SHA = "68154a054f712bb07692146904bcba57f10e079c7efc92723aa0bccba9f6273b"


def test_e01b_kaggle_notebook_contract():
    notebook = Path(__file__).resolve().parents[3] / "notebooks" / "pixel-relational-motif-e01b-kaggle.ipynb"
    assert notebook.is_file(), notebook
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    code = ""
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        compile(src, filename=f"cell_{idx}", mode="exec")
        code += "\n" + src

    m = re.search(r"(?m)^SOURCE_SHA = (.+)$", code)
    assert m is not None and ast.literal_eval(m.group(1)) == SOURCE_SHA
    d = re.search(r"(?m)^EXPECTED_DICTIONARY_SHA256 = (.+)$", code)
    assert d is not None and ast.literal_eval(d.group(1)) == DICTIONARY_SHA

    assert "RUN_TESTS = True" in code
    assert "RUN_E01B = True" in code
    assert "PYTHONPATH" in code
    assert "platform.platform()" in code
    assert "sklearn.__version__" in code
    assert "run_e01b" in code
    assert "e01_dictionary.npz" in code
    assert "val.csv" in code
    assert "train.csv" in code
    assert "test.csv" not in code
    assert "PrivateTest" in code
    assert "public_test_labels_used" in code
    assert "registered_verdict_overwritten" in code
