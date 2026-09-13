import ast
import json
import re
from pathlib import Path


SOURCE_SHA = "02ba8623e24880881678e33817d257a6bb1b22b6"


def test_e01_kaggle_notebook_contract():
    notebook = (
        Path(__file__).resolve().parents[3]
        / "notebooks"
        / "pixel-relational-motif-e01-kaggle.ipynb"
    )
    assert notebook.is_file(), notebook
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    assert cells
    top = "".join(cells[0].get("source", []))
    assert "Pixel Relational Motif" in top and "E0.1" in top

    all_code = ""
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        compile(source, filename=f"cell_{idx}", mode="exec")
        all_code += "\n" + source

    source_match = re.search(r'(?m)^SOURCE_SHA = (.+)$', all_code)
    assert source_match is not None
    assert ast.literal_eval(source_match.group(1)) == SOURCE_SHA

    assert 'TRAIN_CSV = FER_INPUT_ROOT / "train.csv"' in all_code
    assert "VAL_CSV" not in all_code
    assert "PUBLIC_TEST_CSV" not in all_code
    assert "PRIVATE_TEST_CSV" not in all_code
    assert 'FER_INPUT_ROOT / "val.csv"' not in all_code
    assert 'FER_INPUT_ROOT / "test.csv"' not in all_code

    assert "checkout" in all_code and "--detach" in all_code
    assert "rev-parse" in all_code and "HEAD" in all_code
    assert "head != SOURCE_SHA" in all_code
    assert "diff" in all_code and "--quiet" in all_code
    assert "sys.path.insert" in all_code
    assert "import pixel_relational_motif_e0" in all_code

    assert "RUN_TESTS = True" in all_code
    assert "RUN_E01 = True" in all_code
    assert "RUN_OCCURRENCE = True" in all_code
    assert "pytest" in all_code
    assert "run_e01" in all_code
    assert "run_occurrence_calibration" in all_code
    assert "heartbeat" in all_code
    assert "e01_occurrence_counts.npz" in all_code

    assert "pip install" not in all_code.lower()
    assert 'private_test_read' in all_code
    assert 'public_test_read' in all_code
