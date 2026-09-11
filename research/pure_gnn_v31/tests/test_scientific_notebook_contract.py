"""Tests verifying scientific notebook contracts, fail-closed guards, and absence of test paths."""

import json
from pathlib import Path
import pytest


def test_scientific_notebook_contracts():
    nb_path = (
        Path(__file__).resolve().parents[3]
        / "notebooks"
        / "pure-gnn-v31-scientific-screen-kaggle.ipynb"
    )
    assert nb_path.is_file(), f"Notebook not found: {nb_path}"

    with nb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    # 1. Top markdown assertion
    cells = nb.get("cells", [])
    assert len(cells) > 0
    top_source = "".join(cells[0].get("source", []))
    assert "PURE-GNN v3.1 HISTORICAL-COMPATIBLE SCIENTIFIC SCREEN" in top_source
    assert "STATUS: SOURCE REVIEW REQUIRED" in top_source
    assert "SCIENTIFIC TRAINING DISABLED" in top_source

    # 2. Check code cells: compile each code cell to ensure valid Python syntax
    all_code = ""
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            all_code += "\n" + source
            try:
                compile(source, filename=f"cell_{idx}", mode="exec")
            except SyntaxError as e:
                pytest.fail(f"Notebook code cell {idx} failed syntax compilation: {e}")

    # 3. Assert no filesystem path to test.csv in notebook code
    lower_code = all_code.lower()
    assert "test.csv" not in lower_code
    assert "/test/" not in lower_code
    assert "official_test" not in lower_code

    # 4. Assert fail-closed guard
    assert "RUN_SCIENTIFIC_SCREEN = False" in all_code
    assert "raise PermissionError" in all_code
    assert "REVIEWED_SOURCE_TAG = None" in all_code

    # 5. Assert fail-closed behavior on missing data
    assert "raise FileNotFoundError" in all_code
