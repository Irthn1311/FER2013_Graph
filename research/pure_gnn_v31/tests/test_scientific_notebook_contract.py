"""Tests verifying scientific notebook contracts, fail-closed guards, and absence of test paths."""

import ast
import json
import re
from pathlib import Path
import pytest
import yaml


AUTHORIZED_TAG = "pure-gnn-v31-scientific-screen-v1"


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

    # 4. Assert calls and imports for canonical runner and source lock
    assert "verify_immutable_source_lock" in all_code
    assert "run_production_scientific_screen" in all_code
    assert "load_scientific_config" in all_code
    assert "CONFIG_PATH" in all_code

    # 5. Assert fresh-kernel bootstrap order: PACKAGE_SRC and sys.path before pure_gnn_v31 import
    assert "PACKAGE_SRC" in all_code
    assert "sys.path.insert" in all_code
    package_src_idx = all_code.find("sys.path.insert")
    first_import_idx = all_code.find("import pure_gnn_v31")
    assert package_src_idx != -1 and first_import_idx != -1
    assert package_src_idx < first_import_idx, "sys.path.insert must appear before first pure_gnn_v31 import"

    # 6. Assert PACKAGE_PATH is assigned before its first use (.is_dir() and / "tests")
    assert "PACKAGE_PATH =" in all_code
    package_path_assign_idx = all_code.find("PACKAGE_PATH =")
    first_isdir_idx = all_code.find("PACKAGE_PATH.is_dir()")
    first_tests_idx = all_code.find('PACKAGE_PATH / "tests"')
    assert package_path_assign_idx != -1
    assert first_isdir_idx != -1 and package_path_assign_idx < first_isdir_idx, (
        "PACKAGE_PATH assignment must appear before PACKAGE_PATH.is_dir()"
    )
    assert first_tests_idx != -1 and package_path_assign_idx < first_tests_idx, (
        "PACKAGE_PATH assignment must appear before PACKAGE_PATH / 'tests'"
    )

    # 7. Assert no pip install -e in notebook
    assert "pip install -e" not in all_code
    assert "pip install" not in all_code

    # 8. Canonical source must be in exactly one coherent authorization state.
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "scientific_screen_historical_v1.yaml"
    )
    config_data = yaml.safe_load(config_path.read_bytes())
    config_authorized = config_data["scientific_execution_authorized"]

    tag_match = re.search(r"(?m)^REVIEWED_SOURCE_TAG = (.+)$", all_code)
    run_match = re.search(r"(?m)^RUN_SCIENTIFIC_SCREEN = (.+)$", all_code)
    assert tag_match is not None
    assert run_match is not None
    reviewed_source_tag = ast.literal_eval(tag_match.group(1))
    run_scientific_screen = ast.literal_eval(run_match.group(1))

    actual_state = (
        config_authorized,
        reviewed_source_tag,
        run_scientific_screen,
    )
    allowed_states = {
        (False, None, False),
        (True, AUTHORIZED_TAG, True),
    }
    assert actual_state in allowed_states, (
        f"Incoherent scientific authorization state: {actual_state}"
    )

    # 9. Assert fail-closed behavior on missing data
    assert "raise FileNotFoundError" in all_code
