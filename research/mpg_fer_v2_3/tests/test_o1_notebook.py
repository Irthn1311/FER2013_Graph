from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _generator():
    path = ROOT / "tools" / "sync_o1_notebook.py"
    spec = importlib.util.spec_from_file_location("sync_o1_notebook", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_canonical_o1_notebook_matches_generator_and_compiles() -> None:
    generator = _generator()
    notebook = ROOT / "notebooks" / "MPG_FER_O1_Wave1_Kaggle_T4.ipynb"
    actual = json.loads(notebook.read_text(encoding="utf-8"))
    assert actual == generator.build_notebook()
    code_cells = [
        "".join(cell["source"])
        for cell in actual["cells"]
        if cell["cell_type"] == "code"
    ]
    assert len(code_cells) == 5
    for index, code in enumerate(code_cells):
        compile(code, f"o1-notebook-cell-{index}", "exec")
    code = "\n".join(code_cells)
    guard = "validate_mounted_input_firewall(INPUT_ROOT)"
    assert guard in code
    assert code.index(guard) < code.index('exactly_one("O1_HPO_DESIGN_LOCK.json")')
    assert code.index(guard) < code.index('exactly_one("train.csv")')
    assert 'secrets.get_secret("MPG_FER_O1_CONFIG_ID")' in code
    assert "if CONFIG_ID not in O1_REGISTRY" in code
    assert 'exactly_one("test.csv")' not in code
    assert "evaluate_private_once" not in code
    assert "shutil.make_archive" in code
    assert actual["metadata"]["private_test_permitted"] is False


def test_o1_design_lock_binds_source_notebook_and_refuses_wave1() -> None:
    design = json.loads((ROOT / "O1_HPO_DESIGN_LOCK.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "O1_SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    implementation = (ROOT / "O1_IMPLEMENTATION_COMMIT.txt").read_text(encoding="utf-8").strip()
    assert design["o1_implementation_commit"] == implementation
    assert manifest["o1_implementation_commit"] == implementation
    assert design["o1_source_tree_sha256"] == manifest["o1_source_tree_sha256"]
    assert design["notebook_sha256"] == manifest["notebook"]["sha256"]
    assert design["wave1_execution_authorized"] is False
    assert design["private_test_permitted"] is False
    assert design["screen_stop_epoch"] == 65
