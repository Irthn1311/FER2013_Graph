from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _generator():
    path = ROOT / "tools" / "sync_ablation_notebook.py"
    spec = importlib.util.spec_from_file_location("sync_ablation_notebook", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_canonical_ablation_notebook_matches_generator_and_compiles_every_cell() -> (
    None
):
    generator = _generator()
    notebook_path = ROOT / "notebooks" / "MPG_FER_Table_VI_Ablation_Kaggle_T4.ipynb"
    actual = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert actual == generator.build_notebook()
    code_cells = [
        (index, "".join(cell["source"]))
        for index, cell in enumerate(actual["cells"])
        if cell["cell_type"] == "code"
    ]
    assert len(code_cells) == 5
    for index, source in code_cells:
        compile(source, f"ablation-notebook-cell-{index}", "exec")
    code = "\n".join(source for _, source in code_cells)
    assert 'secrets.get_secret("MPG_FER_ABLATION_MODE")' in code
    assert 'secrets.get_secret("MPG_FER_ABLATION_RUN_ID")' in code
    assert "run_ablation_training(" in code
    assert 'exactly_one("FINAL_RECIPE_LOCK.json")' in code
    assert 'exactly_one("ABLATION_DESIGN_LOCK.json")' in code
    assert 'exactly_one("train.csv")' in code
    assert 'exactly_one("val.csv")' in code
    assert 'exactly_one("test.csv")' not in code
    assert "evaluate_private_once" not in code
    assert "create_private_dataloader" not in code
    assert "shutil.make_archive" in code
    assert actual["metadata"]["mpg_fer_ablation_issue"] == 101
    assert actual["metadata"]["private_test_permitted"] is False


def test_design_lock_binds_notebook_and_source_but_refuses_training() -> None:
    design = json.loads(
        (ROOT / "ABLATION_DESIGN_LOCK.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (ROOT / "source_checksum_manifest.json").read_text(encoding="utf-8")
    )
    assert design["base_commit"] == "232e7a9f09251e7c3353684d34351356bd2b023b"
    assert design["ablation_source_sha256"] == manifest["ablation_source_tree_sha256"]
    assert design["notebook_sha256"] == manifest["notebook"]["sha256"]
    assert design["final_recipe_lock_sha256"] is None
    assert design["scientific_training_authorized"] is False
    assert design["private_test_permitted"] is False
