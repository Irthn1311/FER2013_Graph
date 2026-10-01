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
    assert 'for package_name in ("mpg_fer_v2_3", "mpg_fer_table_vi")' in code
    assert "for name in sorted(EMBEDDED_SOURCES)" not in code
    assert "validate_kaggle_mounted_input_contract(INPUT_ROOT)" in code
    assert code.index("validate_kaggle_mounted_input_contract(INPUT_ROOT)") < code.index(
        'exactly_one("FINAL_RECIPE_LOCK.json")'
    )
    assert 'exactly_one("FINAL_RECIPE_LOCK.json")' in code
    assert 'exactly_one("ABLATION_DESIGN_LOCK.json")' in code
    assert 'exactly_one("train.csv")' in code
    assert 'exactly_one("val.csv")' in code
    assert 'exactly_one("test.csv")' in code
    assert code.index('exactly_one("test.csv")') < code.index(
        "run_ablation_training("
    )
    assert "shutil.make_archive" in code
    implementation_commit = (
        ROOT / "ABLATION_IMPLEMENTATION_COMMIT.txt"
    ).read_text(encoding="utf-8").strip()
    assert (
        f'os.environ["MPG_FER_ABLATION_IMPLEMENTATION_COMMIT"] = '
        f"'{implementation_commit}'"
    ) in code
    assert (
        f'os.environ["MPG_FER_SOURCE_GIT_COMMIT"] = \'{implementation_commit}\''
        in code
    )
    assert 'os.environ["MPG_FER_ARCHITECTURE_BASE_COMMIT"]' in code
    assert actual["metadata"]["mpg_fer_ablation_issue"] == 106
    assert actual["metadata"]["ablation_implementation_commit"] == implementation_commit
    assert actual["metadata"]["private_test_permitted"] is True


def test_design_lock_binds_notebook_source_recipe_and_authorizes_seven_jobs() -> None:
    design = json.loads(
        (ROOT / "ABLATION_DESIGN_LOCK.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (ROOT / "source_checksum_manifest.json").read_text(encoding="utf-8")
    )
    implementation_commit = (
        ROOT / "ABLATION_IMPLEMENTATION_COMMIT.txt"
    ).read_text(encoding="utf-8").strip()
    assert design["architecture_base_commit"] == "232e7a9f09251e7c3353684d34351356bd2b023b"
    assert design["ablation_implementation_commit"] == implementation_commit
    assert manifest["architecture_base_commit"] == design["architecture_base_commit"]
    assert manifest["ablation_implementation_commit"] == implementation_commit
    assert design["ablation_source_sha256"] == manifest["ablation_source_tree_sha256"]
    assert design["notebook_sha256"] == manifest["notebook"]["sha256"]
    assert design["final_recipe_lock_sha256"]
    assert design["scientific_training_authorized"] is True
    assert design["final_test_reporting_authorized"] is True
    assert design["private_test_permitted"] is True
    assert design["private_test_selection_permitted"] is False
    assert design["kaggle_mounted_input_contract"] == {
        "input_root": "/kaggle/input",
        "required_unique_basenames": ["train.csv", "val.csv", "test.csv"],
        "ambiguous_path_markers": [
            "private-test",
            "private_test",
            "privatetest",
        ],
        "inspection": "path_names_only_no_file_open",
    }
    authorization = json.loads(
        (ROOT / "ABLATION_LAUNCH_AUTHORIZATION.json").read_text(encoding="utf-8")
    )
    assert len(authorization["jobs"]) == 7
    assert len({job["mode"] for job in authorization["jobs"]}) == 7
    assert len({job["run_id"] for job in authorization["jobs"]}) == 7
    assert authorization["full_control_launched"] is False
