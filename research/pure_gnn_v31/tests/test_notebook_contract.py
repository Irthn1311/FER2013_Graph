"""Static guardrails for the user-run Kaggle technical notebook."""

import json
import importlib
import site
from pathlib import Path

import pytest


def _notebook_text() -> tuple[dict, str]:
    path = Path(__file__).resolve().parents[3] / "notebooks" / "pure-gnn-v31-kaggle-end-to-end.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    return notebook, text


def test_notebook_uses_immutable_tag_and_locked_scientific_switch():
    notebook, text = _notebook_text()
    assert "SOURCE_TAG = 'pure-gnn-v31-preflight-v3'" in text
    assert "RUN_RESEARCH_SCREEN = False" in text
    assert "REPO_BRANCH" not in text
    assert "checkout', '--detach', SOURCE_TAG" in text
    assert all(cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert all(cell.get("outputs") == [] for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_notebook_has_only_bounded_train_candidates_and_no_split_creation():
    _, text = _notebook_text()
    assert "doduyquynii/fer13-split" in text
    assert "/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split/train.csv" in text
    assert "/kaggle/input/fer13-split/train.csv" not in text
    assert "create_research_split_manifest" not in text
    assert ".glob(" not in text
    assert ".rglob('*')" in text  # package source manifest only, never Kaggle input discovery


def test_notebook_requires_t4_and_training_step_b32_benchmark():
    _, text = _notebook_text()
    assert "REQUIRED_GPU_SUBSTRING = 'T4'" in text
    assert "benchmark_batch_sizes([16, 32, 64]" in text
    assert "Mandatory B32 inference plus optimizer-step benchmark" in text
    assert "pure_gnn_v31_preflight_evidence.tar.gz" in text


def test_notebook_imports_only_installed_benchmark_module():
    _, text = _notebook_text()
    assert "from pure_gnn_v31.tools.benchmark_runtime import benchmark_batch_sizes" in text
    assert "sys.path.insert" not in text
    assert "sys.path.append" not in text


def test_notebook_reprocesses_editable_install_metadata_before_import():
    _, text = _notebook_text()
    install_index = text.index("pip', 'install'")
    addsitedir_index = text.index("site.addsitedir")
    import_index = text.index("import pure_gnn_v31")
    assert install_index < addsitedir_index < import_index
    assert "importlib.invalidate_caches()" in text
    assert "site.getsitepackages()" in text


def test_addsitedir_activates_pth_written_after_interpreter_start(tmp_path):
    site_directory = tmp_path / "site-packages"
    source_directory = tmp_path / "editable-source"
    package_directory = source_directory / "post_install_probe_package"
    site_directory.mkdir()
    package_directory.mkdir(parents=True)
    (package_directory / "__init__.py").write_text("VALUE = 'editable-metadata-loaded'\n", encoding="utf-8")
    (site_directory / "post-install-probe.pth").write_text(str(source_directory), encoding="utf-8")

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("post_install_probe_package")
    site.addsitedir(str(site_directory))
    importlib.invalidate_caches()
    module = importlib.import_module("post_install_probe_package")
    assert module.VALUE == "editable-metadata-loaded"
