import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
WRAPPER_SHA = "0b030632db66ba7e8fe47c1240f1a27e412b9ef4"


def test_all_map_notebooks_have_kernel_and_exact_wrapper_lock():
    paths = sorted((ROOT / "notebooks").glob("pixel-relational-motif-m0-*-kaggle.ipynb"))
    assert len(paths) == 9
    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["metadata"]["kernelspec"]["name"] == "python3"
        source = "".join(part for cell in notebook["cells"] if cell["cell_type"] == "code" for part in cell.get("source", []))
        assert WRAPPER_SHA in source
        assert "PrivateTest" not in source and "test.csv" not in source
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))


def test_map_schedule_is_exact():
    expected = {
        "g42": ("G", [42]), "g43": ("G", [43]), "g44": ("G", [44]),
        "g45": ("G", [45]), "g46": ("G", [46]),
        "m42-44": ("M", [42, 43, 44]), "m45-46": ("M", [45, 46]), "l": ("L", []),
    }
    for slug, (family, seeds) in expected.items():
        source = (ROOT / f"notebooks/pixel-relational-motif-m0-{slug}-kaggle.ipynb").read_text(encoding="utf-8")
        assert f"'--family','{family}'" in source
        for seed in seeds:
            assert f"'{seed}'" in source
