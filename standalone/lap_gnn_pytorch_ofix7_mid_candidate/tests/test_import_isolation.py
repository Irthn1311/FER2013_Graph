from pathlib import Path

import lap_gnn

from _helpers import ROOT


def test_runtime_import_resolves_inside_package():
    resolved = Path(lap_gnn.__file__).resolve()
    assert resolved.name == "__init__.py"
    assert "lap_gnn" in resolved.parts
    if "FER_2013_GRAPH" in str(resolved):
        assert resolved.is_relative_to((ROOT / "src").resolve())
    assert not any(name in __import__("sys").modules for name in ("d16", "d17", "d18", "d19"))
