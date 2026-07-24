import ast

from _helpers import ROOT


def test_no_parent_imports():
    forbidden = ("d16", "d17", "d18", "d19", "lap_gnn")
    for path in (ROOT / "src" / "lap_gnn_tf").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = node.module if isinstance(node, ast.ImportFrom) else None
            if module and not module.startswith("lap_gnn_tf"):
                assert not any(module == name or module.startswith(name + ".") for name in forbidden)

