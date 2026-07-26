import ast

from _helpers import ROOT


def test_no_torch_runtime():
    for path in (ROOT / "src" / "lap_gnn_tf").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
        assert not any(name == "torch" or name.startswith("torch.") for name in modules)

