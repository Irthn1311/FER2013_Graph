import ast
from pathlib import Path


def test_crs_train_entry_is_train_only_and_imports_locked_runner():
    path = Path(__file__).resolve().parents[3] / "notebooks" / "crs-train-entry.py"
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "main" in imports
    assert "crs_train_runner" in source
    assert "public" not in source.lower()
    assert "private" not in source.lower()
    assert "test.csv" not in source.lower()
