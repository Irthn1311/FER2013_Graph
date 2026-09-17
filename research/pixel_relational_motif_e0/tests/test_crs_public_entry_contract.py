import ast
from pathlib import Path


def test_crs_public_entry_imports_inference_only_runner():
    path = Path(__file__).resolve().parents[3] / "notebooks" / "crs-public-entry.py"
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
    assert "crs_public_runner" in source
    assert "train-csv" not in source.lower()
    assert "private" not in source.lower()
    assert "test.csv" not in source.lower()
