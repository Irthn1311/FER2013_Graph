import ast
from pathlib import Path


def test_kaggle_aggregate_entry_is_source_locked_predata_and_zero_fit():
    path = Path(__file__).resolve().parents[3] / "notebooks" / "e0r-r2-aggregate-entry.py"
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")
    tree = ast.parse(source)
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"SCIENTIFIC_SHA", "CONTINUATION_WRAPPER_SHA"}
    }
    assert constants == {
        "SCIENTIFIC_SHA": "671e3c2f69607778f08a923026e76744561e13b2",
        "CONTINUATION_WRAPPER_SHA": "4abcdeaefb95d73038d9dab5299407dcb6c534fa",
    }
    assert source.index("pytest") < source.index("occurrences_path =")
    assert "protected_scientific_diff_empty" in source
    assert "continuation_wrapper_diff_empty" in source
    assert "range(42, 62)" in source
    assert "exactly four distinct shard roots" in source
    assert "No PrivateTest path" in source
    assert "zero model fits" in source
    assert ".fit(" not in source
    assert "_fit_predict_probe" not in source
    assert "aggregate_r2(" in source
