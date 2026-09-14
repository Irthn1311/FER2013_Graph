import ast
from pathlib import Path


def test_kaggle_shard_entry_is_source_locked_and_predata_tested():
    path = Path(__file__).resolve().parents[3] / "notebooks" / "e0r-r2-shard-entry.py"
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")
    tree = ast.parse(source)
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {
            "SCIENTIFIC_SHA", "CONTINUATION_WRAPPER_SHA", "R1_OCCURRENCES_SHA256", "R1_RESULTS_SHA256"
        }
    }
    assert constants["SCIENTIFIC_SHA"] == "671e3c2f69607778f08a923026e76744561e13b2"
    assert constants["CONTINUATION_WRAPPER_SHA"] == "326eed067431936972b1d47de5957cad53485c7c"
    assert constants["R1_OCCURRENCES_SHA256"] == "30f1a5b642af2ecdfc29ae73960fb01f90c391964db845bd4b33cf3c017f7fa9"
    assert constants["R1_RESULTS_SHA256"] == "b6675ce694f5a607cfea07abb3ed1065753f42ee848f7595d1cb8e682361a0ed"
    assert source.index("pytest") < source.index("train_csv =")
    assert "protected_scientific_diff_empty" in source
    assert "continuation_wrapper_diff_empty" in source
    assert "need exactly one frozen R1 dataset mount" in source
    assert "No PrivateTest path" in source
    assert "run_r2_shard(" in source
    assert "scientific_verdict_written" in source
