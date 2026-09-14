import ast
import json
import re
from pathlib import Path


EXECUTION_WRAPPER_SHA = "4abcdeaefb95d73038d9dab5299407dcb6c534fa"


def test_all_four_shard_notebooks_lock_exact_predeclared_schedule():
    root = Path(__file__).resolve().parents[3] / "notebooks"
    observed = {}
    for shard in "ABCD":
        path = root / f"pixel-relational-motif-e0r-r2-shard-{shard.lower()}-kaggle.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["metadata"]["kernelspec"]["name"] == "python3"
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        compile(code, str(path), "exec")
        wrapper_match = re.search(r"(?m)^EXECUTION_WRAPPER_SHA = (.+)$", code)
        shard_match = re.search(r"(?m)^SHARD_NAME = (.+)$", code)
        assert wrapper_match and ast.literal_eval(wrapper_match.group(1)) == EXECUTION_WRAPPER_SHA
        assert shard_match and ast.literal_eval(shard_match.group(1)) == shard
        assert "checkout','--detach',EXECUTION_WRAPPER_SHA" in code
        assert "e0r-r2-shard-entry.py" in code
        observed[shard] = path.name
    assert set(observed) == set("ABCD")
