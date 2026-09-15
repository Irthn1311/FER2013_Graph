import ast
import json
import re
from pathlib import Path


ORCHESTRATION_SHA = "8b43d930c62b5f4fe808fd596645bff7e10dcf65"


def test_aggregate_notebook_is_locked_to_exact_five_inputs_and_zero_fit_entry():
    path = Path(__file__).resolve().parents[3] / "notebooks" / "pixel-relational-motif-e0r-r2-aggregate-kaggle.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    compile(code, str(path), "exec")
    match = re.search(r"(?m)^ORCHESTRATION_SHA = (.+)$", code)
    assert match and ast.literal_eval(match.group(1)) == ORCHESTRATION_SHA
    assert "e0r-r2-aggregate-entry.py" in code
    assert "pgm-e0r-v538-r1-frozen" not in code  # resolved and verified inside the locked entry
    for slug in (
        "pgm-e0r-r2-shard-a-v539",
        "pgm-e0r-r2-shard-b-v2",
        "pgm-e0r-r2-shard-c-v3",
        "pgm-e0r-r2-shard-d-v3",
    ):
        assert slug in code
    assert "test.csv" not in code
    assert ".fit(" not in code
