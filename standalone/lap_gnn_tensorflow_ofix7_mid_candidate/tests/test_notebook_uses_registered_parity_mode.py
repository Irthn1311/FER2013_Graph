import json
from pathlib import Path


def test_notebook_uses_registered_parity_mode():
    notebook = Path(__file__).resolve().parents[3] / "notebooks" / "kaggle-end-to-end.ipynb"
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert source.count("lap_gnn_tf.cli.train") == 1
    assert "SEED = 42" in source
    assert "RESUME = False" in source
    assert "XLA_ENABLED = False" in source
    assert "READY_FOR_TENSORFLOW_KAGGLE_SEED42" in source
    assert "lap_gnn_tf.cli.compare_golden" in source
