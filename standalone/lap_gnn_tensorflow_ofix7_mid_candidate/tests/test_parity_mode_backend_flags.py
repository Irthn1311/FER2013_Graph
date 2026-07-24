import json
from pathlib import Path


def test_parity_mode_backend_flags():
    notebook = Path(__file__).resolve().parents[3] / "notebooks" / "kaggle-end-to-end.ipynb"
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert 'parity_env["CUDA_VISIBLE_DEVICES"] = ""' in source
    assert 'parity_env["TF_DETERMINISTIC_OPS"] = "1"' in source
    assert 'parity_env["TF_ENABLE_ONEDNN_OPTS"] = "0"' not in source
    assert "lap_gnn_tf.cli.compare_golden" in source
