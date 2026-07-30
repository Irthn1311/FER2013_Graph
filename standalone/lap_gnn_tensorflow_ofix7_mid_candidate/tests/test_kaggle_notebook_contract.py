import json
from pathlib import Path


def test_kaggle_notebook_contract():
    notebook = Path(__file__).resolve().parents[3] / "notebooks" / "kaggle-end-to-end.ipynb"
    if not notebook.is_file():
        return
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    for token in [
        "RUN_FULL_TRAINING", "ALLOW_CPU_TRAINING", "lap_gnn_tf",
        "READY_FOR_TENSORFLOW_KAGGLE_SEED42", "ofix7_mid_seed42_tensorflow_outputs.zip",
        'FINAL_TEST_CHECKPOINT = "best_val_accuracy"',
        'selected_metrics_name = f"test_metrics_{selected_checkpoint_stem}.json"',
        '"best_val_accuracy.keras"',
        '"best_val_accuracy.weights.h5"',
        '"best_val_accuracy.metadata.json"',
        "Single-checkpoint inventory mismatch",
    ]:
        assert token in source
    for forbidden in [
        '"best.keras"',
        '"best_val_macro_f1.keras"',
        '"last.keras"',
    ]:
        assert forbidden not in source
