"""Tests statically auditing new scientific code and notebook for forbidden operators and creep."""

from pathlib import Path
import pytest


FORBIDDEN_TOKENS = [
    "Conv1D",
    "Conv2D",
    "Conv3D",
    "MultiHeadAttention",
    "transformer",
    "knn",
    "mediapipe",
    "landmark",
    "ROI",
    "SupCon",
]


def test_static_source_safety_scan():
    sci_dir = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "pure_gnn_v31"
        / "scientific"
    )
    nb_path = (
        Path(__file__).resolve().parents[3]
        / "notebooks"
        / "pure-gnn-v31-scientific-screen-kaggle.ipynb"
    )

    targets = list(sci_dir.glob("*.py"))
    if nb_path.is_file():
        targets.append(nb_path)

    for target in targets:
        content = target.read_text(encoding="utf-8")

        # 1. Check forbidden tokens
        for token in FORBIDDEN_TOKENS:
            assert token not in content, (
                f"Forbidden token '{token}' found in scientific file: {target.name}"
            )

        # 2. Check that scientific source does not redefine PureGNNv31 class
        if target.name != "model.py":
            assert "class PureGNNv31" not in content, (
                f"Illegal redefinition of PureGNNv31 class found in {target.name}"
            )
