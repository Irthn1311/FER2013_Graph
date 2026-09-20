import ast
import json
import re
from pathlib import Path


SOURCE_SHA = "16a84b2b36f0d3584afd0a547487c4373dc22128"
TRAIN_SHA256 = "deb82c4b4e01b90776a718c34934666b0bdde6696ca1d0149f8fe807a8ff4ba8"
DICTIONARY_SHA256 = "68154a054f712bb07692146904bcba57f10e079c7efc92723aa0bccba9f6273b"
TRAIN_MODEL_SHA256 = "77b8a41d4a7de79b2216b4b9c7ad2d19e326cac25a46ec2a7a3dcaaa1f95607a"


def _constant(code, name):
    match = re.search(rf"(?m)^{name} = (.+)$", code)
    assert match
    return ast.literal_eval(match.group(1))


def test_motif_train_notebook_is_source_locked_test_first_and_train_only():
    path = (
        Path(__file__).resolve().parents[3]
        / "notebooks"
        / "motif-qualification-train-kaggle.ipynb"
    )
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    compile(code, str(path), "exec")
    assert _constant(code, "SOURCE_SHA") == SOURCE_SHA
    assert _constant(code, "TRAIN_SHA256") == TRAIN_SHA256
    assert _constant(code, "DICTIONARY_SHA256") == DICTIONARY_SHA256
    assert _constant(code, "TRAIN_MODEL_SHA256") == TRAIN_MODEL_SHA256
    assert "research/motif-qualification-sparsification" in code
    assert "git', 'checkout', '--detach', SOURCE_SHA" in code
    assert "pytest" in code
    assert code.index("pytest") < code.index("observed = sha256_file(path)")
    assert "pixel_relational_motif_e0.motif_train_runner" in code
    assert "MOTIF_SCIENTIFIC_SHA" in code
    assert "BUILD_SUBSTRATE" in code
    assert "RUN_REPLICATE" in code
    assert "FINALIZE_TRAIN" in code
    assert "train.csv" in code
    for forbidden in (
        "val.csv",
        "test.csv",
        "--public-csv",
        "--private-csv",
        "--test-csv",
    ):
        assert forbidden not in code


def test_notebook_keeps_scientific_parameters_out_of_execution_routing():
    path = (
        Path(__file__).resolve().parents[3]
        / "notebooks"
        / "motif-qualification-train-kaggle.ipynb"
    )
    code = path.read_text(encoding="utf-8")
    for forbidden in (
        "--n-init",
        "--max-iter",
        "--jaccard-cutoff",
        "--support-threshold",
        "--node-cap",
    ):
        assert forbidden not in code
