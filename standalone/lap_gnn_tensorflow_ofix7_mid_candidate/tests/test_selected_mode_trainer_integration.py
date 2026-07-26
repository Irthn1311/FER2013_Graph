from pathlib import Path

from lap_gnn_tf.config import load_config
from lap_gnn_tf.training.execution import G1_A_OPTIONS

from _execution_contract_evidence import ROOT


def test_selected_mode_trainer_integration():
    config = load_config(
        ROOT / "configs" / "fer2013_ofix7_mid_tensorflow_seed42.yaml"
    )
    assert config["training"]["gradient_execution_mode"] == "tf_function"
    assert config["training"]["optimizer_execution_mode"] == (
        "restricted_tf_function"
    )
    assert config["training"]["grappler_profile"] == "G1-A"
    assert G1_A_OPTIONS == {
        "arithmetic_optimization": False,
        "remapping": False,
    }
    source = (
        ROOT / "src" / "lap_gnn_tf" / "training" / "trainer.py"
    ).read_text(encoding="utf-8")
    assert "build_restricted_graph_train_step" in source
    assert '"execution_contract_sha256"' in source
    assert "model.compile(optimizer=optimizer, run_eagerly=False)" in source
