from pathlib import Path

from lap_gnn_tf.config import load_config
from lap_gnn_tf.data.graph_generator import GraphBatchGenerator
from lap_gnn_tf.training.execution import (
    G1_A_OPTIONS,
    MAX_REGISTERED_TRAIN_STEP_TRACES,
    build_restricted_graph_train_step,
)

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
    assert "input_signature=GraphBatchGenerator.output_signature()" in source
    assert "experimental_get_tracing_count" in source
    assert MAX_REGISTERED_TRAIN_STEP_TRACES == 1


def test_selected_train_step_has_dynamic_graph_signature():
    train_step = build_restricted_graph_train_step(
        model=None,
        optimizer=None,
        input_signature=GraphBatchGenerator.output_signature(),
    )
    signature = train_step.input_signature
    assert signature is not None
    batch_signature = signature[0]
    assert batch_signature["node_features"].shape == (None, 37)
    assert batch_signature["edge_index"].shape == (2, None)
    assert batch_signature["edge_features"].shape == (None, 8)
    assert batch_signature["labels"].shape == (None,)
