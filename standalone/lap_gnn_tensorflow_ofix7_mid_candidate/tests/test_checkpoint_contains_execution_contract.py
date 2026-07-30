import json

import tensorflow as tf

from lap_gnn_tf.training.checkpointing import CheckpointPolicy

from _execution_contract_evidence import CONTRACT_SHA


def test_checkpoint_contains_execution_contract(tmp_path):
    model = tf.keras.Sequential([
        tf.keras.layers.Input((1,)),
        tf.keras.layers.Dense(1),
    ])
    optimizer = tf.keras.optimizers.SGD()
    model.compile(optimizer=optimizer)
    model(tf.constant([[1.0]], tf.float32))
    policy = CheckpointPolicy(tmp_path)
    result = policy.update_best(
        model,
        optimizer,
        1,
        {"macro_f1": 0.0, "accuracy": 0.0},
        {
            "execution_contract_sha256": CONTRACT_SHA,
            "scheduler_state": {"best": None},
            "early_stopping_state": {"bad_epochs": 0},
        },
    )
    assert result["saved"] == ["best_val_accuracy"]
    metadata = json.loads(
        (tmp_path / "checkpoints" / "best_val_accuracy.metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["execution_contract_sha256"] == CONTRACT_SHA
    assert "scheduler_state" in metadata
    assert "early_stopping_state" in metadata
    checkpoint_names = sorted(
        path.name for path in (tmp_path / "checkpoints").iterdir()
    )
    assert checkpoint_names == [
        "best_val_accuracy.keras",
        "best_val_accuracy.metadata.json",
        "best_val_accuracy.weights.h5",
    ]
