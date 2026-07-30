from types import SimpleNamespace

import pytest

from lap_gnn_tf.training.trainer import resolve_final_checkpoint


POLICY = SimpleNamespace(best_accuracy_epoch=32)


def test_resolve_final_checkpoint():
    config = {"training": {"final_test_checkpoint": "best_val_accuracy"}}
    assert resolve_final_checkpoint(config, POLICY) == (
        "best_val_accuracy.keras",
        32,
    )


def test_resolve_final_checkpoint_defaults_to_accuracy():
    assert resolve_final_checkpoint({"training": {}}, POLICY) == (
        "best_val_accuracy.keras",
        32,
    )


@pytest.mark.parametrize("requested", ["best", "best_val_macro_f1", "test_accuracy"])
def test_resolve_final_checkpoint_rejects_non_accuracy_policy(requested):
    config = {"training": {"final_test_checkpoint": requested}}
    with pytest.raises(ValueError, match="final_test_checkpoint"):
        resolve_final_checkpoint(config, POLICY)
