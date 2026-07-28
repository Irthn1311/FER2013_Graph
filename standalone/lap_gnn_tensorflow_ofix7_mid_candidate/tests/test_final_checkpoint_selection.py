from types import SimpleNamespace

import pytest

from lap_gnn_tf.training.trainer import resolve_final_checkpoint


POLICY = SimpleNamespace(best_macro_epoch=26, best_accuracy_epoch=32)


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("best", ("best_val_macro_f1.keras", 26)),
        ("best_val_macro_f1", ("best_val_macro_f1.keras", 26)),
        ("best_val_accuracy", ("best_val_accuracy.keras", 32)),
    ],
)
def test_resolve_final_checkpoint(requested, expected):
    config = {"training": {"final_test_checkpoint": requested}}
    assert resolve_final_checkpoint(config, POLICY) == expected


def test_resolve_final_checkpoint_rejects_unknown_value():
    config = {"training": {"final_test_checkpoint": "test_accuracy"}}
    with pytest.raises(ValueError, match="final_test_checkpoint"):
        resolve_final_checkpoint(config, POLICY)
