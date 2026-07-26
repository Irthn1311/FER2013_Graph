from lap_gnn_tf.training.early_stopping import ValidationLossEarlyStopping


def test_early_stopping_semantics():
    stopper = ValidationLossEarlyStopping(min_epochs=30, patience=15)
    assert not stopper.update(1, 1.0)
    stopped = False
    for epoch in range(2, 31):
        stopped = stopper.update(epoch, 1.1)
    assert stopped

