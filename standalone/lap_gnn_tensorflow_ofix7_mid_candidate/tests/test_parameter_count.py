from _helpers import loaded


def test_parameter_count():
    model, _ = loaded()
    assert sum(int(variable.shape.num_elements()) for variable in model.trainable_variables) == 1_061_192
    assert sum(int(variable.shape.num_elements()) for variable in model.non_trainable_variables) == 0

