from _helpers import loaded


def test_weight_mapping_complete():
    model, _ = loaded()
    bindings = model.state_bindings()
    assert len(bindings) == 127
    assert len({item.source_key for item in bindings}) == 127
    assert len({id(item.variable) for item in bindings}) == 127

