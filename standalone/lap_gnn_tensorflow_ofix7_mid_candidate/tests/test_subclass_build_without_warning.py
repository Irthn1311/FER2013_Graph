import warnings

from _helpers import loaded


def test_subclass_build_without_warning():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model, batch = loaded()
        model(batch, training=False)
    messages = [str(item.message) for item in captured]
    assert not any("does not have a `build()` method" in message for message in messages)
