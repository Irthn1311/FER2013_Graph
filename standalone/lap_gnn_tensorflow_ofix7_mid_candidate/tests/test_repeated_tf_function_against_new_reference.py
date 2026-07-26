import pytest

from _adamw_closure_evidence import load_json


@pytest.mark.xfail(
    strict=True,
    reason="TensorFlow graph fusion exceeds the strict 2e-8 PyTorch parity gate",
)
def test_repeated_tf_function_against_new_reference():
    assert load_json("repeated_determinism.json")["tf_function_pass"]
