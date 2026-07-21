import pytest
from d16.scripts.analyze_ofix7_mid_5seed import require_policy_lock


def test_test_reveal_requires_lock(tmp_path):
    with pytest.raises(RuntimeError): require_policy_lock(tmp_path / "checkpoint_policy_lock.json")
