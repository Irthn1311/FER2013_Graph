import pytest
from d16.scripts.run_ofix7_mid_replication import refuse_resume


def test_runner_refuses_resume():
    with pytest.raises(RuntimeError): refuse_resume({}, False)
    with pytest.raises(RuntimeError): refuse_resume({"init_checkpoint": "x.pt"}, True)
