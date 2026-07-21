import pytest
from d16.scripts.run_ofix7_mid_replication import refuse_contaminated_output


def test_runner_refuses_contaminated_output(tmp_path):
    (tmp_path / "partial.txt").write_text("x")
    with pytest.raises(RuntimeError): refuse_contaminated_output(tmp_path)
