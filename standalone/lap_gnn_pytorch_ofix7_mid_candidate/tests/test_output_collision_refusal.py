import pytest

from lap_gnn.cli.common import refuse_output_collision


def test_nonempty_output_is_refused(tmp_path):
    (tmp_path / "existing.txt").write_text("occupied")
    with pytest.raises(FileExistsError):
        refuse_output_collision(tmp_path)
