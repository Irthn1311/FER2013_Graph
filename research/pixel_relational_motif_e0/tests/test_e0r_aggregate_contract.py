import inspect

import pytest

from pixel_relational_motif_e0 import e0r_aggregate
from pixel_relational_motif_e0.e0r_geometry import GEOMETRY_SEEDS


def test_aggregator_is_zero_fit_and_uses_original_registered_comparison():
    source = inspect.getsource(e0r_aggregate)
    assert ".fit(" not in source
    assert "fit_probe" not in source
    assert "_fit_predict_probe" not in source
    assert "_comparison(" in source
    assert 'stage="R2"' in source


def test_registered_aggregate_seed_set_remains_exactly_twenty():
    assert GEOMETRY_SEEDS == tuple(range(42, 62))


def test_aggregate_rejects_wrong_r1_diagnostics_before_loading_substrate(tmp_path):
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"not canonical")
    with pytest.raises(ValueError, match="diagnostics SHA"):
        e0r_aggregate.aggregate_r2(
            tmp_path / "occ.npz",
            bad,
            tmp_path / "r1.npz",
            tmp_path / "actual.npz",
            [],
            tmp_path / "out",
            execution_wrapper_sha="0" * 40,
        )
