from d16.scripts.validate_ofix7_mid_limit_audit import tiny_optimizer_trajectories

def test_bounded_smoke_never_completes_epoch():
    result=tiny_optimizer_trajectories()
    assert result["steps_per_trajectory"]==2

