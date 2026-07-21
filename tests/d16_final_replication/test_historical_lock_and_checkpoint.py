from d16.scripts.prepare_ofix7_mid_final_replication import sha256_file, verify_lock


def test_historical_lock_and_checkpoint():
    lock, paths = verify_lock()
    assert lock["run_id"] == "d16r_a5b_ofix7_prior_drop_mid_seed42"
    assert sha256_file(paths["best"]) == lock["checkpoint_sha256"]
