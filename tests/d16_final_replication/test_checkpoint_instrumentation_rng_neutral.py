from d16.scripts.validate_ofix7_mid_final_replication import rng_neutral_checkpoint_test


def test_checkpoint_instrumentation_rng_neutral():
    passed, details = rng_neutral_checkpoint_test()
    assert passed, details
