from d16.scripts.prepare_ofix7_mid_final_replication import REGISTERED_SEEDS


def test_seed_registration():
    assert REGISTERED_SEEDS == [42, 1009, 1337, 777, 3407]
