from test_optimizer_slot_roundtrip import test_optimizer_slot_roundtrip


def test_adamw_checkpoint_slots(tmp_path):
    test_optimizer_slot_roundtrip(tmp_path)
