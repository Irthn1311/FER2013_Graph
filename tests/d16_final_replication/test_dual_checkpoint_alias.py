import torch
from d16.training.train_d16 import _atomic_copy_checkpoint, _atomic_torch_save, canonical_model_state_hash


def test_dual_checkpoint_alias(tmp_path):
    source = tmp_path / "best.pt"; alias = tmp_path / "best_val_macro_f1.pt"
    _atomic_torch_save({"model_state_dict": {"w": torch.arange(4)}}, source)
    _atomic_copy_checkpoint(source, alias)
    assert source.read_bytes() == alias.read_bytes()
    assert canonical_model_state_hash(torch.load(source, weights_only=False)) == canonical_model_state_hash(torch.load(alias, weights_only=False))
