"""Checkpoint policy helpers."""

from pathlib import Path

import torch


def load_checkpoint(path, map_location="cpu"):
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def assert_macro_alias(checkpoint_dir) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    best = checkpoint_dir / "best.pt"
    macro = checkpoint_dir / "best_val_macro_f1.pt"
    if not best.is_file() or not macro.is_file() or best.read_bytes() != macro.read_bytes():
        raise ValueError("best.pt is not the canonical byte-identical macro-F1 checkpoint")
