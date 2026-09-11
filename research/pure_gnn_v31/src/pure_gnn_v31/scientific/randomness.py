"""Reproducibility and paired deterministic seed management for Pure-GNN scientific runs."""

import os
import random
import numpy as np
import tensorflow as tf


def set_scientific_seed(seed: int = 42) -> None:
    """Configures deterministic seeds across Python, NumPy, and TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def derive_paired_seed(base_seed: int, salt: int) -> int:
    """Derives a deterministic paired seed for fair cross-condition comparisons."""
    return (base_seed * 10007 + salt) % (2**31 - 1)
