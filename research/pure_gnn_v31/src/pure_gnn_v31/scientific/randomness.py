"""Reproducibility and paired deterministic seed management for Pure-GNN scientific runs."""

import os
import random
import numpy as np
import tensorflow as tf


def set_scientific_seed(seed: int) -> None:
    """Configures deterministic seeds across Python, NumPy, and TensorFlow.

    Explicit seed is required (NO default seed).

    Guarantees:
    - Sets random.seed(seed)
    - Sets np.random.seed(seed)
    - Sets tf.random.set_seed(seed)
    - Sets TF_DETERMINISTIC_OPS=1 in environment
    - Invokes tf.config.experimental.enable_op_determinism() when available

    Note on PYTHONHASHSEED:
    Setting PYTHONHASHSEED after Python interpreter startup does NOT retroactively
    re-seed Python's built-in string/bytes hash randomization for the current process.
    True hash determinism requires setting PYTHONHASHSEED prior to python startup.
    """
    if seed is None or not isinstance(seed, int):
        raise ValueError(f"Explicit integer seed is required, got: {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    if hasattr(tf.config.experimental, "enable_op_determinism"):
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def derive_paired_seed(base_seed: int, salt: int) -> int:
    """Derives a deterministic paired seed for fair cross-condition comparisons."""
    if base_seed is None or not isinstance(base_seed, int):
        raise ValueError(f"Explicit integer base_seed is required, got: {base_seed}")
    return (base_seed * 10007 + salt) % (2**31 - 1)
