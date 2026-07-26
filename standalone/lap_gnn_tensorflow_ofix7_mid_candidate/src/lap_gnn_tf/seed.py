"""Deterministic seed controls."""

from __future__ import annotations

import os
import random

import numpy as np
import tensorflow as tf


def seed_everything(seed: int, deterministic_ops: bool = True) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if deterministic_ops:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

