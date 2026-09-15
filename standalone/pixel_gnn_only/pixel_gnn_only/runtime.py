"""Runtime-only CPU sizing and explicit GPU selection for Kaggle."""

import math
import os
from pathlib import Path


def available_cpu_count():
    limits = [os.cpu_count() or 1]
    if hasattr(os, "sched_getaffinity"):
        limits.append(len(os.sched_getaffinity(0)))
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            limits.append(max(1, math.ceil(int(quota) / int(period))))
    except (OSError, ValueError):
        try:
            quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if quota > 0:
                limits.append(max(1, math.ceil(quota / period)))
        except (OSError, ValueError):
            pass
    return max(1, min(limits))


def select_gpu_count(requested, available):
    if str(requested) == "auto":
        count = min(2, available)
    else:
        count = int(requested)
    if count not in (1, 2) or count > available:
        raise RuntimeError(f"Requested {requested} GPUs; found {available}. Select Kaggle GPU T4 x2 or use --gpus 1.")
    return count
