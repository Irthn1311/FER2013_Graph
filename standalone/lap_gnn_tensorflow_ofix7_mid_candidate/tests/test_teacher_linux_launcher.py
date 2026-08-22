from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "teacher_linux_launcher",
    ROOT / "tools/run_teacher_linux_seed42.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_driver_and_pci_parsing():
    assert MODULE.driver_major("470.256.02") == 470
    assert MODULE.driver_is_supported("470.256.02")
    assert not MODULE.driver_is_supported("449.99")
    assert MODULE.sysfs_pci_id("00000000:05:00.0") == "0000:05:00.0"


def test_training_command_preserves_locked_runtime():
    root = ROOT / "nonexistent_test_paths"
    args = SimpleNamespace(
        config=root / "config.yaml",
        fer_csv=root / "train.csv",
        prior_root=root / "priors",
        output_root=root / "output",
        graph_cache_dir=root / "cache",
        graph_workers=8,
        intra_op_threads=4,
        inter_op_threads=2,
        tf_data_prefetch=8,
    )
    command = MODULE.training_command(args)
    joined = " ".join(command)
    assert "--batch-size 16" in joined
    assert "--eval-batch-size 32" in joined
    assert "--graph-workers 8" in joined
    assert "--intra-op-threads 4" in joined
    assert "--inter-op-threads 2" in joined
    assert "--tf-data-prefetch 8" in joined
    assert "--mixed-precision" in command
    assert "--no-xla" in command
    assert "--no-resume" in command
