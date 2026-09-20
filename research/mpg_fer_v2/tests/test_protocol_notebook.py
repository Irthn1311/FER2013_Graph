from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
import pytest
import torch

from mpg_fer_v2.config import MPGConfig
from mpg_fer_v2.checkpoint import config_hash
from mpg_fer_v2.kaggle import resolve_resume_artifact
from mpg_fer_v2.train import (
    WarmupCosineScheduler,
    consistency_selected,
    should_early_stop,
    should_end_segment,
)


ROOT = Path(__file__).resolve().parents[1]


def test_v2_training_contract_and_scheduler() -> None:
    config = MPGConfig()
    assert (config.max_epochs, config.min_epochs, config.early_stop_patience) == (120, 50, 20)
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([parameter], lr=config.learning_rate)
    scheduler = WarmupCosineScheduler(optimizer, config.learning_rate, 5, 120, 1e-6)
    assert scheduler.lr_for_epoch(1) == pytest.approx(6e-5)
    assert scheduler.lr_for_epoch(5) == pytest.approx(3e-4)
    assert scheduler.lr_for_epoch(120) == pytest.approx(1e-6)
    representative = {
        epoch: scheduler.lr_for_epoch(epoch)
        for epoch in (1, 2, 3, 4, 5, 6, 40, 80, 100, 120)
    }
    assert representative == pytest.approx(
        {
            1: 0.00006,
            2: 0.00012,
            3: 0.00018,
            4: 0.00024,
            5: 0.00030,
            6: 0.0002999442187486556,
            40: 0.00023671370815617259,
            80: 0.00008172027685919274,
            100: 0.000022764299020299957,
            120: 0.000001,
        },
        rel=1e-8,
    )
    scheduler.step(17); state = scheduler.state_dict()
    clone = WarmupCosineScheduler(optimizer, config.learning_rate, 5, 120, 1e-6)
    clone.load_state_dict(state)
    assert clone.state_dict() == state
    assert not should_early_stop(49, 20, config)
    assert should_early_stop(50, 20, config)


def test_scheduler_resume_produces_identical_future_lr_sequence() -> None:
    config = MPGConfig()
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([parameter], lr=config.learning_rate)
    continuous = WarmupCosineScheduler(optimizer, config.learning_rate, 5, 120, 1e-6)
    values = {epoch: continuous.step(epoch) for epoch in range(1, 121)}
    state = {**continuous.state_dict(), "last_epoch": 57}
    clone = WarmupCosineScheduler(optimizer, config.learning_rate, 5, 120, 1e-6)
    clone.load_state_dict(state)
    assert [clone.step(epoch) for epoch in range(58, 121)] == [values[epoch] for epoch in range(58, 121)]


def test_wallclock_guard_includes_fifteen_minute_safety_margin() -> None:
    config = MPGConfig(segment_soft_limit_hours=10.5, segment_safety_margin_minutes=15)
    assert not should_end_segment(9.5 * 3600, 20 * 60, config)
    assert should_end_segment(10.0 * 3600, 20 * 60, config)


def test_config_hash_ignores_only_declared_runtime_fields() -> None:
    config = MPGConfig()
    runtime_changed = replace(
        config, num_workers=7, output_dir="elsewhere", resume_path="input/resume.pt",
        segment_number=9, segment_soft_limit_hours=9.5,
        segment_safety_margin_minutes=12.0, run_id="same-run",
    )
    assert config_hash(runtime_changed) == config_hash(config)
    assert config_hash(replace(config, lambda_mi=0.06)) != config_hash(config)
    assert config_hash(replace(config, motif_window_sizes=(6, 12, 18))) != config_hash(config)


def test_consistency_schedule_is_resume_stable() -> None:
    first = [consistency_selected(42, 7, group, 0.2) for group in range(200)]
    second = [consistency_selected(42, 7, group, 0.2) for group in range(200)]
    assert first == second
    assert 0.1 < sum(first) / len(first) < 0.3


def test_resume_auto_never_discovers_v1_or_best_checkpoint(tmp_path) -> None:
    v1 = tmp_path / "mpg-fer-v1-resume"; v1.mkdir(); (v1 / "resume_latest.pt").write_bytes(b"x")
    generic = tmp_path / "other"; generic.mkdir(); (generic / "best_val_acc.pt").write_bytes(b"x")
    assert resolve_resume_artifact("auto", input_root=tmp_path) == (None, None)
    with pytest.raises(FileNotFoundError):
        resolve_resume_artifact("required", input_root=tmp_path)


def test_resume_auto_finds_nested_explicitly_named_v2_dataset(tmp_path) -> None:
    import hashlib
    nested = tmp_path / "datasets" / "owner" / "mpg-fer-v2-resume-run-a"
    nested.mkdir(parents=True)
    checkpoint = nested / "resume_latest.pt"
    checkpoint.write_bytes(b"valid-resume")
    digest = hashlib.sha256(b"valid-resume").hexdigest()
    (nested / "resume_latest.json").write_text(
        json.dumps({"checkpoint_sha256": digest}), encoding="utf-8"
    )
    assert resolve_resume_artifact("auto", input_root=tmp_path) == (checkpoint, digest)


def test_corrupt_latest_reports_but_does_not_auto_load_valid_snapshot(tmp_path) -> None:
    import hashlib
    nested = tmp_path / "mpg-fer-v2-resume-run-b"
    nested.mkdir()
    latest = nested / "resume_latest.pt"
    latest.write_bytes(b"corrupt")
    (nested / "resume_latest.json").write_text(
        json.dumps({"checkpoint_sha256": hashlib.sha256(b"expected").hexdigest()}),
        encoding="utf-8",
    )
    snapshot = nested / "resume_epoch_020.pt"
    snapshot.write_bytes(b"valid-fallback")
    snapshot_digest = hashlib.sha256(b"valid-fallback").hexdigest()
    snapshot.with_suffix(".json").write_text(
        json.dumps({"checkpoint_sha256": snapshot_digest}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="refusing automatic fallback.*resume_epoch_020"):
        resolve_resume_artifact("auto", input_root=tmp_path)


def test_generated_notebook_matches_all_reviewed_sources() -> None:
    script = ROOT / "tools" / "sync_notebook.py"
    spec = importlib.util.spec_from_file_location("sync_v2", script)
    module = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(module)
    expected = module.build_notebook()
    actual = json.loads((ROOT / "notebooks" / "MPG_FER_v2_Kaggle_T4.ipynb").read_text(encoding="utf-8"))
    assert actual == expected
    code = "\n".join("".join(cell["source"]) for cell in actual["cells"] if cell["cell_type"] == "code")
    assert 'RESUME_MODE = "fresh"' in code
    assert "resolve_resume_artifact(RESUME_MODE, RESUME_PATH)" in code
    assert "RESUMING EXISTING MPG-FER v2 RUN" in code
    assert "mpg_fer_v1" not in code
    assert "best_val_acc.pt" in code  # inference checkpoint name only
