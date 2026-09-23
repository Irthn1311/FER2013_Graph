from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
import pytest
import torch

from mpg_fer_v2_2.config import MPGConfig
from mpg_fer_v2_2.checkpoint import config_hash
from mpg_fer_v2_2.kaggle import resolve_resume_artifact
from mpg_fer_v2_2.train import (
    WarmupCosineScheduler,
    consistency_selected,
    should_early_stop,
    should_end_segment,
    update_early_stop_patience,
)


ROOT = Path(__file__).resolve().parents[1]


def test_v2_2_training_contract_and_scheduler() -> None:
    config = MPGConfig()
    assert (
        config.max_epochs,
        config.lr_decay_end_epoch,
        config.early_stop_monitor_start_epoch,
        config.early_stop_patience,
    ) == (120, 85, 85, 15)
    assert (
        config.pixel_dropout,
        config.motif_dropout,
        config.classifier_dropout,
        config.pixel_drop_path_max,
        config.motif_drop_path_max,
    ) == pytest.approx((0.10, 0.10, 0.25, 0.03, 0.05))
    assert (
        config.lambda_mi,
        config.consistency_probability,
        config.lambda_consistency,
        config.lambda_supcon,
        config.supcon_temperature,
    ) == pytest.approx((0.025, 0.50, 0.15, 0.05, 0.10))
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([parameter], lr=config.learning_rate)
    scheduler = WarmupCosineScheduler(
        optimizer, config.learning_rate, 5, 85, 120, 1e-6
    )
    assert scheduler.lr_for_epoch(1) == pytest.approx(6e-5)
    assert scheduler.lr_for_epoch(5) == pytest.approx(3e-4)
    assert scheduler.lr_for_epoch(120) == pytest.approx(1e-6)
    representative = {
        epoch: scheduler.lr_for_epoch(epoch)
        for epoch in (1, 5, 20, 40, 50, 62, 70, 80, 85, 86, 120)
    }
    assert representative == pytest.approx(
        {
            1: 0.00006000000000000000,
            5: 0.00030000000000000000,
            20: 0.00027480470703923050,
            40: 0.00017966600314141116,
            50: 0.00012133399685858881,
            62: 0.00005794545457817979,
            70: 0.00002619529296076947,
            80: 0.00000387260057971705,
            85: 0.000001,
            86: 0.000001,
            120: 0.000001,
        },
        rel=1e-8,
    )
    scheduler.step(17); state = scheduler.state_dict()
    clone = WarmupCosineScheduler(
        optimizer, config.learning_rate, 5, 85, 120, 1e-6
    )
    clone.load_state_dict(state)
    assert clone.state_dict() == state
    patience = 0
    for epoch in range(1, 85):
        patience = update_early_stop_patience(epoch, False, patience, config)
    assert patience == 0 and not should_early_stop(84, patience, config)
    for epoch in range(85, 99):
        patience = update_early_stop_patience(epoch, False, patience, config)
    assert patience == 14 and not should_early_stop(98, patience, config)
    patience = update_early_stop_patience(99, False, patience, config)
    assert patience == 15 and should_early_stop(99, patience, config)
    assert update_early_stop_patience(100, True, patience, config) == 0


def test_scheduler_is_independent_of_max_epochs_after_decay_horizon() -> None:
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([parameter], lr=3e-4)
    sched_100 = WarmupCosineScheduler(optimizer, 3e-4, 5, 85, 100, 1e-6)
    sched_120 = WarmupCosineScheduler(optimizer, 3e-4, 5, 85, 120, 1e-6)
    sched_200 = WarmupCosineScheduler(optimizer, 3e-4, 5, 85, 200, 1e-6)
    lr_100 = [sched_100.lr_for_epoch(epoch) for epoch in range(1, 86)]
    lr_120 = [sched_120.lr_for_epoch(epoch) for epoch in range(1, 86)]
    lr_200 = [sched_200.lr_for_epoch(epoch) for epoch in range(1, 86)]
    assert lr_100 == lr_120 == lr_200
    for sched, horizon in [(sched_100, 100), (sched_120, 120), (sched_200, 200)]:
        floor_lrs = [sched.lr_for_epoch(epoch) for epoch in range(85, horizon + 1)]
        assert len(floor_lrs) == horizon - 85 + 1
        assert all(lr == 1e-6 for lr in floor_lrs)


def test_scheduler_resume_produces_identical_future_lr_sequence() -> None:
    config = MPGConfig()
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([parameter], lr=config.learning_rate)
    continuous = WarmupCosineScheduler(
        optimizer, config.learning_rate, 5, 85, 120, 1e-6
    )
    values = {epoch: continuous.step(epoch) for epoch in range(1, 121)}
    state = {**continuous.state_dict(), "last_epoch": 57}
    clone = WarmupCosineScheduler(
        optimizer, config.learning_rate, 5, 85, 120, 1e-6
    )
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
    assert config_hash(
        replace(config, motif_topk_schedule=(16, 16, 16, 16, 24))
    ) != config_hash(config)


def test_consistency_schedule_is_resume_stable() -> None:
    first = [consistency_selected(42, 7, group, 0.5) for group in range(200)]
    second = [consistency_selected(42, 7, group, 0.5) for group in range(200)]
    assert first == second
    assert 0.4 < sum(first) / len(first) < 0.6


def test_resume_auto_never_discovers_v1_v2_v21_or_best_checkpoint(tmp_path) -> None:
    v1 = tmp_path / "mpg-fer-v1-resume"; v1.mkdir(); (v1 / "resume_latest.pt").write_bytes(b"x")
    v2 = tmp_path / "mpg-fer-v2-resume"; v2.mkdir(); (v2 / "resume_latest.pt").write_bytes(b"x")
    v21 = tmp_path / "mpg-fer-v2-1-resume"; v21.mkdir(); (v21 / "resume_latest.pt").write_bytes(b"x")
    generic = tmp_path / "other"; generic.mkdir(); (generic / "best_val_acc.pt").write_bytes(b"x")
    assert resolve_resume_artifact("auto", input_root=tmp_path) == (None, None)
    with pytest.raises(FileNotFoundError):
        resolve_resume_artifact("required", input_root=tmp_path)


def test_resume_auto_finds_nested_explicitly_named_v2_2_dataset(tmp_path) -> None:
    import hashlib
    nested = tmp_path / "datasets" / "owner" / "mpg-fer-v2-2-resume-run-a"
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
    nested = tmp_path / "mpg-fer-v2-2-resume-run-b"
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
    actual = json.loads((ROOT / "notebooks" / "MPG_FER_v2_2_Kaggle_T4.ipynb").read_text(encoding="utf-8"))
    assert actual == expected
    code = "\n".join("".join(cell["source"]) for cell in actual["cells"] if cell["cell_type"] == "code")
    assert 'RESUME_MODE = "fresh"' in code
    assert "resolve_resume_artifact(RESUME_MODE, RESUME_PATH)" in code
    assert "RESUMING EXISTING MPG-FER v2.2 RUN" in code
    assert "mpg_fer_v1" not in code
    assert "mpg_fer_v2_1" not in code
    assert "mpg-fer-v2-1-resume" not in code
    assert "Issue #95" in code
    assert "best_val_acc.pt" in code  # inference checkpoint name only
