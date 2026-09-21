from __future__ import annotations

import copy
import random
import numpy as np
import torch
import torch.nn as nn
import pytest
from torch.utils.data import DataLoader, TensorDataset

from mpg_fer_v2_1.checkpoint import (
    atomic_save_resume, build_resume_bundle, find_latest_valid_snapshot,
    load_resume_bundle, restore_training_state, save_periodic_snapshot,
)
from mpg_fer_v2_1.config import MPGConfig
from mpg_fer_v2_1.ema import ModelEMA
from mpg_fer_v2_1.motif import SpatialMotifComposer, scheduled_motif_temperature
from mpg_fer_v2_1.train import (
    MOTIF_DIAGNOSTICS,
    WarmupCosineScheduler,
    _save_best_ema,
    train_one_epoch,
)
from mpg_fer_v2_1.utils import set_seed


class FakeScaler:
    def __init__(self, scale: float = 1024.0) -> None:
        self.scale = scale
    def state_dict(self) -> dict:
        return {"scale": self.scale}
    def load_state_dict(self, state: dict) -> None:
        self.scale = state["scale"]


class AuditTrainingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(3, 2)
        self.diagnostic_scalar = nn.Parameter(torch.tensor(0.2))

    def forward(self, images):
        logits = self.classifier(images) + self.diagnostic_scalar
        zero = self.diagnostic_scalar.square() * 0.01
        outputs = {
            "pixel_logits": logits,
            "motif_logits": logits,
            "supcon_embeddings": nn.functional.normalize(logits, dim=-1),
            "loss_diversity": zero,
            "loss_mi": zero,
        }
        outputs.update({name: self.diagnostic_scalar for name in MOTIF_DIAGNOSTICS})
        return logits, outputs


def _model() -> nn.Module:
    return nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Dropout(0.2), nn.Linear(8, 2))


def test_ema_update_and_serialization() -> None:
    model = _model()
    ema = ModelEMA(model, decay=0.5)
    before = copy.deepcopy(ema.module.state_dict())
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(2.0)
    ema.update(model)
    for name, value in ema.module.state_dict().items():
        if torch.is_floating_point(value):
            assert torch.allclose(value, before[name] * 0.5 + model.state_dict()[name] * 0.5)
    clone = ModelEMA(model, decay=0.9); clone.load_state_dict(ema.state_dict())
    assert clone.num_updates == 1 and clone.decay == 0.5
    for name, value in clone.module.state_dict().items():
        assert torch.equal(value, ema.module.state_dict()[name])


def test_ema_updates_once_per_successful_accumulation_group() -> None:
    model = AuditTrainingModel()
    ema = ModelEMA(model, decay=0.9)
    config = MPGConfig(
        num_classes=2, use_amp=False, gradient_accumulation_steps=2,
        consistency_probability=0.0,
    )
    loader = DataLoader(
        TensorDataset(torch.randn(8, 3), torch.arange(8) % 2), batch_size=2
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    _stats, steps = train_one_epoch(
        model, loader, optimizer, "cpu", None, config, criterion,
        ema=ema, epoch=1, global_optimizer_step=0,
    )
    assert steps == 2
    assert ema.num_updates == 2


def test_best_inference_checkpoint_is_explicitly_ema(tmp_path) -> None:
    model = _model()
    ema = ModelEMA(model, decay=0.9)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    ema.update(model)
    path = tmp_path / "best_val_acc.pt"
    _save_best_ema(
        path, ema, epoch=3,
        metrics={"raw": {"accuracy": 0.5}, "tta": {"accuracy": 0.6}},
        config=MPGConfig(), source_hash="source",
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["weights_type"] == "EMA"
    assert checkpoint["checkpoint_type"] == "EMA_INFERENCE_ONLY"
    for name, value in checkpoint["model_state_dict"].items():
        assert torch.equal(value, ema.module.state_dict()[name])


class TemperatureCheckpointModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.motif_composer = SpatialMotifComposer(
            d_pixel=2, num_motifs=4, d_type=2, d_motif=6
        )
        self.linear = nn.Linear(2, 2)


def test_best_checkpoint_preserves_its_epoch_temperature_not_final_tau(tmp_path) -> None:
    model = TemperatureCheckpointModel()
    ema = ModelEMA(model, decay=0.9)
    best_epoch = 10
    model.motif_composer.set_epoch_temperature(best_epoch)
    ema.update(model)
    path = tmp_path / "best_val_acc.pt"
    _save_best_ema(
        path, ema, epoch=best_epoch,
        metrics={"raw": {"accuracy": 0.5}, "tta": {"accuracy": 0.6}},
        config=MPGConfig(), source_hash="source",
    )
    model.motif_composer.set_epoch_temperature(80)
    ema.module.motif_composer.set_epoch_temperature(80)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    clone = TemperatureCheckpointModel()
    clone.load_state_dict(checkpoint["model_state_dict"], strict=True)
    expected = scheduled_motif_temperature(best_epoch)
    assert checkpoint["scheduled_tau"] == pytest.approx(expected)
    assert float(clone.motif_composer.temperature) == pytest.approx(expected)
    assert float(clone.motif_composer.temperature) != pytest.approx(0.30)


def _objects(config: MPGConfig):
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = WarmupCosineScheduler(
        optimizer, config.learning_rate, config.warmup_epochs,
        config.lr_decay_end_epoch, config.max_epochs, config.min_learning_rate,
    )
    ema = ModelEMA(model, 0.9)
    scaler = FakeScaler()
    generator = torch.Generator().manual_seed(config.seed)
    return model, optimizer, scheduler, ema, scaler, generator


def test_full_resume_round_trip_restores_all_declared_state(tmp_path) -> None:
    config = MPGConfig(
        max_epochs=4, warmup_epochs=1, lr_decay_end_epoch=4,
        early_stop_monitor_start_epoch=4,
    )
    model, optimizer, scheduler, ema, scaler, generator = _objects(config)
    scheduler.step(1)
    loss = model(torch.randn(3, 3)).sum(); loss.backward(); optimizer.step(); ema.update(model)
    bundle = build_resume_bundle(
        config=config, run_id="run-a", source_hash="source-a", completed_epoch=1,
        global_optimizer_step=1, model=model, ema=ema, optimizer=optimizer,
        scheduler=scheduler, scaler=scaler,
        best_comparator_state={"accuracy": 0.6, "macro_f1": 0.5, "loss": 1.0},
        best_epoch=1, best_metrics={"tta": {"accuracy": 0.6}},
        early_stop_counter=3, history=[{"epoch": 1}], loader_generator=generator,
        consistency_state={"selected_groups": [1]},
    )
    path, digest = atomic_save_resume(bundle, tmp_path)
    loaded = load_resume_bundle(path, expected_sha256=digest, config=config, run_id="run-a", source_hash="source-a")
    assert loaded["next_epoch"] == 2
    assert loaded["best_epoch"] == 1 and loaded["early_stop_counter"] == 3
    assert loaded["history"] == [{"epoch": 1}]
    new = _objects(config)
    restore_training_state(loaded, model=new[0], optimizer=new[1], scheduler=new[2], ema=new[3], scaler=new[4], loader_generator=new[5])
    assert new[2].state_dict() == scheduler.state_dict()
    assert new[4].scale == scaler.scale
    assert new[1].state_dict()["state"].keys() == optimizer.state_dict()["state"].keys()
    for name, value in new[0].state_dict().items():
        assert torch.equal(value, model.state_dict()[name])


def _run_epochs(objects, start: int, end: int, history: list[dict]):
    model, optimizer, scheduler, ema, _, generator = objects
    for epoch in range(start, end + 1):
        scheduler.step(epoch); model.train()
        order = torch.randperm(12, generator=generator)
        x = torch.arange(36, dtype=torch.float32).view(12, 3) / 36
        y = torch.arange(12) % 2
        for index in order.split(4):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[index])
            loss = nn.functional.cross_entropy(logits, y[index])
            loss.backward(); optimizer.step(); ema.update(model)
        history.append({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]})


def _assert_nested_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            _assert_nested_equal(first, second)
    else:
        assert left == right


def test_continuous_four_epochs_matches_two_resume_two(tmp_path) -> None:
    config = MPGConfig(
        max_epochs=4, warmup_epochs=1, lr_decay_end_epoch=4,
        early_stop_monitor_start_epoch=4,
    )
    set_seed(123); continuous = _objects(config); continuous_history = []
    _run_epochs(continuous, 1, 4, continuous_history)

    set_seed(123); interrupted = _objects(config); interrupted_history = []
    _run_epochs(interrupted, 1, 2, interrupted_history)
    bundle = build_resume_bundle(
        config=config, run_id="same-run", source_hash="same-source", completed_epoch=2,
        global_optimizer_step=6, model=interrupted[0], ema=interrupted[3],
        optimizer=interrupted[1], scheduler=interrupted[2], scaler=interrupted[4],
        best_comparator_state={"accuracy": 0.4}, best_epoch=2,
        best_metrics={"tta": {"accuracy": 0.4}}, early_stop_counter=1,
        history=interrupted_history, loader_generator=interrupted[5],
        consistency_state={"selected_groups": []},
    )
    path, digest = atomic_save_resume(bundle, tmp_path)
    set_seed(999); resumed = _objects(config)
    loaded = load_resume_bundle(path, expected_sha256=digest, config=config, run_id="same-run", source_hash="same-source")
    restore_training_state(loaded, model=resumed[0], optimizer=resumed[1], scheduler=resumed[2], ema=resumed[3], scaler=resumed[4], loader_generator=resumed[5])
    resumed_history = list(loaded["history"])
    _run_epochs(resumed, 3, 4, resumed_history)

    for name, value in continuous[0].state_dict().items():
        assert torch.equal(value, resumed[0].state_dict()[name])
    for name, value in continuous[3].module.state_dict().items():
        assert torch.equal(value, resumed[3].module.state_dict()[name])
    _assert_nested_equal(continuous[1].state_dict(), resumed[1].state_dict())
    assert continuous[2].state_dict() == resumed[2].state_dict()
    assert continuous_history == resumed_history


def test_rng_states_restore_for_python_numpy_torch_and_loader(tmp_path) -> None:
    config = MPGConfig(
        max_epochs=4, warmup_epochs=1, lr_decay_end_epoch=4,
        early_stop_monitor_start_epoch=4,
    )
    set_seed(71); objects = _objects(config)
    bundle = build_resume_bundle(
        config=config, run_id="rng", source_hash="source", completed_epoch=1,
        global_optimizer_step=0, model=objects[0], ema=objects[3], optimizer=objects[1],
        scheduler=objects[2], scaler=objects[4], best_comparator_state=None,
        best_epoch=None, best_metrics=None, early_stop_counter=0, history=[],
        loader_generator=objects[5], consistency_state={},
    )
    expected = (random.random(), np.random.rand(), torch.rand(1), torch.rand(1, generator=objects[5]))
    path, digest = atomic_save_resume(bundle, tmp_path)
    random.random(); np.random.rand(); torch.rand(1); torch.rand(1, generator=objects[5])
    loaded = load_resume_bundle(path, expected_sha256=digest, config=config, run_id="rng", source_hash="source")
    restore_training_state(loaded, model=objects[0], ema=objects[3], optimizer=objects[1], scheduler=objects[2], scaler=objects[4], loader_generator=objects[5])
    actual = (random.random(), np.random.rand(), torch.rand(1), torch.rand(1, generator=objects[5]))
    assert actual[0] == expected[0] and actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2]) and torch.equal(actual[3], expected[3])


def test_periodic_snapshots_have_independent_hashes_and_report_fallback(tmp_path) -> None:
    config = MPGConfig(
        max_epochs=4, warmup_epochs=1, lr_decay_end_epoch=4,
        early_stop_monitor_start_epoch=4,
    )
    objects = _objects(config)
    for epoch in (10, 20, 30):
        bundle = build_resume_bundle(
            config=config, run_id="fallback", source_hash="source",
            completed_epoch=epoch, global_optimizer_step=0,
            model=objects[0], ema=objects[3], optimizer=objects[1],
            scheduler=objects[2], scaler=objects[4],
            best_comparator_state=None, best_epoch=None, best_metrics=None,
            early_stop_counter=0, history=[], loader_generator=objects[5],
            consistency_state={},
        )
        latest, _ = atomic_save_resume(bundle, tmp_path)
        save_periodic_snapshot(latest, epoch, interval=10, keep=2)
    snapshots = sorted(tmp_path.glob("resume_epoch_*.pt"))
    assert [path.name for path in snapshots] == ["resume_epoch_020.pt", "resume_epoch_030.pt"]
    assert all(path.with_suffix(".json").is_file() for path in snapshots)
    fallback = find_latest_valid_snapshot(tmp_path)
    assert fallback is not None and fallback[0].name == "resume_epoch_030.pt"
    snapshots[-1].write_bytes(b"corrupt")
    fallback = find_latest_valid_snapshot(tmp_path)
    assert fallback is not None and fallback[0].name == "resume_epoch_020.pt"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_real_cuda_grad_scaler_state_round_trip() -> None:
    model = nn.Linear(3, 2).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=True):
        loss = model(torch.randn(4, 3, device="cuda")).square().mean()
    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
    state = scaler.state_dict()
    clone = torch.amp.GradScaler("cuda")
    clone.load_state_dict(state)
    assert clone.state_dict() == state
