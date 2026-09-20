from __future__ import annotations

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mpg_fer_v2.checkpoint import (
    atomic_save_resume,
    build_resume_bundle,
    load_resume_bundle,
    restore_training_state,
)
from mpg_fer_v2.config import MPGConfig
from mpg_fer_v2.data import EpochRandomSampler
from mpg_fer_v2.ema import ModelEMA
from mpg_fer_v2.losses import symmetric_js_divergence
from mpg_fer_v2.model import DropPath
from mpg_fer_v2.motif import SpatialMotifComposer
from mpg_fer_v2.train import WarmupCosineScheduler, consistency_selected, is_better_checkpoint
from mpg_fer_v2.utils import set_seed


class TinyActualV2(nn.Module):
    """Small harness using the actual v2 motif, temperature, MI and DropPath."""

    def __init__(self) -> None:
        super().__init__()
        self.composer = SpatialMotifComposer(
            d_pixel=4, num_motifs=6, d_type=3, d_motif=12,
            tau_min=0.15, tau_max=1.50, tau_init=0.70,
        )
        self.drop_path = DropPath(0.20)
        self.classifier = nn.Linear(12, 2)

    def forward(self, features: torch.Tensor):
        motif, _assignments, diagnostics = self.composer(features)
        pooled = self.drop_path(motif.mean(dim=1))
        return self.classifier(pooled), diagnostics


def _objects(config: MPGConfig):
    model = TinyActualV2()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = WarmupCosineScheduler(
        optimizer, config.learning_rate, config.warmup_epochs,
        config.max_epochs, config.min_learning_rate,
    )
    ema = ModelEMA(model, decay=0.9)
    generator = torch.Generator().manual_seed(config.seed)
    feature_generator = torch.Generator().manual_seed(812)
    features = torch.randn(6, 2304, 4, generator=feature_generator)
    labels = torch.arange(6) % 2
    dataset = TensorDataset(features, labels, torch.arange(6))
    sampler = EpochRandomSampler(dataset, seed=config.seed)
    return model, optimizer, scheduler, ema, generator, dataset, sampler


def _initial_state() -> dict:
    return {
        "global_step": 0,
        "best_comparator": None,
        "best_epoch": None,
        "best_metrics": None,
        "patience": 0,
        "history": [],
        "sample_order": [],
        "consistency_decisions": [],
    }


def _run(objects, config: MPGConfig, start: int, end: int, state: dict) -> None:
    model, optimizer, scheduler, ema, generator, dataset, sampler = objects
    loader = DataLoader(
        dataset, batch_size=2, sampler=sampler, num_workers=0, generator=generator
    )
    criterion = nn.CrossEntropyLoss()
    for epoch in range(start, end + 1):
        sampler.set_epoch(epoch)
        lr = scheduler.step(epoch)
        model.train(); optimizer.zero_grad(set_to_none=True)
        epoch_order = []
        decisions = []
        batches = len(loader)
        for batch_index, (features, labels, indices) in enumerate(loader):
            group = batch_index // config.gradient_accumulation_steps
            group_start = group * config.gradient_accumulation_steps
            group_size = min(config.gradient_accumulation_steps, batches - group_start)
            selected = consistency_selected(
                config.seed, epoch, group, config.consistency_probability
            )
            epoch_order.extend(indices.tolist())
            decisions.append(selected)
            logits, diagnostics = model(features)
            loss = criterion(logits, labels) + config.lambda_mi * diagnostics["loss_mi"]
            if selected:
                flipped = features.reshape(-1, 48, 48, 4).flip(2).reshape_as(features)
                flipped_logits, _ = model(flipped)
                loss = loss + config.lambda_consistency * symmetric_js_divergence(
                    logits, flipped_logits
                )
            (loss / group_size).backward()
            end_group = (
                (batch_index + 1) % config.gradient_accumulation_steps == 0
                or batch_index + 1 == batches
            )
            if end_group:
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                ema.update(model); state["global_step"] += 1

        ema.module.eval()
        with torch.no_grad():
            validation_logits, validation_diags = ema.module(dataset.tensors[0])
            accuracy = float(
                (validation_logits.argmax(-1) == dataset.tensors[1]).float().mean()
            )
            candidate = {
                "accuracy": accuracy,
                "macro_f1": accuracy,
                "loss": float(criterion(validation_logits, dataset.tensors[1])),
            }
            tau = float(validation_diags["tau"])
            mi = float(validation_diags["loss_mi"])
        if is_better_checkpoint(candidate, state["best_comparator"]):
            state["best_comparator"] = candidate
            state["best_epoch"] = epoch
            state["best_metrics"] = {"tta": candidate}
            state["patience"] = 0
        else:
            state["patience"] += 1
        state["sample_order"].append(epoch_order)
        state["consistency_decisions"].append(decisions)
        state["history"].append(
            {
                "epoch": epoch, "lr": lr, "tau": tau, "mi": mi,
                "global_step": state["global_step"], "best_epoch": state["best_epoch"],
                "patience": state["patience"],
            }
        )


def _assert_nested_close(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_close(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            _assert_nested_close(first, second)
    else:
        assert left == right


def _assert_optimizer_numerically_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        if torch.is_floating_point(left):
            assert torch.allclose(left, right, atol=1e-9, rtol=1e-7)
        else:
            assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_optimizer_numerically_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            _assert_optimizer_numerically_equal(first, second)
    else:
        assert left == right


def test_actual_v2_stochastic_trajectory_is_resume_invariant(tmp_path) -> None:
    config = MPGConfig(
        max_epochs=4, warmup_epochs=1, gradient_accumulation_steps=2,
        consistency_probability=0.20,
    )
    set_seed(404); continuous = _objects(config); continuous_state = _initial_state()
    _run(continuous, config, 1, 4, continuous_state)

    set_seed(404); interrupted = _objects(config); interrupted_state = _initial_state()
    _run(interrupted, config, 1, 2, interrupted_state)
    bundle = build_resume_bundle(
        config=config, run_id="actual-v2", source_hash="actual-source",
        completed_epoch=2, global_optimizer_step=interrupted_state["global_step"],
        model=interrupted[0], ema=interrupted[3], optimizer=interrupted[1],
        scheduler=interrupted[2], scaler=None,
        best_comparator_state=interrupted_state["best_comparator"],
        best_epoch=interrupted_state["best_epoch"],
        best_metrics=interrupted_state["best_metrics"],
        early_stop_counter=interrupted_state["patience"],
        history=interrupted_state["history"], loader_generator=interrupted[4],
        consistency_state={"decisions": interrupted_state["consistency_decisions"]},
        sampler_state=interrupted[6].state_dict(),
        augmentation_state={"scheme": "covered_by_multiworker_test", "epoch": 2},
    )
    path, digest = atomic_save_resume(bundle, tmp_path)

    set_seed(999); resumed = _objects(config)
    loaded = load_resume_bundle(
        path, expected_sha256=digest, config=config,
        run_id="actual-v2", source_hash="actual-source",
    )
    restore_training_state(
        loaded, model=resumed[0], ema=resumed[3], optimizer=resumed[1],
        scheduler=resumed[2], scaler=None, loader_generator=resumed[4],
        sampler=resumed[6],
    )
    resumed_state = {
        "global_step": loaded["global_optimizer_step"],
        "best_comparator": copy.deepcopy(loaded["best_comparator_state"]),
        "best_epoch": loaded["best_epoch"],
        "best_metrics": copy.deepcopy(loaded["best_metrics"]),
        "patience": loaded["early_stop_counter"],
        "history": copy.deepcopy(loaded["history"]),
        "sample_order": copy.deepcopy(continuous_state["sample_order"][:2]),
        "consistency_decisions": copy.deepcopy(
            loaded["consistency_state"]["decisions"]
        ),
    }
    _run(resumed, config, 3, 4, resumed_state)

    _assert_nested_close(continuous[0].state_dict(), resumed[0].state_dict())
    _assert_nested_close(continuous[3].state_dict(), resumed[3].state_dict())
    _assert_optimizer_numerically_equal(
        continuous[1].state_dict(), resumed[1].state_dict()
    )
    assert continuous[2].state_dict() == resumed[2].state_dict()
    assert continuous_state == resumed_state
    assert continuous[0].composer.raw_tau.equal(resumed[0].composer.raw_tau)
    assert continuous[3].num_updates == resumed[3].num_updates
    assert any(any(epoch) for epoch in continuous_state["consistency_decisions"])
