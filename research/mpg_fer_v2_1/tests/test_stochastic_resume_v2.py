from __future__ import annotations

import copy
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
from torch.utils.data import DataLoader, Dataset

from mpg_fer_v2_1.checkpoint import (
    atomic_save_resume,
    build_resume_bundle,
    load_resume_bundle,
    restore_training_state,
)
from mpg_fer_v2_1.config import MPGConfig
from mpg_fer_v2_1.data import EpochRandomSampler
from mpg_fer_v2_1.ema import ModelEMA
from mpg_fer_v2_1.losses import supervised_contrastive_loss, symmetric_js_divergence
from mpg_fer_v2_1.model import DropPath
from mpg_fer_v2_1.motif import SpatialMotifComposer
from mpg_fer_v2_1.train import (
    WarmupCosineScheduler,
    consistency_selected,
    is_better_checkpoint,
    update_early_stop_patience,
)
from mpg_fer_v2_1.utils import set_seed


class EpochAugmentedFeatureDataset(Dataset):
    """Small stateless epoch/sample augmentation fixture, safe with workers."""

    def __init__(self, seed: int = 812) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.features = torch.randn(4, 2304, 2, generator=generator)
        self.labels = torch.tensor([0, 0, 1, 1])
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.labels)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def state_dict(self) -> dict:
        return {"scheme": "epoch_sample_noise_v1", "seed": self.seed, "epoch": self.epoch}

    def load_state_dict(self, state: dict) -> None:
        if state["scheme"] != "epoch_sample_noise_v1" or state["seed"] != self.seed:
            raise RuntimeError("augmentation state mismatch")
        self.epoch = int(state["epoch"])

    def __getitem__(self, index: int):
        generator = torch.Generator().manual_seed(
            self.seed + self.epoch * 1_000_003 + int(index) * 9_176
        )
        noise = torch.rand(self.features[index].shape, generator=generator) * 0.01
        return self.features[index] + noise, self.labels[index], index


class TinyActualV21(nn.Module):
    """Compact harness using actual v2.1 motif, tau, MI, DropPath and SupCon."""

    def __init__(self) -> None:
        super().__init__()
        self.motif_composer = SpatialMotifComposer(
            d_pixel=2, num_motifs=4, d_type=2, d_motif=6,
            tau_start=0.70, tau_final=0.30, tau_anneal_end_epoch=35,
        )
        self.drop_path = DropPath(0.20)
        self.supcon_head = nn.Sequential(nn.Linear(6, 4), nn.LayerNorm(4))
        self.classifier = nn.Linear(6, 2)

    def set_epoch_temperature(self, epoch: int) -> float:
        return self.motif_composer.set_epoch_temperature(epoch)

    def forward(self, features: torch.Tensor):
        motif, _assignments, diagnostics = self.motif_composer(features)
        pooled = self.drop_path(motif.mean(dim=1))
        diagnostics["supcon_embeddings"] = F.normalize(self.supcon_head(pooled), dim=-1)
        return self.classifier(pooled), diagnostics


def _objects(config: MPGConfig):
    model = TinyActualV21()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = WarmupCosineScheduler(
        optimizer, config.learning_rate, config.warmup_epochs,
        config.lr_decay_end_epoch, config.max_epochs, config.min_learning_rate,
    )
    ema = ModelEMA(model, decay=0.9)
    generator = torch.Generator().manual_seed(config.seed)
    dataset = EpochAugmentedFeatureDataset()
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


def _run(objects, config: MPGConfig, epochs: Iterable[int], state: dict) -> None:
    model, optimizer, scheduler, ema, generator, dataset, sampler = objects
    loader = DataLoader(
        dataset, batch_size=4, sampler=sampler, num_workers=2,
        persistent_workers=False, generator=generator,
    )
    criterion = nn.CrossEntropyLoss()
    for epoch in epochs:
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        lr = scheduler.step(epoch)
        model.set_epoch_temperature(epoch)
        ema.module.set_epoch_temperature(epoch)
        model.train(); optimizer.zero_grad(set_to_none=True)
        epoch_order = []
        decisions = []
        for group, (features, labels, indices) in enumerate(loader):
            selected = consistency_selected(
                config.seed, epoch, group, config.consistency_probability
            )
            epoch_order.extend(indices.tolist())
            decisions.append(selected)
            logits, diagnostics = model(features)
            supcon, _ = supervised_contrastive_loss(
                diagnostics["supcon_embeddings"], labels, config.supcon_temperature
            )
            loss = (
                criterion(logits, labels)
                + config.lambda_mi * diagnostics["loss_mi"]
                + config.lambda_supcon * supcon
            )
            if selected:
                flipped = features.reshape(-1, 48, 48, 2).flip(2).reshape_as(features)
                flipped_logits, _ = model(flipped)
                loss = loss + config.lambda_consistency * symmetric_js_divergence(
                    logits, flipped_logits
                )
            loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
            ema.update(model); state["global_step"] += 1

        ema.module.set_epoch_temperature(epoch)
        ema.module.eval()
        with torch.no_grad():
            validation_features = torch.stack([dataset[index][0] for index in range(4)])
            validation_labels = dataset.labels
            validation_logits, validation_diags = ema.module(validation_features)
            accuracy = float(
                (validation_logits.argmax(-1) == validation_labels).float().mean()
            )
            candidate = {
                "accuracy": accuracy,
                "macro_f1": accuracy,
                "loss": float(criterion(validation_logits, validation_labels)),
            }
            tau = float(validation_diags["tau"])
            mi = float(validation_diags["loss_mi"])
        improved = is_better_checkpoint(candidate, state["best_comparator"])
        if improved:
            state["best_comparator"] = candidate
            state["best_epoch"] = epoch
            state["best_metrics"] = {"tta": candidate}
        state["patience"] = update_early_stop_patience(
            epoch, improved, state["patience"], config
        )
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


def test_actual_v2_1_stochastic_multiworker_trajectory_is_resume_invariant(tmp_path) -> None:
    config = MPGConfig(gradient_accumulation_steps=1)
    all_epochs = [1, 5, 6, 34, 35, 36, 84, 85, 86]
    first_epochs, resumed_epochs = all_epochs[:5], all_epochs[5:]

    set_seed(404); continuous = _objects(config); continuous_state = _initial_state()
    _run(continuous, config, all_epochs, continuous_state)

    set_seed(404); interrupted = _objects(config); interrupted_state = _initial_state()
    _run(interrupted, config, first_epochs, interrupted_state)
    bundle = build_resume_bundle(
        config=config, run_id="actual-v2-1", source_hash="actual-source",
        completed_epoch=35, global_optimizer_step=interrupted_state["global_step"],
        model=interrupted[0], ema=interrupted[3], optimizer=interrupted[1],
        scheduler=interrupted[2], scaler=None,
        best_comparator_state=interrupted_state["best_comparator"],
        best_epoch=interrupted_state["best_epoch"],
        best_metrics=interrupted_state["best_metrics"],
        early_stop_counter=interrupted_state["patience"],
        history=interrupted_state["history"], loader_generator=interrupted[4],
        consistency_state={"decisions": interrupted_state["consistency_decisions"]},
        sampler_state=interrupted[6].state_dict(),
        augmentation_state=interrupted[5].state_dict(),
    )
    path, digest = atomic_save_resume(bundle, tmp_path)

    set_seed(999); resumed = _objects(config)
    loaded = load_resume_bundle(
        path, expected_sha256=digest, config=config,
        run_id="actual-v2-1", source_hash="actual-source",
    )
    restore_training_state(
        loaded, model=resumed[0], ema=resumed[3], optimizer=resumed[1],
        scheduler=resumed[2], scaler=None, loader_generator=resumed[4],
        sampler=resumed[6], dataset=resumed[5],
    )
    resumed_state = {
        "global_step": loaded["global_optimizer_step"],
        "best_comparator": copy.deepcopy(loaded["best_comparator_state"]),
        "best_epoch": loaded["best_epoch"],
        "best_metrics": copy.deepcopy(loaded["best_metrics"]),
        "patience": loaded["early_stop_counter"],
        "history": copy.deepcopy(loaded["history"]),
        "sample_order": copy.deepcopy(continuous_state["sample_order"][:5]),
        "consistency_decisions": copy.deepcopy(
            loaded["consistency_state"]["decisions"]
        ),
    }
    _run(resumed, config, resumed_epochs, resumed_state)

    _assert_nested_close(continuous[0].state_dict(), resumed[0].state_dict())
    _assert_nested_close(continuous[3].state_dict(), resumed[3].state_dict())
    _assert_optimizer_numerically_equal(
        continuous[1].state_dict(), resumed[1].state_dict()
    )
    assert continuous[2].state_dict() == resumed[2].state_dict()
    assert continuous_state == resumed_state
    assert float(resumed[0].motif_composer.temperature) == pytest.approx(0.30)
    assert "supcon_head.0.weight" in resumed[0].state_dict()
    assert resumed[3].num_updates == continuous[3].num_updates
    assert any(any(epoch) for epoch in continuous_state["consistency_decisions"])


def test_actual_v2_1_epoch_84_to_85_resume_boundary_is_invariant(tmp_path) -> None:
    config = MPGConfig(gradient_accumulation_steps=1)
    all_epochs = [84, 85, 86]
    first_epochs = [84]
    resumed_epochs = [85, 86]

    set_seed(505)
    continuous = _objects(config)
    continuous_state = _initial_state()
    _run(continuous, config, all_epochs, continuous_state)

    set_seed(505)
    interrupted = _objects(config)
    interrupted_state = _initial_state()
    _run(interrupted, config, first_epochs, interrupted_state)

    bundle = build_resume_bundle(
        config=config,
        run_id="actual-v2-1-e84",
        source_hash="actual-source",
        completed_epoch=84,
        global_optimizer_step=interrupted_state["global_step"],
        model=interrupted[0],
        ema=interrupted[3],
        optimizer=interrupted[1],
        scheduler=interrupted[2],
        scaler=None,
        best_comparator_state=interrupted_state["best_comparator"],
        best_epoch=interrupted_state["best_epoch"],
        best_metrics=interrupted_state["best_metrics"],
        early_stop_counter=interrupted_state["patience"],
        history=interrupted_state["history"],
        loader_generator=interrupted[4],
        consistency_state={"decisions": interrupted_state["consistency_decisions"]},
        sampler_state=interrupted[6].state_dict(),
        augmentation_state=interrupted[5].state_dict(),
    )
    path, digest = atomic_save_resume(bundle, tmp_path)

    del interrupted
    set_seed(999)
    resumed = _objects(config)
    loaded = load_resume_bundle(
        path,
        expected_sha256=digest,
        config=config,
        run_id="actual-v2-1-e84",
        source_hash="actual-source",
    )
    restore_training_state(
        loaded,
        model=resumed[0],
        ema=resumed[3],
        optimizer=resumed[1],
        scheduler=resumed[2],
        scaler=None,
        loader_generator=resumed[4],
        sampler=resumed[6],
        dataset=resumed[5],
    )
    resumed_state = {
        "global_step": loaded["global_optimizer_step"],
        "best_comparator": copy.deepcopy(loaded["best_comparator_state"]),
        "best_epoch": loaded["best_epoch"],
        "best_metrics": copy.deepcopy(loaded["best_metrics"]),
        "patience": loaded["early_stop_counter"],
        "history": copy.deepcopy(loaded["history"]),
        "sample_order": copy.deepcopy(continuous_state["sample_order"][:1]),
        "consistency_decisions": copy.deepcopy(
            loaded["consistency_state"]["decisions"]
        ),
    }
    _run(resumed, config, resumed_epochs, resumed_state)

    # 1. model state
    _assert_nested_close(continuous[0].state_dict(), resumed[0].state_dict())

    # 2. EMA state
    _assert_nested_close(continuous[3].state_dict(), resumed[3].state_dict())
    assert continuous[3].num_updates == resumed[3].num_updates

    # 3. AdamW state/moments
    _assert_optimizer_numerically_equal(
        continuous[1].state_dict(), resumed[1].state_dict()
    )

    # 4. scheduler state
    assert continuous[2].state_dict() == resumed[2].state_dict()

    # 5. LR
    assert [h["lr"] for h in continuous_state["history"]] == [
        h["lr"] for h in resumed_state["history"]
    ]
    assert continuous_state["history"][0]["lr"] > 1e-6
    assert continuous_state["history"][1]["lr"] == 1e-6
    assert continuous_state["history"][2]["lr"] == 1e-6
    assert resumed_state["history"][0]["lr"] > 1e-6
    assert resumed_state["history"][1]["lr"] == 1e-6
    assert resumed_state["history"][2]["lr"] == 1e-6

    # 6. global optimizer step
    assert continuous_state["global_step"] == resumed_state["global_step"]

    # 7. best comparator
    assert continuous_state["best_comparator"] == resumed_state["best_comparator"]

    # 8. best epoch/metrics
    assert continuous_state["best_epoch"] == resumed_state["best_epoch"]
    assert continuous_state["best_metrics"] == resumed_state["best_metrics"]

    # 9. patience
    assert continuous_state["patience"] == resumed_state["patience"]

    # 10. history
    assert continuous_state["history"] == resumed_state["history"]

    # 11. sampler/sample order
    assert continuous[6].state_dict() == resumed[6].state_dict()
    assert continuous_state["sample_order"] == resumed_state["sample_order"]

    # 12. augmentation trajectory
    assert continuous[5].state_dict() == resumed[5].state_dict()
    assert continuous[5].epoch == resumed[5].epoch == 86

    # 13. consistency decisions
    assert (
        continuous_state["consistency_decisions"]
        == resumed_state["consistency_decisions"]
    )

    # 14. tau: tau at 84/85/86 == 0.30
    for h in continuous_state["history"]:
        assert h["tau"] == pytest.approx(0.30)
    for h in resumed_state["history"]:
        assert h["tau"] == pytest.approx(0.30)
    assert float(continuous[0].motif_composer.temperature) == pytest.approx(0.30)
    assert float(resumed[0].motif_composer.temperature) == pytest.approx(0.30)
    assert float(continuous[3].module.motif_composer.temperature) == pytest.approx(0.30)
    assert float(resumed[3].module.motif_composer.temperature) == pytest.approx(0.30)

    # Full state equivalence
    assert continuous_state == resumed_state
