from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler

from mpg_fer_o1.protocol import resolve_o1_config
from mpg_fer_v2_3 import train as baseline_train
from mpg_fer_v2_3.checkpoint import (
    atomic_save_resume,
    build_resume_bundle,
    load_resume_bundle,
    restore_training_state,
)
from mpg_fer_v2_3.ema import ModelEMA
from mpg_fer_v2_3.model import MPGFER
from mpg_fer_v2_3.utils import set_seed


@pytest.fixture(autouse=True)
def _deterministic_cpu_backend():
    prior_algorithms = torch.are_deterministic_algorithms_enabled()
    prior_threads = torch.get_num_threads()
    prior_mkldnn = torch.backends.mkldnn.enabled
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    try:
        yield
    finally:
        torch.backends.mkldnn.enabled = prior_mkldnn
        torch.set_num_threads(prior_threads)
        torch.use_deterministic_algorithms(prior_algorithms)


class _SyntheticEpochDataset(Dataset):
    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(103)
        self.images = torch.rand(4, 1, 48, 48, generator=generator)
        self.labels = torch.tensor([0, 0, 0, 0])
        self.epoch = 0

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return (self.images[index] + self.epoch * 1e-4).clamp(0, 1), self.labels[index]

    def state_dict(self):
        return {"scheme": "o1_synthetic_epoch_v1", "epoch": self.epoch}

    def load_state_dict(self, state):
        assert state["scheme"] == "o1_synthetic_epoch_v1"
        self.epoch = int(state["epoch"])


class _EpochSampler(Sampler[int]):
    def __init__(self, dataset):
        self.dataset = dataset
        self.epoch = 0

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        generator = torch.Generator().manual_seed(10300 + self.epoch)
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())

    def state_dict(self):
        return {"scheme": "o1_epoch_sampler_v1", "epoch": self.epoch}

    def load_state_dict(self, state):
        assert state["scheme"] == "o1_epoch_sampler_v1"
        self.epoch = int(state["epoch"])


class _DeterministicScaler:
    def __init__(self):
        self.updates = 0

    def scale(self, loss):
        return loss

    def unscale_(self, _optimizer):
        return None

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        self.updates += 1

    def get_scale(self):
        return 1.0

    def state_dict(self):
        return {"scheme": "o1_cpu_scaler_v1", "updates": self.updates}

    def load_state_dict(self, state):
        assert state["scheme"] == "o1_cpu_scaler_v1"
        self.updates = int(state["updates"])


def _objects(config):
    model = MPGFER(config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = baseline_train.WarmupCosineScheduler(
        optimizer,
        config.learning_rate,
        config.warmup_epochs,
        config.lr_decay_end_epoch,
        config.max_epochs,
        config.min_learning_rate,
    )
    ema = ModelEMA(model, decay=config.ema_decay)
    generator = torch.Generator().manual_seed(config.seed)
    scaler = _DeterministicScaler()
    dataset = _SyntheticEpochDataset()
    sampler = _EpochSampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=2,
        sampler=sampler,
        num_workers=0,
        generator=generator,
    )
    return model, optimizer, scheduler, ema, generator, scaler, sampler, dataset, loader


def _state():
    return {
        "global_step": 0,
        "best_comparator": None,
        "best_epoch": None,
        "best_metrics": None,
        "patience": 0,
        "history": [],
    }


def _run(objects, config, epochs, state):
    model, optimizer, scheduler, ema, generator, scaler, sampler, dataset, loader = objects
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    assert len(loader) == 2
    assert config.gradient_accumulation_steps == 2
    for epoch in epochs:
        sampler.epoch = dataset.epoch = epoch
        lr = scheduler.step(epoch)
        prior_step = state["global_step"]
        stats, state["global_step"] = baseline_train.train_one_epoch(
            model,
            loader,
            optimizer,
            "cpu",
            scaler,
            config,
            criterion,
            ema=ema,
            epoch=epoch,
            global_optimizer_step=prior_step,
        )
        assert state["global_step"] == prior_step + 1
        expected_groups = (
            [0]
            if baseline_train.consistency_selected(
                config.seed, epoch, 0, config.consistency_probability
            )
            else []
        )
        assert stats["consistency_groups"] == expected_groups
        candidate = {
            "accuracy": stats["train_accuracy"],
            "macro_f1": stats["train_accuracy"],
            "loss": stats["train_loss"],
        }
        improved = baseline_train.is_better_checkpoint(candidate, state["best_comparator"])
        if improved:
            state["best_comparator"] = copy.deepcopy(candidate)
            state["best_epoch"] = epoch
            state["best_metrics"] = {"synthetic": copy.deepcopy(candidate)}
        state["patience"] = baseline_train.update_early_stop_patience(
            epoch, improved, state["patience"], config
        )
        state["history"].append(
            {
                "epoch": epoch,
                "lr": lr,
                "stats": stats,
                "rng_probe": float(torch.rand(())),
                "loader_rng_probe": float(torch.rand((), generator=generator)),
            }
        )


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _equal(a, b)
    else:
        assert left == right


def test_o1_t7_real_two_microbatch_lifecycle_exact_resume(tmp_path) -> None:
    config = resolve_o1_config("O1_01")
    set_seed(7103)
    continuous = _objects(config)
    continuous_state = _state()
    _run(continuous, config, (1, 2, 3), continuous_state)

    set_seed(7103)
    interrupted = _objects(config)
    interrupted_state = _state()
    _run(interrupted, config, (1, 2), interrupted_state)
    bundle = build_resume_bundle(
        config=config,
        run_id="o1-resume-test",
        source_hash="o1-source",
        completed_epoch=2,
        global_optimizer_step=interrupted_state["global_step"],
        model=interrupted[0],
        ema=interrupted[3],
        optimizer=interrupted[1],
        scheduler=interrupted[2],
        scaler=interrupted[5],
        best_comparator_state=interrupted_state["best_comparator"],
        best_epoch=interrupted_state["best_epoch"],
        best_metrics=interrupted_state["best_metrics"],
        early_stop_counter=interrupted_state["patience"],
        history=interrupted_state["history"],
        loader_generator=interrupted[4],
        consistency_state={"algorithm": "stateless_seed_epoch_group_v1"},
        sampler_state=interrupted[6].state_dict(),
        augmentation_state=interrupted[7].state_dict(),
    )
    checkpoint, digest = atomic_save_resume(bundle, tmp_path)

    set_seed(999)
    resumed = _objects(config)
    loaded = load_resume_bundle(
        checkpoint,
        expected_sha256=digest,
        config=config,
        run_id="o1-resume-test",
        source_hash="o1-source",
    )
    restore_training_state(
        loaded,
        model=resumed[0],
        ema=resumed[3],
        optimizer=resumed[1],
        scheduler=resumed[2],
        scaler=resumed[5],
        loader_generator=resumed[4],
        sampler=resumed[6],
        dataset=resumed[7],
    )
    resumed_state = {
        "global_step": loaded["global_optimizer_step"],
        "best_comparator": copy.deepcopy(loaded["best_comparator_state"]),
        "best_epoch": loaded["best_epoch"],
        "best_metrics": copy.deepcopy(loaded["best_metrics"]),
        "patience": loaded["early_stop_counter"],
        "history": copy.deepcopy(loaded["history"]),
    }
    _run(resumed, config, (3,), resumed_state)
    for index in (0, 1, 3):
        _equal(continuous[index].state_dict(), resumed[index].state_dict())
    assert continuous[2].state_dict() == resumed[2].state_dict()
    assert continuous[5].state_dict() == resumed[5].state_dict()
    assert continuous[6].state_dict() == resumed[6].state_dict()
    assert continuous[7].state_dict() == resumed[7].state_dict()
    assert continuous_state == resumed_state
    assert continuous_state["global_step"] == 3
    assert continuous[3].num_updates == 3
    assert continuous[5].updates == 3
