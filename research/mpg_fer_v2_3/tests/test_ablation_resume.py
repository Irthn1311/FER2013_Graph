from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler

from mpg_fer_table_vi.model import AblationMPGFER, AblationMode
from mpg_fer_table_vi.protocol import AblationConfig
from mpg_fer_table_vi.train import (
    build_recipe_optimizer,
    build_recipe_scheduler,
    train_ablation_one_epoch,
)
from mpg_fer_v2_3 import train as baseline_train
from mpg_fer_v2_3.checkpoint import (
    atomic_save_resume,
    build_resume_bundle,
    load_resume_bundle,
    restore_training_state,
)
from mpg_fer_v2_3.ema import ModelEMA
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
        generator = torch.Generator().manual_seed(101)
        self.images = torch.rand(4, 1, 48, 48, generator=generator)
        # Every physical microbatch has valid positive pairs for the in-batch
        # SupCon objective; accumulation still spans exactly two microbatches.
        self.labels = torch.tensor([0, 0, 0, 0])
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = (self.images[index] + self.epoch * 1e-4).clamp(0.0, 1.0)
        return image, self.labels[index]

    def state_dict(self):
        return {"scheme": "synthetic_epoch_v1", "epoch": self.epoch}

    def load_state_dict(self, state):
        assert state["scheme"] == "synthetic_epoch_v1"
        self.epoch = int(state["epoch"])


class _EpochSampler(Sampler[int]):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(9000 + self.epoch)
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.dataset)

    def state_dict(self):
        return {"scheme": "epoch_randperm_v1", "epoch": self.epoch}

    def load_state_dict(self, state):
        assert state["scheme"] == "epoch_randperm_v1"
        self.epoch = int(state["epoch"])


class _DeterministicScaler:
    """CPU test double exercising the real train loop scaler lifecycle."""

    def __init__(self) -> None:
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
        return {"scheme": "deterministic_cpu_v1", "updates": self.updates}

    def load_state_dict(self, state):
        assert state["scheme"] == "deterministic_cpu_v1"
        self.updates = int(state["updates"])


def _config(mode: AblationMode) -> AblationConfig:
    return AblationConfig(
        ablation_mode=mode.value,
        use_amp=False,
        gradient_accumulation_steps=2,
        optimizer_family="AdamW",
        optimizer_kwargs={},
        scheduler_family="linear_warmup_cosine_then_floor",
        scheduler_kwargs={},
    )


def _objects(config: AblationConfig):
    model = AblationMPGFER(config, config.ablation_mode)
    optimizer = build_recipe_optimizer(model, config)
    scheduler = build_recipe_scheduler(optimizer, config)
    ema = ModelEMA(model, decay=0.9)
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


def _initial_state():
    return {
        "global_step": 0,
        "best_comparator": None,
        "best_epoch": None,
        "best_metrics": None,
        "early_stop_counter": 0,
        "history": [],
    }


def _run_actual_lifecycle(objects, config: AblationConfig, epochs, state):
    model, optimizer, scheduler, ema, generator, scaler, sampler, dataset, loader = objects
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    assert len(loader) == 2
    assert config.gradient_accumulation_steps == 2
    for epoch in epochs:
        sampler.epoch = dataset.epoch = epoch
        lr = scheduler.step(epoch)
        prior_global_step = state["global_step"]
        stats, state["global_step"] = train_ablation_one_epoch(
            model,
            loader,
            optimizer,
            "cpu",
            scaler,
            config,
            criterion,
            ema=ema,
            epoch=epoch,
            global_optimizer_step=state["global_step"],
        )
        assert state["global_step"] == prior_global_step + 1
        expected_consistency = (
            [0]
            if baseline_train.consistency_selected(
                config.seed, epoch, 0, config.consistency_probability
            )
            else []
        )
        assert stats["consistency_groups"] == expected_consistency
        assert stats["consistency_group_fraction"] == float(
            bool(expected_consistency)
        )
        candidate = {
            "accuracy": stats["train_accuracy"],
            "macro_f1": stats["train_accuracy"],
            "loss": stats["train_loss"],
        }
        improved = baseline_train.is_better_checkpoint(
            candidate, state["best_comparator"]
        )
        if improved:
            state["best_comparator"] = copy.deepcopy(candidate)
            state["best_epoch"] = epoch
            state["best_metrics"] = {"synthetic": copy.deepcopy(candidate)}
        state["early_stop_counter"] = baseline_train.update_early_stop_patience(
            epoch, improved, state["early_stop_counter"], config
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


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_nested_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("mode", [AblationMode.FULL, AblationMode.SINGLE_SCALE_12])
def test_real_epoch_lifecycle_is_identical_across_exact_epoch_2_resume(
    tmp_path, mode
) -> None:
    """T7: real train loop plus all persisted stochastic lifecycle state."""
    config = _config(mode)
    set_seed(7101)
    continuous = _objects(config)
    continuous_state = _initial_state()
    _run_actual_lifecycle(continuous, config, (1, 2, 3), continuous_state)

    set_seed(7101)
    interrupted = _objects(config)
    interrupted_state = _initial_state()
    _run_actual_lifecycle(interrupted, config, (1, 2), interrupted_state)
    bundle = build_resume_bundle(
        config=config,
        run_id=f"table-vi-{mode.value.lower()}",
        source_hash="ablation-source",
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
        early_stop_counter=interrupted_state["early_stop_counter"],
        history=interrupted_state["history"],
        loader_generator=interrupted[4],
        consistency_state={"algorithm": "hash_v1", "last_completed_epoch": 2},
        sampler_state=interrupted[6].state_dict(),
        augmentation_state=interrupted[7].state_dict(),
    )
    checkpoint, digest = atomic_save_resume(bundle, tmp_path / mode.value)

    set_seed(999)
    resumed = _objects(config)
    loaded = load_resume_bundle(
        checkpoint,
        expected_sha256=digest,
        config=config,
        run_id=f"table-vi-{mode.value.lower()}",
        source_hash="ablation-source",
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
        "early_stop_counter": loaded["early_stop_counter"],
        "history": copy.deepcopy(loaded["history"]),
    }
    _run_actual_lifecycle(resumed, config, (3,), resumed_state)

    _assert_nested_equal(continuous[0].state_dict(), resumed[0].state_dict())
    _assert_nested_equal(continuous[3].state_dict(), resumed[3].state_dict())
    _assert_nested_equal(continuous[1].state_dict(), resumed[1].state_dict())
    assert continuous[2].state_dict() == resumed[2].state_dict()
    assert continuous[5].state_dict() == resumed[5].state_dict()
    assert continuous[6].state_dict() == resumed[6].state_dict()
    assert continuous[7].state_dict() == resumed[7].state_dict()
    assert continuous_state == resumed_state
    assert continuous_state["global_step"] == 3
    assert continuous[3].num_updates == 3
    assert continuous[5].state_dict()["updates"] == 3


def test_resume_identity_rejects_cross_variant_checkpoint(tmp_path) -> None:
    full = _config(AblationMode.FULL)
    single = _config(AblationMode.SINGLE_SCALE_12)
    set_seed(42)
    objects = _objects(full)
    state = _initial_state()
    _run_actual_lifecycle(objects, full, (1,), state)
    bundle = build_resume_bundle(
        config=full,
        run_id="full",
        source_hash="source",
        completed_epoch=1,
        global_optimizer_step=state["global_step"],
        model=objects[0],
        ema=objects[3],
        optimizer=objects[1],
        scheduler=objects[2],
        scaler=objects[5],
        best_comparator_state=state["best_comparator"],
        best_epoch=state["best_epoch"],
        best_metrics=state["best_metrics"],
        early_stop_counter=state["early_stop_counter"],
        history=state["history"],
        loader_generator=objects[4],
        consistency_state={},
        sampler_state=objects[6].state_dict(),
        augmentation_state=objects[7].state_dict(),
    )
    checkpoint, digest = atomic_save_resume(bundle, tmp_path)
    with pytest.raises(RuntimeError, match="config mismatch"):
        load_resume_bundle(
            checkpoint,
            expected_sha256=digest,
            config=single,
            run_id="full",
            source_hash="source",
        )
