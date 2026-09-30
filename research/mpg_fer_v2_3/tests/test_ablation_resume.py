from __future__ import annotations

import copy

import pytest
import torch

from mpg_fer_table_vi.model import AblationMPGFER, AblationMode
from mpg_fer_table_vi.protocol import AblationConfig
from mpg_fer_v2_3.checkpoint import (
    atomic_save_resume,
    build_resume_bundle,
    load_resume_bundle,
    restore_training_state,
)
from mpg_fer_v2_3.ema import ModelEMA
from mpg_fer_v2_3.train import WarmupCosineScheduler
from mpg_fer_v2_3.utils import set_seed


class _Stateful:
    def __init__(self, state):
        self.state = copy.deepcopy(state)

    def state_dict(self):
        return copy.deepcopy(self.state)

    def load_state_dict(self, state):
        self.state = copy.deepcopy(state)


def _objects(config: AblationConfig):
    model = AblationMPGFER(config, config.ablation_mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = WarmupCosineScheduler(
        optimizer,
        config.learning_rate,
        config.warmup_epochs,
        config.lr_decay_end_epoch,
        config.max_epochs,
        config.min_learning_rate,
    )
    ema = ModelEMA(model, decay=0.9)
    generator = torch.Generator().manual_seed(config.seed)
    scaler = _Stateful({"scale": 1024.0, "growth_tracker": 7})
    sampler = _Stateful({"scheme": "test", "epoch": 0})
    dataset = _Stateful({"scheme": "test", "epoch": 0})
    return model, optimizer, scheduler, ema, generator, scaler, sampler, dataset


def _run(objects, config: AblationConfig, epochs, state):
    model, optimizer, scheduler, ema, generator, scaler, sampler, dataset = objects
    for epoch in epochs:
        scheduler.step(epoch)
        model.set_epoch_temperature(epoch)
        optimizer.zero_grad(set_to_none=True)
        coefficient = torch.rand((), generator=generator) + torch.rand(())
        loss = coefficient * model.classifier[-1].weight.square().mean()
        loss.backward()
        optimizer.step()
        ema.update(model)
        sampler.state["epoch"] = epoch
        dataset.state["epoch"] = epoch
        scaler.state["growth_tracker"] += 1
        state["global_step"] += 1
        state["best_comparator"] = {
            "accuracy": float(epoch),
            "macro_f1": float(epoch),
            "loss": float(loss.detach()),
        }
        state["best_epoch"] = epoch
        state["best_metrics"] = {"tta": copy.deepcopy(state["best_comparator"])}
        state["early_stop_counter"] = epoch - 1
        state["history"].append(
            {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "coefficient": float(coefficient),
            }
        )


def _initial_state():
    return {
        "global_step": 0,
        "best_comparator": None,
        "best_epoch": None,
        "best_metrics": None,
        "early_stop_counter": 0,
        "history": [],
    }


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
def test_epoch_1_to_3_is_identical_across_exact_epoch_2_resume(tmp_path, mode) -> None:
    config = AblationConfig(ablation_mode=mode.value)
    set_seed(7101)
    continuous = _objects(config)
    continuous_state = _initial_state()
    _run(continuous, config, (1, 2, 3), continuous_state)

    set_seed(7101)
    interrupted = _objects(config)
    interrupted_state = _initial_state()
    _run(interrupted, config, (1, 2), interrupted_state)
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
        consistency_state={"algorithm": "test", "last_completed_epoch": 2},
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
    _run(resumed, config, (3,), resumed_state)

    _assert_nested_equal(continuous[0].state_dict(), resumed[0].state_dict())
    _assert_nested_equal(continuous[3].state_dict(), resumed[3].state_dict())
    _assert_nested_equal(continuous[1].state_dict(), resumed[1].state_dict())
    assert continuous[2].state_dict() == resumed[2].state_dict()
    assert continuous[5].state_dict() == resumed[5].state_dict()
    assert continuous[6].state_dict() == resumed[6].state_dict()
    assert continuous[7].state_dict() == resumed[7].state_dict()
    assert continuous_state == resumed_state


def test_resume_identity_rejects_cross_variant_checkpoint(tmp_path) -> None:
    full = AblationConfig(ablation_mode=AblationMode.FULL.value)
    single = AblationConfig(ablation_mode=AblationMode.SINGLE_SCALE_12.value)
    set_seed(42)
    objects = _objects(full)
    state = _initial_state()
    _run(objects, full, (1,), state)
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
        best_epoch=1,
        best_metrics=state["best_metrics"],
        early_stop_counter=0,
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
