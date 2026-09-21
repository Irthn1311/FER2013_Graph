"""Measure full v2.1 resume serialization, hashing, and load overhead locally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

import torch

from mpg_fer_v2_1.checkpoint import (
    atomic_save_resume,
    build_resume_bundle,
    load_resume_bundle,
)
from mpg_fer_v2_1.config import MPGConfig
from mpg_fer_v2_1.ema import ModelEMA
from mpg_fer_v2_1.model import MPGFER
from mpg_fer_v2_1.train import WarmupCosineScheduler, source_tree_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    config = MPGConfig(device="cpu", use_amp=False)
    model = MPGFER(config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    # Populate complete AdamW moment state without a costly FER forward.
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    scheduler = WarmupCosineScheduler(
        optimizer, config.learning_rate, config.warmup_epochs,
        config.lr_decay_end_epoch, config.max_epochs, config.min_learning_rate,
    )
    scheduler.step(1)
    ema = ModelEMA(model, config.ema_decay); ema.update(model)
    generator = torch.Generator().manual_seed(config.seed)
    bundle = build_resume_bundle(
        config=config, run_id="resume-benchmark", source_hash=source_tree_hash(),
        completed_epoch=1, global_optimizer_step=1, model=model, ema=ema,
        optimizer=optimizer, scheduler=scheduler, scaler=None,
        best_comparator_state={"accuracy": 0.0, "macro_f1": 0.0, "loss": 0.0},
        best_epoch=1, best_metrics={"tta": {"accuracy": 0.0}},
        early_stop_counter=0, history=[{"epoch": 1}],
        loader_generator=generator,
        consistency_state={"algorithm": "stateless_seed_epoch_group_v1"},
        sampler_state={"scheme": "base_seed_epoch_v1", "seed": config.seed, "epoch": 1},
        augmentation_state={"scheme": "sample_local_seed_v1", "seed": config.seed, "epoch": 1},
    )
    temporary_context = None
    if args.output_dir is None:
        temporary_context = tempfile.TemporaryDirectory(prefix="mpg-fer-v2-1-resume-")
        output = Path(temporary_context.name)
    else:
        output = args.output_dir
        output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    path, digest = atomic_save_resume(bundle, output)
    save_and_hash_seconds = time.perf_counter() - start
    start = time.perf_counter()
    loaded = load_resume_bundle(
        path, expected_sha256=digest, config=config,
        run_id="resume-benchmark", source_hash=source_tree_hash(),
    )
    load_and_hash_seconds = time.perf_counter() - start
    result = {
        "checkpoint_bytes": path.stat().st_size,
        "checkpoint_mib": path.stat().st_size / 2**20,
        "save_and_sha256_seconds": save_and_hash_seconds,
        "load_and_sha256_seconds": load_and_hash_seconds,
        "schema": loaded["resume_schema_version"],
        "tensor_device": "cpu",
    }
    print(json.dumps(result, indent=2))
    if temporary_context is not None:
        temporary_context.cleanup()


if __name__ == "__main__":
    main()
