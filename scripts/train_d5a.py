"""Train D5A class-level pixel motif graph retrieval."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.log_experiment import log_experiment
from common import apply_cli_overrides, build_dataloader, create_trainer, load_config


def run_train(config):
    train_loader = build_dataloader(config, split="train", shuffle=True)
    val_loader = build_dataloader(config, split="val", shuffle=False)
    trainer = create_trainer(config)
    resume_cfg = config.get("resume", {}) or {}
    init_cfg = config.get("init_checkpoint", {}) or {}
    if resume_cfg.get("path"):
        trainer.load_checkpoint(
            resume_cfg["path"],
            load_optimizer=bool(resume_cfg.get("load_optimizer", True)),
            load_scheduler=bool(resume_cfg.get("load_scheduler", True)),
            load_epoch=bool(resume_cfg.get("load_epoch", True)),
            strict=bool(resume_cfg.get("strict", True)),
            best_metric=resume_cfg.get("best_metric"),
            best_epoch=resume_cfg.get("best_epoch"),
        )
        best_checkpoint_path = resume_cfg.get("best_checkpoint_path")
        if best_checkpoint_path:
            src = Path(best_checkpoint_path)
            dst = trainer.checkpoint_dir / "best.pth"
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                print(f"[Resume] seeded output best checkpoint from {src}")
            elif not src.exists():
                print(f"[Resume] best_checkpoint_path not found: {src}")
    elif init_cfg.get("path"):
        trainer.init_from_checkpoint(
            init_cfg["path"],
            strict=bool(init_cfg.get("strict", True)),
        )
    training_cfg = config.get("training", {})
    scheduler_cfg = config.get("scheduler", {}) or {}
    checkpoint_cfg = config.get("checkpoint", {}) or {}
    early_stopping_cfg = config.get("early_stopping", {}) or {}
    classification_monitor = training_cfg.get("monitor", "val_macro_f1")
    scheduler_name = str(scheduler_cfg.get("name", "")).lower()
    if scheduler_name == "reducelronplateau":
        scheduler_monitor = scheduler_cfg.get("monitor", "val_loss")
        scheduler_mode = scheduler_cfg.get("mode", "min")
    else:
        scheduler_monitor = None
        scheduler_mode = "max"
        if scheduler_cfg.get("monitor") is not None:
            print(
                f"[Scheduler] WARNING: scheduler={scheduler_cfg.get('name')} "
                f"steps by epoch; ignoring monitor={scheduler_cfg.get('monitor')!r}."
            )
    checkpoint_monitor = checkpoint_cfg.get("save_best_metric", classification_monitor)
    checkpoint_mode = checkpoint_cfg.get("save_best_mode", checkpoint_cfg.get("mode", "max"))
    early_stopping_monitor = early_stopping_cfg.get("monitor", classification_monitor)
    early_stopping_mode = early_stopping_cfg.get("mode", "max")
    early_stopping_patience = early_stopping_cfg.get(
        "patience",
        training_cfg.get("early_stopping_patience", 20),
    )
    result = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(training_cfg.get("epochs", 80)),
        monitor=scheduler_monitor,
        scheduler_mode=scheduler_mode,
        checkpoint_monitor=checkpoint_monitor,
        checkpoint_mode=checkpoint_mode,
        early_stopping_monitor=early_stopping_monitor,
        early_stopping_mode=early_stopping_mode,
        early_stopping_patience=int(early_stopping_patience),
        max_train_batches=training_cfg.get("max_train_batches"),
        max_val_batches=training_cfg.get("max_val_batches"),
    )
    print(f"Training done best_epoch={result['best_epoch']} best_metric={result['best_metric']:.6f}")
    try:
        log_experiment(trainer.output_root)
    except Exception as e:
        print(f"Failed to log experiment: {e}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/d5a.yaml")
    parser.add_argument("--environment", "--env", choices=["local", "kaggle"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--graph_repo_path", default=None)
    parser.add_argument("--csv_root", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--max_test_batches", type=int, default=None)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--resume_best_checkpoint", default=None)
    parser.add_argument("--init_checkpoint", default=None)
    # --- Performance profiling & optimisation ---
    parser.add_argument("--profile_batches", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--pin_memory", default=None)
    parser.add_argument("--persistent_workers", default=None)
    parser.add_argument("--prefetch_factor", type=int, default=None)
    parser.add_argument("--graph_cache_chunks", type=int, default=None)
    parser.add_argument("--chunk_cache_size", type=int, default=None)
    parser.add_argument("--chunk_aware_shuffle", action="store_true", default=False)
    parser.add_argument("--no_chunk_aware_shuffle", action="store_true", default=False)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--no_amp", action="store_true", default=False)
    args = parser.parse_args()
    config = apply_cli_overrides(load_config(args.config, environment=args.environment), args)
    if args.resume is not None:
        config["resume"] = {
            **dict(config.get("resume", {}) or {}),
            "path": args.resume,
        }
        config.pop("init_checkpoint", None)
    if args.resume_best_checkpoint is not None:
        config["resume"] = {
            **dict(config.get("resume", {}) or {}),
            "best_checkpoint_path": args.resume_best_checkpoint,
        }
    if args.init_checkpoint is not None:
        config["init_checkpoint"] = {
            **dict(config.get("init_checkpoint", {}) or {}),
            "path": args.init_checkpoint,
        }
        config.pop("resume", None)
    run_train(config)


if __name__ == "__main__":
    main()
