"""Small D16 resume-integrity check for full-state Kaggle continuation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from d16.data.graph_builder import collate_d16_graphs
from d16.models.d16_model import D16Model
from d16.training.train_d16 import (
    _append_csv,
    _loader_kwargs,
    _make_grad_scaler,
    _write_json,
    attach_hard_proto_loss_if_needed,
    build_dataset,
    load_config,
    resume_training,
    resolve_device,
    save_checkpoint,
    set_seed,
    train_one_epoch,
)


def _model_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _adam_exp_avg_norms(optimizer: torch.optim.Optimizer, limit: int = 5) -> List[float]:
    norms: List[float] = []
    for state in optimizer.state.values():
        value = state.get("exp_avg") if isinstance(state, dict) else None
        if torch.is_tensor(value):
            norms.append(float(value.detach().float().norm().cpu().item()))
            if len(norms) >= int(limit):
                break
    return norms


def _first_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _make_loader(cfg: Dict[str, Any], prior_dir: Path):
    train_ds = build_dataset(cfg, prior_dir, "train")
    return train_ds, DataLoader(train_ds, **_loader_kwargs(cfg.get("data", {}) or {}, cfg.get("training", {}) or {}, shuffle=False))


def _reset_check_artifacts(output_dir: Path) -> None:
    for path in [
        output_dir / "train_log.csv",
        output_dir / "resume_events.jsonl",
        output_dir / "resume_info.json",
        output_dir / "resume_integrity_summary.json",
        output_dir / "D16_RESUME_INTEGRITY_REPORT.md",
        output_dir / "checkpoints" / "last.pt",
        output_dir / "checkpoints" / "last_prev.pt",
        output_dir / "checkpoints" / "last.pt.tmp",
    ]:
        if path.exists() and path.is_file():
            path.unlink()


def run_check(
    config_path: Path,
    prior_dir: Path,
    output_dir: Path,
    device_name: str,
    num_batches_before_ckpt: int,
    num_batches_after_resume: int,
) -> Dict[str, Any]:
    cfg = load_config(config_path)
    seed = cfg.get("seed", (cfg.get("training", {}) or {}).get("seed"))
    if seed is not None:
        set_seed(int(seed))
    data_cfg = cfg.setdefault("data", {})
    training_cfg = cfg.setdefault("training", {})
    # Keep the check tiny and Windows-safe; Kaggle full run keeps config workers.
    training_cfg["num_workers"] = 0
    data_cfg["max_train_samples"] = int(training_cfg.get("batch_size", data_cfg.get("batch_size", 16))) * (
        int(num_batches_before_ckpt) + int(num_batches_after_resume) + 2
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    _reset_check_artifacts(output_dir)
    device = resolve_device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp_enabled = bool(training_cfg.get("amp", training_cfg.get("mixed_precision", False))) and device.type == "cuda"

    train_ds, train_loader = _make_loader(cfg, prior_dir)
    first_batch = next(iter(DataLoader(train_ds, batch_size=1, shuffle=False, collate_fn=collate_d16_graphs)))
    input_dim = int(first_batch.x_cat.size(1))
    model = D16Model.from_config(cfg, input_dim=input_dim).to(device)
    loss_cfg = cfg.get("loss", {}) or {}
    hard_proto_loss_fn = attach_hard_proto_loss_if_needed(
        model,
        loss_cfg,
        embedding_dim=int((cfg.get("model", {}) or {}).get("hidden_dim", 96)) * 5,
    )
    if hard_proto_loss_fn is not None:
        hard_proto_loss_fn.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("lr", 3e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    scaler = _make_grad_scaler(bool(amp_enabled))

    before_stats = train_one_epoch(
        model,
        train_loader,
        optimizer,
        device,
        epoch=1,
        progress_interval=0,
        loss_cfg=loss_cfg,
        hard_proto_loss_fn=hard_proto_loss_fn,
        limit_batches=int(num_batches_before_ckpt),
        amp_enabled=amp_enabled,
        scaler=scaler,
    )
    global_step_before = int(before_stats["train_num_batches"])
    best_score = 0.25
    save_checkpoint(
        output_dir / "checkpoints" / "last.pt",
        model,
        optimizer,
        epoch=1,
        best_val_macro_f1=0.123,
        config=cfg,
        global_step=global_step_before,
        input_dim=input_dim,
        best_epoch=1,
        epochs_without_improvement=2,
        best_monitor_metric="val_accuracy",
        best_monitor_mode="max",
        best_monitor_score=best_score,
        scaler=scaler,
    )
    _append_csv(
        output_dir / "train_log.csv",
        {"epoch": 1, "global_step": global_step_before, "train_loss": before_stats["train_loss"]},
        ["epoch", "global_step", "train_loss"],
    )
    model_hash_before = _model_hash(model)
    lr_before = _first_lr(optimizer)
    exp_avg_before = _adam_exp_avg_norms(optimizer)
    checkpoint = torch.load(output_dir / "checkpoints" / "last.pt", map_location=device, weights_only=False)

    model2 = D16Model.from_config(cfg, input_dim=input_dim).to(device)
    hard_proto_loss_fn2 = attach_hard_proto_loss_if_needed(
        model2,
        loss_cfg,
        embedding_dim=int((cfg.get("model", {}) or {}).get("hidden_dim", 96)) * 5,
    )
    if hard_proto_loss_fn2 is not None:
        hard_proto_loss_fn2.to(device)
    optimizer2 = torch.optim.AdamW(
        model2.parameters(),
        lr=float(training_cfg.get("lr", 3e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    scaler2 = _make_grad_scaler(bool(amp_enabled))
    resume_state = resume_training(
        output_dir / "checkpoints" / "last.pt",
        model2,
        optimizer2,
        device,
        output_dir,
        restore_rng=True,
        scaler=scaler2,
        current_config=cfg,
        current_input_dim=input_dim,
        strict=True,
    )
    model_hash_after_resume = _model_hash(model2)
    lr_after_resume = _first_lr(optimizer2)
    exp_avg_after = _adam_exp_avg_norms(optimizer2)

    _, train_loader2 = _make_loader(cfg, prior_dir)
    after_stats = train_one_epoch(
        model2,
        train_loader2,
        optimizer2,
        device,
        epoch=int(resume_state["start_epoch"]),
        progress_interval=0,
        loss_cfg=loss_cfg,
        hard_proto_loss_fn=hard_proto_loss_fn2,
        limit_batches=int(num_batches_after_resume),
        amp_enabled=amp_enabled,
        scaler=scaler2,
    )
    global_step_after = int(resume_state["global_step"]) + int(after_stats["train_num_batches"])
    save_checkpoint(
        output_dir / "checkpoints" / "last.pt",
        model2,
        optimizer2,
        epoch=int(resume_state["start_epoch"]),
        best_val_macro_f1=float(resume_state["best_val_macro_f1"]),
        config=cfg,
        global_step=global_step_after,
        input_dim=input_dim,
        best_epoch=int(resume_state["best_epoch"]),
        epochs_without_improvement=int(resume_state["epochs_without_improvement"]),
        resume_source=str(output_dir / "checkpoints" / "last.pt"),
        best_monitor_metric=str(resume_state["best_monitor_metric"]),
        best_monitor_mode=str(resume_state["best_monitor_mode"]),
        best_monitor_score=float(resume_state["best_monitor_score"]),
        scaler=scaler2,
    )
    _append_csv(
        output_dir / "train_log.csv",
        {"epoch": int(resume_state["start_epoch"]), "global_step": global_step_after, "train_loss": after_stats["train_loss"]},
        ["epoch", "global_step", "train_loss"],
    )
    final_checkpoint = torch.load(output_dir / "checkpoints" / "last.pt", map_location=device, weights_only=False)
    train_log_rows = _read_csv(output_dir / "train_log.csv")

    optimizer_state_restored = bool(exp_avg_before) and len(exp_avg_before) == len(exp_avg_after)
    if optimizer_state_restored:
        optimizer_state_restored = all(abs(a - b) <= max(1e-6, abs(a) * 1e-5) for a, b in zip(exp_avg_before, exp_avg_after))
    summary = {
        "decision": "PASS",
        "config": str(config_path),
        "prior_dir": str(prior_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "amp": bool(amp_enabled),
        "checkpoint_keys": sorted(str(k) for k in checkpoint.keys()),
        "model_state_restored": model_hash_before == model_hash_after_resume,
        "optimizer_state_restored": bool(optimizer_state_restored),
        "scheduler_state_restored": checkpoint.get("scheduler_state_dict") is None and checkpoint.get("scheduler_type") == "none",
        "scaler_state_restored": bool(not amp_enabled or scaler2.state_dict()),
        "lr_continues": math.isclose(lr_before, lr_after_resume, rel_tol=0.0, abs_tol=1e-12),
        "global_step_before": global_step_before,
        "global_step_after_resume": int(resume_state["global_step"]),
        "global_step_after": global_step_after,
        "global_step_continues": int(resume_state["global_step"]) == global_step_before and global_step_after == global_step_before + int(num_batches_after_resume),
        "rng_restored": bool((json.loads((output_dir / "resume_info.json").read_text()))["rng_restored"]),
        "start_epoch": int(resume_state["start_epoch"]),
        "start_epoch_correct": int(resume_state["start_epoch"]) == 2,
        "best_metric_restored": math.isclose(float(resume_state["best_monitor_score"]), best_score, rel_tol=0.0, abs_tol=1e-12),
        "best_epoch_restored": int(resume_state["best_epoch"]) == 1,
        "early_stop_restored": int(resume_state["epochs_without_improvement"]) == 2,
        "log_append_safe": len(train_log_rows) == 2 and [int(float(r["epoch"])) for r in train_log_rows] == [1, 2],
        "no_architecture_mismatch": bool((json.loads((output_dir / "resume_info.json").read_text())["config_compatibility"])["compatible"]),
        "no_nan": all(torch.isfinite(p).all().item() for p in model2.parameters()),
        "final_checkpoint_global_step": int(final_checkpoint.get("global_step", -1)),
        "resume_state": resume_state,
    }
    required = [
        "model_state_restored",
        "optimizer_state_restored",
        "scheduler_state_restored",
        "scaler_state_restored",
        "lr_continues",
        "global_step_continues",
        "rng_restored",
        "start_epoch_correct",
        "best_metric_restored",
        "early_stop_restored",
        "log_append_safe",
        "no_architecture_mismatch",
        "no_nan",
    ]
    failures = [key for key in required if not bool(summary.get(key))]
    summary["failures"] = failures
    if failures:
        summary["decision"] = "FAIL"
    _write_json(output_dir / "resume_integrity_summary.json", summary)
    lines = [
        "# D16 Resume Integrity Report",
        "",
        f"- decision: `{summary['decision']}`",
        f"- model_state_restored: `{summary['model_state_restored']}`",
        f"- optimizer_state_restored: `{summary['optimizer_state_restored']}`",
        f"- scheduler_state_restored: `{summary['scheduler_state_restored']}`",
        f"- scaler_state_restored: `{summary['scaler_state_restored']}`",
        f"- lr_continues: `{summary['lr_continues']}`",
        f"- global_step_continues: `{summary['global_step_continues']}`",
        f"- rng_restored: `{summary['rng_restored']}`",
        f"- start_epoch_correct: `{summary['start_epoch_correct']}`",
        f"- best_metric_restored: `{summary['best_metric_restored']}`",
        f"- early_stop_restored: `{summary['early_stop_restored']}`",
        f"- log_append_safe: `{summary['log_append_safe']}`",
        f"- no_architecture_mismatch: `{summary['no_architecture_mismatch']}`",
        f"- failures: `{failures}`",
    ]
    (output_dir / "D16_RESUME_INTEGRITY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str), flush=True)
    if summary["decision"] != "PASS":
        raise SystemExit(1)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prior_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_batches_before_ckpt", type=int, default=3)
    parser.add_argument("--num_batches_after_resume", type=int, default=2)
    args = parser.parse_args()
    run_check(
        config_path=Path(args.config),
        prior_dir=Path(args.prior_dir),
        output_dir=Path(args.output_dir),
        device_name=str(args.device),
        num_batches_before_ckpt=int(args.num_batches_before_ckpt),
        num_batches_after_resume=int(args.num_batches_after_resume),
    )


if __name__ == "__main__":
    main()
