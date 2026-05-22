"""Check D15 Kaggle-safe resume integrity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch


REQUIRED_CHECKPOINT_KEYS = {
    "epoch",
    "global_step",
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "best_val_macro_f1",
    "best_epoch",
    "early_stopping_state",
    "rng_state",
    "resolved_config",
    "run_name",
    "from_scratch",
    "init_checkpoint",
    "resume_source",
    "dataloader_sampler_state_info",
}


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"parse_error": line})
    return rows


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return torch.load(path, map_location="cpu", weights_only=False)


def _checkpoint_missing_fields(ckpt: Dict[str, Any]) -> List[str]:
    return sorted(key for key in REQUIRED_CHECKPOINT_KEYS if key not in ckpt)


def _epoch_sequence_failures(train_log: pd.DataFrame) -> List[str]:
    if train_log.empty or "epoch" not in train_log:
        return ["train_log.csv missing epoch sequence"]
    epochs = pd.to_numeric(train_log["epoch"], errors="coerce").dropna().astype(int).tolist()
    if not epochs:
        return ["train_log.csv has no numeric epochs"]
    failures = []
    if len(epochs) != len(set(epochs)):
        failures.append("duplicate epoch rows in train_log.csv")
    expected = list(range(min(epochs), max(epochs) + 1))
    if sorted(epochs) != expected:
        failures.append(f"epoch gap: got {epochs[:5]}...{epochs[-5:]} expected contiguous {expected[0]}..{expected[-1]}")
    return failures


def _lr_jump_warning(train_log: pd.DataFrame, resume_events: List[Dict[str, Any]]) -> List[str]:
    if train_log.empty or "epoch" not in train_log or "lr" not in train_log:
        return []
    work = train_log[["epoch", "lr"]].copy()
    work["epoch"] = pd.to_numeric(work["epoch"], errors="coerce")
    work["lr"] = pd.to_numeric(work["lr"], errors="coerce")
    work = work.dropna().sort_values("epoch")
    if work.empty:
        return []
    warnings = []
    by_epoch = {int(row.epoch): float(row.lr) for row in work.itertuples(index=False)}
    for event in resume_events:
        next_epoch = int(event.get("next_epoch", -1) or -1)
        prev_epoch = next_epoch - 1
        if prev_epoch in by_epoch and next_epoch in by_epoch:
            prev_lr = by_epoch[prev_epoch]
            next_lr = by_epoch[next_epoch]
            if prev_lr > 0 and next_lr / prev_lr > 2.0:
                warnings.append(f"LR jump after resume epoch {prev_epoch}->{next_epoch}: {prev_lr} -> {next_lr}")
    return warnings


def _best_reset_failures(train_log: pd.DataFrame) -> List[str]:
    if train_log.empty or "best_val_macro_f1" not in train_log:
        return []
    vals = pd.to_numeric(train_log["best_val_macro_f1"], errors="coerce").dropna().tolist()
    failures = []
    for prev, cur in zip(vals, vals[1:]):
        if cur + 1e-12 < prev:
            failures.append(f"best_val_macro_f1 reset/drop detected: {prev} -> {cur}")
            break
    return failures


def check_run(run_dir: Path, allow_resume_from_best: bool = False) -> Dict[str, Any]:
    ckpt_dir = run_dir / "checkpoints"
    last_path = ckpt_dir / "last.pt"
    best_path = ckpt_dir / "best.pt"
    train_log = _load_csv(run_dir / "train_log.csv")
    resume_events = _load_jsonl(run_dir / "resume_events.jsonl")
    last = _load_checkpoint(last_path)
    best = _load_checkpoint(best_path)
    failures: List[str] = []
    warnings: List[str] = []

    if not last_path.exists():
        failures.append("last.pt missing")
    if not best_path.exists():
        failures.append("best.pt missing")
    if not last:
        failures.append("last.pt could not be loaded")
    if best_path.exists() and not best:
        failures.append("best.pt could not be loaded")

    if last:
        missing = _checkpoint_missing_fields(last)
        if missing:
            failures.append("checkpoint missing fields: " + ", ".join(missing))
        if last.get("optimizer_state_dict") is None:
            failures.append("optimizer_state_dict missing")
        if last.get("scheduler_state_dict") is None:
            failures.append("scheduler_state_dict missing")
        if not last.get("rng_state"):
            failures.append("rng_state missing")
        if not last.get("early_stopping_state"):
            failures.append("early_stopping_state missing")
        if not bool(last.get("from_scratch", False)):
            failures.append("from_scratch is not true")
        if last.get("init_checkpoint") is not None:
            failures.append("init_checkpoint is not null")
        if bool(last.get("loaded_pretrained", False)):
            failures.append("loaded_pretrained is true")

    failures.extend(_epoch_sequence_failures(train_log))
    failures.extend(_best_reset_failures(train_log))
    warnings.extend(_lr_jump_warning(train_log, resume_events))

    for event in resume_events:
        source = str(event.get("resume_from", ""))
        if source.replace("\\", "/").endswith("/best.pt") or source.lower().endswith("best.pt"):
            msg = f"resume_from best.pt detected: {source}"
            if allow_resume_from_best or bool(event.get("allow_resume_from_best", False)):
                warnings.append("ALLOW_RESUME_FROM_BEST " + msg)
            else:
                failures.append(msg)
        if event.get("config_warnings"):
            for warning in event.get("config_warnings", []):
                if "ALLOW_BATCH_SIZE_CHANGE" in str(warning):
                    if bool(event.get("allow_batch_size_change", False)):
                        warnings.append(str(warning))
                    else:
                        failures.append(str(warning))

    if any("optimizer_state_dict missing" in item for item in failures):
        decision = "D15_RESUME_FAIL_MISSING_OPTIMIZER"
    elif any("checkpoint missing fields" in item or "from_scratch" in item or "init_checkpoint" in item for item in failures):
        decision = "D15_RESUME_FAIL_CONFIG_MISMATCH"
    elif any("epoch gap" in item or "duplicate epoch" in item for item in failures):
        decision = "D15_RESUME_FAIL_EPOCH_GAP"
    elif any("best.pt" in item for item in failures):
        decision = "D15_RESUME_FAIL_RESUME_FROM_BEST"
    elif any("BATCH_SIZE" in item or "batch_size" in item for item in failures):
        decision = "D15_RESUME_FAIL_CONFIG_MISMATCH"
    elif failures:
        decision = "D15_RESUME_FAIL_CONFIG_MISMATCH"
    elif warnings:
        decision = "D15_RESUME_WARN_LR_JUMP" if any("LR jump" in item for item in warnings) else "D15_RESUME_INTEGRITY_PASS"
    else:
        decision = "D15_RESUME_INTEGRITY_PASS"

    summary = {
        "run_dir": str(run_dir),
        "last_exists": last_path.exists(),
        "best_exists": best_path.exists(),
        "last_epoch": int(last.get("epoch", -1)) if last else -1,
        "global_step": int(last.get("global_step", -1)) if last else -1,
        "best_val_macro_f1": float(last.get("best_val_macro_f1", np.nan)) if last else float("nan"),
        "best_epoch": int(last.get("best_epoch", -1)) if last else -1,
        "from_scratch": bool(last.get("from_scratch", False)) if last else False,
        "init_checkpoint": last.get("init_checkpoint") if last else None,
        "loaded_pretrained": bool(last.get("loaded_pretrained", False)) if last else False,
        "resume_events": resume_events,
        "warnings": warnings,
        "failures": failures,
        "decision": decision,
    }
    return summary


def write_outputs(run_dir: Path, summary: Dict[str, Any]) -> None:
    (run_dir / "d15_resume_integrity_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    lines = [
        "# D15 Resume Integrity Report",
        "",
        f"- decision: `{summary['decision']}`",
        f"- last_epoch: {summary.get('last_epoch')}",
        f"- global_step: {summary.get('global_step')}",
        f"- best_val_macro_f1: {summary.get('best_val_macro_f1')}",
        f"- best_epoch: {summary.get('best_epoch')}",
        f"- from_scratch: {summary.get('from_scratch')}",
        f"- init_checkpoint: {summary.get('init_checkpoint')}",
        f"- loaded_pretrained: {summary.get('loaded_pretrained')}",
        "",
        "## Warnings",
        *([f"- {item}" for item in summary.get("warnings", [])] or ["- none"]),
        "",
        "## Failures",
        *([f"- {item}" for item in summary.get("failures", [])] or ["- none"]),
    ]
    (run_dir / "d15_resume_integrity_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--allow_resume_from_best", action="store_true", default=False)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    summary = check_run(run_dir, allow_resume_from_best=bool(args.allow_resume_from_best))
    write_outputs(run_dir, summary)
    print(json.dumps(summary, indent=2, default=str))
    if str(summary["decision"]).startswith("D15_RESUME_FAIL"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
