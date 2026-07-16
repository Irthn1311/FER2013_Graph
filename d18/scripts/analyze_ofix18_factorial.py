"""Aggregate completed C0/C1/C2/C3 OFIX18 factorial audits.

This script is post-training only. It refuses partial tables unless
--allow_missing is supplied and labels all effects as seed42 diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch

CELLS = ("C0", "C1", "C2", "C3")
CHECKPOINTS = ("best", "last")
METRIC_KEYS = (
    "official_macro_f1",
    "remove_structure_macro_f1",
    "shuffle_structure_macro_f1",
    "robust_min",
    "official_to_remove_drop",
    "train_val_macro_gap_pp",
)


def checkpoint_epoch(run_dir: Path, checkpoint: str) -> int:
    payload = torch.load(run_dir / "checkpoints" / f"{checkpoint}.pt", map_location="cpu", weights_only=False)
    return int(payload.get("epoch", -1))


def training_point(run_dir: Path, epoch: int) -> tuple[float, float, float]:
    frame = pd.read_csv(run_dir / "train_log.csv")
    point = frame[frame["epoch"] == epoch]
    if point.empty:
        raise RuntimeError(f"epoch {epoch} missing from {run_dir / 'train_log.csv'}")
    row = point.iloc[-1]
    train = float(row["train_macro_f1"])
    val = float(row["val_macro_f1"])
    return train, val, 100.0 * (train - val)


def collect(cell: str, run_dir: Path, eval_root: Path, checkpoint: str) -> dict[str, Any]:
    metrics_path = eval_root / checkpoint / "counterfactual_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    metrics = pd.read_csv(metrics_path).set_index("mode")
    required = {
        "official", "remove_structure", "shuffle_structure",
        "permute_structure_destinations", "degree_matched_random_structure",
    }
    missing = required - set(metrics.index)
    if missing:
        raise RuntimeError(f"{metrics_path} missing modes {sorted(missing)}")
    epoch = checkpoint_epoch(run_dir, checkpoint)
    train_macro, val_macro, gap = training_point(run_dir, epoch)
    official = float(metrics.loc["official", "macro_f1"])
    robust_modes = [
        float(metrics.loc["remove_structure", "macro_f1"]),
        float(metrics.loc["shuffle_structure", "macro_f1"]),
        float(metrics.loc["permute_structure_destinations", "macro_f1"]),
        float(metrics.loc["degree_matched_random_structure", "macro_f1"]),
    ]
    return {
        "cell": cell,
        "checkpoint": checkpoint,
        "run_name": run_dir.name,
        "official_macro_f1": official,
        "remove_structure_macro_f1": robust_modes[0],
        "shuffle_structure_macro_f1": robust_modes[1],
        "permuted_structure_macro_f1": robust_modes[2],
        "degree_matched_random_macro_f1": robust_modes[3],
        "robust_min": min(robust_modes),
        "robust_avg": sum(robust_modes) / len(robust_modes),
        "official_to_remove_drop": official - robust_modes[0],
        "official_to_shuffle_drop": official - robust_modes[1],
        "train_macro_f1_at_selected_epoch": train_macro,
        "validation_macro_f1": val_macro,
        "train_val_macro_gap_pp": gap,
        "selected_epoch": epoch,
    }


def contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for checkpoint in CHECKPOINTS:
        subset = frame[frame.checkpoint == checkpoint].set_index("cell")
        if set(subset.index) != set(CELLS):
            raise RuntimeError(f"incomplete {checkpoint} factorial cells: {sorted(subset.index)}")
        for metric in METRIC_KEYS:
            y00 = float(subset.loc["C0", metric])
            y10 = float(subset.loc["C1", metric])
            y01 = float(subset.loc["C2", metric])
            y11 = float(subset.loc["C3", metric])
            rows += [
                {
                    "checkpoint": checkpoint,
                    "metric": metric,
                    "contrast": "structure_dropedge_main_effect",
                    "value": 0.5 * ((y10 - y00) + (y11 - y01)),
                    "scope": "seed42_diagnostic",
                },
                {
                    "checkpoint": checkpoint,
                    "metric": metric,
                    "contrast": "mode_mix_main_effect",
                    "value": 0.5 * ((y01 - y00) + (y11 - y10)),
                    "scope": "seed42_diagnostic",
                },
                {
                    "checkpoint": checkpoint,
                    "metric": metric,
                    "contrast": "interaction",
                    "value": y11 - y10 - y01 + y00,
                    "scope": "seed42_diagnostic",
                },
            ]
    return pd.DataFrame(rows)


def decision_markdown(frame: pd.DataFrame | None) -> str:
    lines = [
        "# OFIX18 Factorial Decision Template",
        "",
        "Status: NOT EVALUATED until C0/C1/C2 and the matched C3 audit are complete.",
        "",
        "This template must be interpreted separately for best and last checkpoints. Test robustness is not a checkpoint-selection criterion.",
        "",
        "## Predefined Rules",
        "",
        "- Mode mixing support: C2 versus C0 official macro-F1 loss <= 2.5 pp, remove-structure gain >= 8 pp, shuffle-structure gain >= 6 pp, with gains not explained only by checkpoint selection.",
        "- Structure DropEdge support: C1 versus C0 gives comparable robustness improvement while preserving official performance.",
        "- Positive interaction: C3 exceeds C1 and C2 in robust_min while losing <= 1.5 pp official macro-F1 versus the better of C1/C2.",
        "- Ceiling: C1/C2 mainly trade official performance for robustness, C3 has no meaningful positive interaction, semantic structure gain remains small, and no cell jointly reaches acceptable official and robust performance.",
        "",
        "No significance claim is permitted from seed42 alone.",
    ]
    if frame is not None:
        lines += ["", "## Filled Values", "", "Run this script on complete artifacts to populate the CSV tables; interpret against the fixed rules above without changing thresholds."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    for cell in CELLS:
        parser.add_argument(f"--{cell.lower()}_run_dir", required=True)
        parser.add_argument(f"--{cell.lower()}_eval_root", required=True)
    parser.add_argument("--output_dir", default="outputs/d18_analysis/ofix18_factorial_results")
    parser.add_argument("--allow_missing", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, missing = [], []
    for cell in CELLS:
        run_dir = Path(getattr(args, f"{cell.lower()}_run_dir"))
        eval_root = Path(getattr(args, f"{cell.lower()}_eval_root"))
        for checkpoint in CHECKPOINTS:
            try:
                rows.append(collect(cell, run_dir, eval_root, checkpoint))
            except (FileNotFoundError, RuntimeError) as exc:
                missing.append(f"{cell}/{checkpoint}: {exc}")
    if missing and not args.allow_missing:
        raise RuntimeError("factorial artifacts incomplete:\n" + "\n".join(missing))
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "factorial_results.csv", index=False)
    if not missing:
        contrast_frame = contrasts(frame)
        contrast_frame.to_csv(output / "factorial_contrasts.csv", index=False)
    else:
        pd.DataFrame(columns=["checkpoint", "metric", "contrast", "value", "scope"]).to_csv(
            output / "factorial_contrasts.csv", index=False
        )
    (output / "decision_template.md").write_text(decision_markdown(frame if not missing else None), encoding="utf-8")
    manifest = {
        "status": "COMPLETE" if not missing else "INCOMPLETE",
        "scope": "seed42_diagnostic",
        "missing": missing,
        "rows": len(frame),
        "test_used_for_selection": False,
    }
    (output / "factorial_analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
