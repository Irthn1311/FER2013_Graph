"""Run one-shot canonical FP32 PrivateTest evaluation on frozen checkpoints for A0..A5."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "mpg_fer_v2_3" / "src"))

from mpg_fer_cumulative_ablation7.model import (
    CUMULATIVE_ABLATION_ORDER,
    CUMULATIVE_REGISTRY,
    CumulativeAblationMode,
    CumulativeAblationMPGFER,
)
from mpg_fer_cumulative_ablation7.protocol import (
    CANONICAL_DATASET_HASHES,
    CANONICAL_DATASET_ROWS,
    EXPECTED_A6_METRICS,
    sha256_file,
    validate_split_identity,
)
from mpg_fer_v2_3.config import MPGConfig
from mpg_fer_v2_3.data import create_private_dataloader

ANALYSIS_DIR = ROOT / "analysis" / "mpg_fer_cumulative_ablation7"
RUNS_DIR = ANALYSIS_DIR / "runs"
DATA_DIR = Path(r"D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5\data")
TEST_CSV = DATA_DIR / "test.csv"
VAL_CSV = DATA_DIR / "val.csv"


def evaluate_model_canonical_fp32(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device):
    model.eval()
    was_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32 = False

    raw_preds, tta_preds, all_targets = [], [], []
    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            images = images.to(device)
            logits_orig, _ = model(images)
            logits_flip, _ = model(TF.hflip(images))
            logits_tta = (logits_orig + logits_flip) / 2.0

            raw_preds.append(logits_orig.argmax(dim=-1).cpu())
            tta_preds.append(logits_tta.argmax(dim=-1).cpu())
            all_targets.append(targets.cpu())
            if (i + 1) % 50 == 0:
                print(f"  Processed { (i + 1) * len(targets) } / 3589 samples...")

    raw_preds = torch.cat(raw_preds)
    tta_preds = torch.cat(tta_preds)
    all_targets = torch.cat(all_targets)

    raw_acc = float(accuracy_score(all_targets, raw_preds))
    raw_f1 = float(f1_score(all_targets, raw_preds, average="macro"))
    tta_acc = float(accuracy_score(all_targets, tta_preds))
    tta_f1 = float(f1_score(all_targets, tta_preds, average="macro"))

    torch.backends.cuda.matmul.allow_tf32 = was_tf32
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = cudnn_tf32

    return {
        "raw": {"accuracy": raw_acc, "macro_f1": raw_f1},
        "tta": {"accuracy": tta_acc, "macro_f1": tta_f1},
        "sample_count": len(all_targets),
    }


def main():
    print("1. Validating dataset split identities...")
    val_audit = validate_split_identity("val.csv", VAL_CSV)
    test_audit = validate_split_identity("test.csv", TEST_CSV)
    print("  Val audit:", val_audit)
    print("  Test audit:", test_audit)

    device = torch.device("cpu")
    print(f"Using device: {device} for canonical FP32 evaluation")

    test_loader = create_private_dataloader(TEST_CSV, batch_size=32, num_workers=0)

    for mode_enum in CUMULATIVE_ABLATION_ORDER:
        mode = mode_enum.value
        if mode == "A6":
            print(f"[{mode}] Using canonical frozen FULL reference metrics.")
            continue

        out_res = ANALYSIS_DIR / f"{mode}_RESULT.json"
        if out_res.is_file():
            print(f"[{mode}] Result already exists at {out_res}, skipping.")
            continue

        print(f"\n================ EVALUATING [{mode}] ================")
        mode_dir = RUNS_DIR / mode
        ckpt_files = list(mode_dir.rglob("best_val_acc.pt"))
        if not ckpt_files:
            raise FileNotFoundError(f"best_val_acc.pt missing for {mode}")
        ckpt_path = ckpt_files[0]
        ckpt_sha = sha256_file(ckpt_path)
        print(f"[{mode}] Checkpoint path: {ckpt_path}")
        print(f"[{mode}] Checkpoint SHA256: {ckpt_sha}")

        # Check recorded public metrics
        pub_files = list(mode_dir.rglob("canonical_public_metrics.json"))
        pub_data = json.load(open(pub_files[0])) if pub_files else {}
        selected_epoch = pub_data.get("selected_epoch")
        print(f"[{mode}] Selected epoch: {selected_epoch}")

        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = payload.get("model_state_dict", payload)

        model = CumulativeAblationMPGFER(config=MPGConfig(), mode=mode_enum)
        incompat = model.load_state_dict(state_dict, strict=True)
        assert len(incompat.missing_keys) == 0 and len(incompat.unexpected_keys) == 0, f"Strict load failed: {incompat}"
        model.to(device)

        print(f"[{mode}] Running canonical FP32 evaluation on 3,589 PrivateTest samples...")
        t0 = time.monotonic()
        test_metrics = evaluate_model_canonical_fp32(model, test_loader, device)
        elapsed = time.monotonic() - t0
        print(f"[{mode}] Completed in {elapsed:.1f}s")
        print(f"[{mode}] Raw Acc: {test_metrics['raw']['accuracy']*100:.4f}% | Raw F1: {test_metrics['raw']['macro_f1']*100:.4f}%")
        print(f"[{mode}] TTA Acc: {test_metrics['tta']['accuracy']*100:.4f}% | TTA F1: {test_metrics['tta']['macro_f1']*100:.4f}%")

        # Save result JSON
        spec = CUMULATIVE_REGISTRY[mode_enum]
        result_dict = {
            "schema_version": 1,
            "configuration": mode,
            "display_name": spec.paper_name,
            "vietnamese_name": spec.vietnamese_name,
            "status": "TRAINING_COMPLETED",
            "seed": 42,
            "checkpoint_sha256": ckpt_sha,
            "selected_epoch": selected_epoch,
            "weights_type": "EMA",
            "checkpoint_selection": "EMA Public horizontal-flip TTA accuracy, then higher Macro-F1, then lower CE loss",
            "dataset_hashes": {
                "train.csv": CANONICAL_DATASET_HASHES["train.csv"],
                "val.csv": CANONICAL_DATASET_HASHES["val.csv"],
                "test.csv": CANONICAL_DATASET_HASHES["test.csv"],
            },
            "public_metrics": pub_data.get("metrics", {}),
            "private_metrics": test_metrics,
            "canonical_evaluation": {
                "inference": "canonical_fp32_no_autocast_tf32_disabled",
                "sample_count": 3589,
                "evaluated_only_after_freeze": True,
            },
        }

        out_res = ANALYSIS_DIR / f"{mode}_RESULT.json"
        with open(out_res, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2)
        print(f"Wrote {out_res}")

    print("\nAll individual results ready. Generating summary tables...")
    from collect_cumulative_results import generate_summary_tables
    generate_summary_tables()
    print("Summary tables generated successfully!")


if __name__ == "__main__":
    main()
