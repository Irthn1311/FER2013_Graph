"""Verify v2.2 dense equivalence and the preregistered A2-R sparse replay."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
V21_SOURCE = PROJECT_ROOT / "research" / "mpg_fer_v2_1" / "src"
V22_SOURCE = PROJECT_ROOT / "research" / "mpg_fer_v2_2" / "src"
sys.path.insert(0, str(V21_SOURCE))
sys.path.insert(0, str(V22_SOURCE))

from mpg_fer_v2_1.config import MPGConfig as MPGConfigV21  # noqa: E402
from mpg_fer_v2_1.model import MPGFER as MPGFERV21  # noqa: E402
from mpg_fer_v2_2.config import MPGConfig  # noqa: E402
from mpg_fer_v2_2.data import (  # noqa: E402
    FER2013Dataset,
    create_private_dataloader,
)
from mpg_fer_v2_2.evaluate import evaluate_raw_and_tta  # noqa: E402
from mpg_fer_v2_2.model import MPGFER  # noqa: E402
from mpg_fer_v2_2.train import source_tree_hash  # noqa: E402


LOCKED_CHECKPOINT_SHA = (
    "4720a482ff0f6da15a00dc168d7c551b4e9538b4c1ed8780ea891b69b97aeb75"
)
LOCKED_SCHEDULE = (8, 16, 16, 16, 24)
EXPECTED_REPLAY = {
    "public_tta_accuracy": 0.6926720534967957,
    "public_tta_macro_f1": 0.6707896323378159,
    "private_tta_accuracy": 0.6988018946781833,
    "private_tta_macro_f1": 0.6898134831099085,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parameter_manifest(model: torch.nn.Module) -> dict[str, int]:
    return {name: parameter.numel() for name, parameter in model.named_parameters()}


@torch.no_grad()
def _tie_diagnostics(
    model: MPGFER, images: torch.Tensor, device: torch.device
) -> dict[str, dict[str, int]]:
    results: dict[str, dict[str, int]] = {}
    for label, amp_enabled in (("fp32", False), ("amp_fp16", device.type == "cuda")):
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            _, outputs = model(images.to(device))
        results[label] = {
            f"layer_{layer_index}": int(
                outputs[f"motif_l{layer_index}_boundary_tie_count"].item()
            )
            for layer_index in range(1, 6)
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--public-csv", type=Path, required=True)
    parser.add_argument("--private-csv", type=Path, required=True)
    parser.add_argument("--allow-private-replay", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.allow_private_replay:
        raise RuntimeError(
            "Private replay requires the explicit --allow-private-replay flag"
        )

    checkpoint_sha = _sha256(args.checkpoint)
    if checkpoint_sha != LOCKED_CHECKPOINT_SHA:
        raise RuntimeError(
            f"v2.1 checkpoint SHA mismatch: {checkpoint_sha}"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = raw["model_state_dict"]

    v21_config = MPGConfigV21()
    v22_config = MPGConfig(motif_topk_schedule=LOCKED_SCHEDULE)
    v21 = MPGFERV21(v21_config).eval()
    dense = MPGFER(
        MPGConfig(motif_topk_schedule=(48, 48, 48, 48, 48))
    ).eval()
    sparse = MPGFER(v22_config).eval()
    v21.load_state_dict(state, strict=True)
    dense.load_state_dict(state, strict=True)
    sparse.load_state_dict(state, strict=True)

    torch.manual_seed(42)
    synthetic = torch.randn(4, 1, 48, 48)
    with torch.no_grad():
        reference_logits = v21(synthetic)[0]
        dense_logits = dense(synthetic)[0]
    dense_max_abs_error = float(
        (reference_logits - dense_logits).abs().max().item()
    )
    if dense_max_abs_error > 1e-6:
        raise RuntimeError(
            f"Dense equivalence failed: max_abs_error={dense_max_abs_error}"
        )

    v21_parameters = _parameter_manifest(v21)
    v22_parameters = _parameter_manifest(sparse)
    if v21_parameters != v22_parameters:
        raise RuntimeError("v2.1/v2.2 parameter manifests differ")

    v21_config_dict = asdict(v21_config)
    v22_config_dict = asdict(v22_config)
    schedule = v22_config_dict.pop("motif_topk_schedule")
    if v22_config_dict != v21_config_dict:
        raise RuntimeError("v2.2 config differs beyond motif_topk_schedule")

    public_dataset = FER2013Dataset(
        args.public_csv, split="val", augment=False
    )
    public_loader = DataLoader(
        public_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    private_loader = create_private_dataloader(
        args.private_csv, batch_size=args.batch_size, num_workers=0
    )
    sparse = sparse.to(device)
    sample_images = torch.stack(
        [public_dataset[index][0] for index in range(args.batch_size)]
    )
    tie_diagnostics = _tie_diagnostics(sparse, sample_images, device)

    started = time.perf_counter()
    public_metrics = evaluate_raw_and_tta(
        sparse, public_loader, device, use_amp=device.type == "cuda"
    )
    public_seconds = time.perf_counter() - started
    started = time.perf_counter()
    private_metrics = evaluate_raw_and_tta(
        sparse, private_loader, device, use_amp=device.type == "cuda"
    )
    private_seconds = time.perf_counter() - started

    actual = {
        "public_tta_accuracy": public_metrics["tta"]["accuracy"],
        "public_tta_macro_f1": public_metrics["tta"]["macro_f1"],
        "private_tta_accuracy": private_metrics["tta"]["accuracy"],
        "private_tta_macro_f1": private_metrics["tta"]["macro_f1"],
    }
    result = {
        "verification_scope": (
            "frozen v2.1 checkpoint implementation replay only; "
            "PrivateTest not used for model selection"
        ),
        "official_training_launched": False,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "source_hash": source_tree_hash(),
        "motif_topk_schedule": list(schedule),
        "resume_schema_version": v22_config.resume_schema_version,
        "parameter_count_v2_1": sum(v21_parameters.values()),
        "parameter_count_v2_2": sum(v22_parameters.values()),
        "parameter_delta": sum(v22_parameters.values()) - sum(v21_parameters.values()),
        "dense_equivalence_max_abs_error": dense_max_abs_error,
        "practical_cutoff_ties": tie_diagnostics,
        "public": public_metrics,
        "private": private_metrics,
        "a2r_expected": EXPECTED_REPLAY,
        "a2r_differences": {
            key: actual[key] - expected for key, expected in EXPECTED_REPLAY.items()
        },
        "public_evaluation_seconds": public_seconds,
        "private_evaluation_seconds": private_seconds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
