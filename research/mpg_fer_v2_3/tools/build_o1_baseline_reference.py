"""Build the O1 historical control from explicit public-only artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mpg_fer_o1.protocol import (  # noqa: E402
    BASELINE_CONFIG_SHA256,
    BASELINE_ID,
    FROZEN_SCIENTIFIC_SOURCE_SHA256,
    validate_baseline_reference,
    write_json,
)
from mpg_fer_v2_3.checkpoint import sha256_file  # noqa: E402


def build_reference(
    public_results_path: str | Path,
    history_summary_path: str | Path,
    checkpoint_path: str | Path,
) -> dict:
    public_path = Path(public_results_path)
    history_path = Path(history_summary_path)
    checkpoint = Path(checkpoint_path)
    for path in (public_path, history_path, checkpoint):
        if any(
            marker in part.lower()
            for part in path.parts
            for marker in ("privatetest", "private_test", "private-test")
        ):
            raise RuntimeError("Baseline builder refuses private-marked artifact paths")
        if not path.is_file():
            raise FileNotFoundError(path)
    public = json.loads(public_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    public_identity = public.get("identity", {})
    history_identity = history.get("identity", {})
    identity_fields = (
        "run_id",
        "source_sha256",
        "config_sha256",
        "best_epoch",
        "best_checkpoint_sha256",
        "completed_epoch",
    )
    if any(public_identity.get(key) != history_identity.get(key) for key in identity_fields):
        raise RuntimeError("Baseline public/history identity mismatch")
    checkpoint_sha = sha256_file(checkpoint)
    if (
        public_identity.get("source_sha256") != FROZEN_SCIENTIFIC_SOURCE_SHA256
        or public_identity.get("config_sha256") != BASELINE_CONFIG_SHA256
        or public_identity.get("best_checkpoint_sha256") != checkpoint_sha
        or history.get("best_epoch") != public_identity.get("best_epoch")
        or history.get("epoch_count") != public_identity.get("completed_epoch")
    ):
        raise RuntimeError("Baseline source/checkpoint/history provenance mismatch")

    def metrics(view: str) -> dict[str, float]:
        source = public.get(view)
        if not isinstance(source, dict):
            raise RuntimeError(f"Baseline public results lacks {view}")
        return {
            "loss": float(source["loss"]),
            "accuracy": float(source["accuracy"]),
            "macro_f1": float(source["macro_f1"]),
        }

    reference = {
        "schema_version": 1,
        "status": "VERIFIED",
        "usable_for_o1_aggregation": True,
        "config_id": BASELINE_ID,
        "frozen_scientific_source_sha256": FROZEN_SCIENTIFIC_SOURCE_SHA256,
        "seed": 42,
        "learning_rate": 3.0e-4,
        "lr_decay_end_epoch": 85,
        "checkpoint_provenance": {
            "run_id": public_identity["run_id"],
            "artifact_name": checkpoint.name,
            "public_results_sha256": sha256_file(public_path),
        },
        "checkpoint_sha256": checkpoint_sha,
        "history_provenance": {
            "run_id": history_identity["run_id"],
            "artifact_name": history_path.name,
            "epoch_count": history["epoch_count"],
            "first_epoch": history["first_epoch"],
            "last_epoch": history["last_epoch"],
        },
        "history_sha256": sha256_file(history_path),
        "selected_epoch": int(public_identity["best_epoch"]),
        "public_metrics": {"raw": metrics("raw"), "tta": metrics("tta")},
        "private_test_artifacts_read": False,
    }
    return validate_baseline_reference(reference)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-results", type=Path, required=True)
    parser.add_argument("--history-summary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "O1_BASELINE_REFERENCE.json"
    )
    args = parser.parse_args()
    write_json(
        args.output,
        build_reference(args.public_results, args.history_summary, args.checkpoint),
    )
    print(args.output)


if __name__ == "__main__":
    main()
