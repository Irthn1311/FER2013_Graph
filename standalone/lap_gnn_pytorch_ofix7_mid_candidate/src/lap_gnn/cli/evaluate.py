"""Evaluate a locked checkpoint without training."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from lap_gnn.cli.common import add_data_arguments, validate_inputs, write_runtime_config
from lap_gnn.training import engine


def main() -> None:
    parser = argparse.ArgumentParser()
    add_data_arguments(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["test"], default="test")
    parser.add_argument("--predictions-output")
    args = parser.parse_args()
    cfg, prior_root = validate_inputs(args)
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    cfg["data"]["prior_dir"] = str(prior_root)
    cfg["logging"]["wandb"]["enabled"] = False
    runtime_config = write_runtime_config(cfg)
    output_dir = (
        Path(args.predictions_output).resolve().parent
        if args.predictions_output
        else Path(tempfile.mkdtemp(prefix="lap_gnn_eval_"))
    )
    original_argv = sys.argv
    try:
        sys.argv = [
            "lap-gnn-evaluate", "--config", str(runtime_config),
            "--prior_dir", str(prior_root), "--output_dir", str(output_dir),
            "--device", args.device, "--num_workers", str(args.num_workers),
            "--eval_only", "--checkpoint", str(checkpoint),
            "--resume_auto", "false",
        ]
        engine.main()
    finally:
        sys.argv = original_argv
        runtime_config.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
