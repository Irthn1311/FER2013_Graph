"""Train the locked standalone candidate with resume forcibly disabled."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lap_gnn.cli.common import (
    add_data_arguments,
    refuse_output_collision,
    validate_inputs,
    write_runtime_config,
)
from lap_gnn.training import engine


def main() -> None:
    parser = argparse.ArgumentParser()
    add_data_arguments(parser, output=True)
    parser.add_argument("--no-resume", action="store_true", default=True)
    args = parser.parse_args()
    cfg, prior_root = validate_inputs(args)
    cfg["data"]["prior_dir"] = str(prior_root)
    cfg["data"]["num_workers"] = int(args.num_workers)
    cfg["training"]["num_workers"] = int(args.num_workers)
    cfg["logging"]["wandb"]["enabled"] = False
    output_dir = Path(args.output_root).resolve() / str(cfg["run_name"])
    refuse_output_collision(output_dir)
    runtime_config = write_runtime_config(cfg)
    cache_args = []
    if args.cache_root:
        cache_args = ["--graph_cache_dir", str(Path(args.cache_root).resolve())]
    original_argv = sys.argv
    try:
        sys.argv = [
            "lap-gnn-train", "--config", str(runtime_config),
            "--prior_dir", str(prior_root), "--output_dir", str(output_dir),
            "--device", args.device, "--num_workers", str(args.num_workers),
            "--resume_auto", "false", *cache_args,
        ]
        engine.main()
    finally:
        sys.argv = original_argv
        runtime_config.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
