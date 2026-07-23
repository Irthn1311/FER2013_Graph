"""Inspect FER CSV and precomputed-prior compatibility."""

from __future__ import annotations

import argparse
import json

from lap_gnn.cli.common import add_data_arguments, validate_inputs
from lap_gnn.data.fer2013 import inspect_fer_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    add_data_arguments(parser)
    args = parser.parse_args()
    cfg, prior_root = validate_inputs(args)
    print(json.dumps({
        "fer_csv": inspect_fer_csv(args.fer_csv),
        "prior_root": str(prior_root),
        "run_name": cfg["run_name"],
        "seed": cfg["seed"],
    }, indent=2))


if __name__ == "__main__":
    main()
