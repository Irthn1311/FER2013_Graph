"""Research training harness scaffold for Pure-GNN v3.1.

SCIENTIFIC TRAINING IS NOT AUTHORIZED IN THIS TASK.
This CLI exists strictly for pipeline scaffolding and preflight integration.
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Pure-GNN v3.1 Research Training Harness")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--condition", type=str, default="G1", choices=["G0", "G0.5", "G1", "G2", "G3"])
    parser.add_argument("--execute-scientific-training", action="store_true", default=False)
    args = parser.parse_args()

    if not args.execute_scientific_training:
        print("[pure_gnn_v31] Training scaffold invoked in preflight mode.")
        print("[pure_gnn_v31] Scientific training is NOT authorized in this task.")
        print("[pure_gnn_v31] Pipeline configuration parsed successfully.")
        sys.exit(0)
    else:
        raise PermissionError(
            "Scientific training is NOT authorized. "
            "Pure-GNN v3.1 is currently in implementation and technical preflight phase only."
        )


if __name__ == "__main__":
    main()
