"""Inspects architectural contracts, parameter counts, and gate budgets for Pure-GNN v3.1."""

import argparse
import json
import sys
from pathlib import Path
import tensorflow as tf

from pure_gnn_v31.model import PureGNNv31
from pure_gnn_v31.contracts import count_parameters, verify_parameter_budget, audit_forbidden_layers


def inspect(condition: str = "G1") -> dict:
    model = PureGNNv31(condition=condition)
    dummy = tf.zeros([1, 48, 48, 1], dtype=tf.float32)
    _ = model(dummy, training=False)

    passed, param_info = verify_parameter_budget(model, max_share=0.005)
    violations = audit_forbidden_layers(model)

    res = {
        "model_name": "PureGNNv31",
        "condition": condition,
        "parameters": param_info,
        "parameter_budget_pass": passed,
        "forbidden_layer_violations": violations,
        "forbidden_audit_pass": (len(violations) == 0),
    }
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Pure-GNN v3.1 contracts")
    parser.add_argument("--condition", type=str, default="G1", choices=["G0", "G0.5", "G1", "G2", "G3"])
    args = parser.parse_args()

    result = inspect(args.condition)
    print(json.dumps(result, indent=2))
    if not result["parameter_budget_pass"] or not result["forbidden_audit_pass"]:
        sys.exit(1)
