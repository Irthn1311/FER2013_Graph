"""Regenerate A6-R2 decision artifacts from the authoritative master JSON.

This module is deliberately derived-artifact-only.  It does not import either
model, load FER2013, or rerun any A6/A6-R/A6-R2 experiment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


AUDIT_DIR = Path(__file__).resolve().parent
MASTER_NAME = "a6r2_master_results.json"


def _percent(value: float) -> float:
    return round(100.0 * value, 6)


def _readout_evidence(master: dict[str, Any]) -> dict[str, Any]:
    repaired = master["readout_factorial_repaired"]
    result: dict[str, Any] = {}
    for direction in ("dir_A", "dir_B"):
        all_cohort = repaired[direction]["all"]
        result[direction] = {
            "sample_count": all_cohort["sample_count"],
            "f00_baseline_correct_count": all_cohort["f00_baseline"][
                "correct_count"
            ],
            "f01_readout_operator_only_rescue_rate": all_cohort[
                "f01_readout_operator_only"
            ]["correct_rate"],
            "f10_node_states_only_rescue_rate": all_cohort[
                "f10_node_states_only"
            ]["correct_rate"],
            "f11_node_states_and_readout_rescue_rate": all_cohort[
                "f11_both_swapped_s7"
            ]["correct_rate"],
        }
    return result


def _routing_evidence(master: dict[str, Any]) -> dict[str, Any]:
    layers = master["routing_sample_associations"]["layer_statistics"]
    jaccards: list[float] = []
    point_biserial: list[float] = []
    p_values: list[float] = []
    representation_spearman: list[float] = []
    by_layer: dict[str, Any] = {}
    for layer_number in range(1, 6):
        layer = layers[f"layer_{layer_number}"]
        jaccard = layer["per_sample_jaccard_distribution"]["mean"]
        directions: dict[str, Any] = {}
        for direction in ("v21_correct_v22_wrong", "v22_correct_v21_wrong"):
            values = layer["directions"][direction]
            pb = values["point_biserial_routing_divergence_vs_rescue"]
            rep = values[
                "spearman_routing_divergence_vs_representation_cosine_dist"
            ]
            point_biserial.append(pb["estimate"])
            p_values.append(pb["p_value"])
            representation_spearman.append(rep["estimate"])
            directions[direction] = {
                "point_biserial_vs_rescue": pb,
                "spearman_vs_representation_distance": rep,
            }
        jaccards.append(jaccard)
        by_layer[f"layer_{layer_number}"] = {
            "K": layer["K"],
            "mean_support_jaccard": jaccard,
            "directions": directions,
        }
    return {
        "sample_count": master["cohort_counts"]["model_resolvable_all"],
        "by_layer": by_layer,
        "mean_support_jaccard_range": [min(jaccards), max(jaccards)],
        "point_biserial_vs_rescue_range": [
            min(point_biserial),
            max(point_biserial),
        ],
        "minimum_point_biserial_p_value": min(p_values),
        "representation_distance_spearman_range": [
            min(representation_spearman),
            max(representation_spearman),
        ],
        "isolated_intervention_actionability": "NOT_ACTIONABLE",
    }


def _generalization_evidence(master: dict[str, Any]) -> dict[str, Any]:
    source = master["generalization_gap"]
    stages = ("R1", "R2", "R3", "R6", "R7", "R8")
    return {
        version: {
            stage: {
                "train_accuracy": source[version][stage]["train_accuracy"],
                "public_accuracy": source[version][stage]["public_accuracy"],
                "train_public_gap": source[version][stage]["train_public_gap"],
            }
            for stage in stages
        }
        for version in ("v21", "v22")
    }


def build_hypothesis_document(master: dict[str, Any]) -> dict[str, Any]:
    conclusions = master["master_conclusions"]
    composer = master["composer_functional_swaps"]
    readout = _readout_evidence(master)
    routing = _routing_evidence(master)
    generalization = _generalization_evidence(master)

    f10_a = readout["dir_A"]["f10_node_states_only_rescue_rate"]
    f10_b = readout["dir_B"]["f10_node_states_only_rescue_rate"]
    pb_min, pb_max = routing["point_biserial_vs_rescue_range"]
    min_p = routing["minimum_point_biserial_p_value"]
    v21 = generalization["v21"]
    return {
        "audit": "MPG-FER A6-R2",
        "source_of_truth": MASTER_NAME,
        "hypotheses": {
            "H-A6R2-COMPOSER": {
                "status": conclusions["h_a6r2_composer"],
                "evidence": composer,
                "finding": (
                    "WHAT-only rescue is "
                    f"{_percent(composer['v21_to_v22']['what_only_rescue_rate']):.2f}% "
                    "in direction A and "
                    f"{_percent(composer['v22_to_v21']['what_only_rescue_rate']):.2f}% "
                    "in direction B; TYPE is not the primary transferable channel."
                ),
            },
            "H-A6R2-READOUT": {
                "status": conclusions["h_a6r2_readout"],
                "evidence": readout,
                "finding": (
                    "Node states alone rescue "
                    f"{_percent(f10_a):.2f}% in direction A and "
                    f"{_percent(f10_b):.2f}% in direction B; the readout operator "
                    "is secondary but material."
                ),
            },
            "H-A6R2-ROUTING": {
                "status": conclusions["h_a6r2_routing"],
                "evidence": routing,
                "finding": (
                    "True per-sample routing divergence does not predict rescue: "
                    f"point-biserial r ranges from {pb_min:+.3f} to {pb_max:+.3f}, "
                    f"with every p-value at least {min_p:.3f}. Routing is an "
                    "integrated substrate, not an actionable isolated bottleneck."
                ),
            },
            "H-A6R2-GENERALIZATION": {
                "status": conclusions["h_a6r2_generalization"],
                "evidence": generalization,
                "finding": (
                    "For v2.1, the Train-Public gap expands from "
                    f"{_percent(v21['R1']['train_public_gap']):.2f} pp at PRE to "
                    f"{_percent(v21['R2']['train_public_gap']):.2f} pp at L1 and "
                    f"{_percent(v21['R3']['train_public_gap']):.2f} pp at L2. "
                    "v2.2 has the same depth-dependent pattern."
                ),
            },
            "H-A6R2-SINGLE-TARGET": {
                "status": conclusions["h_a6r2_single_target"],
                "evidence": {
                    "final_target_class": conclusions["final_target_class"],
                    "final_verdict": conclusions["final_verdict"],
                },
                "finding": (
                    "Repaired evidence justifies one narrow target class: "
                    "EARLY_DEPTH_GENERALIZATION_TARGET."
                ),
            },
        },
    }


def build_target_document(master: dict[str, Any]) -> dict[str, Any]:
    conclusions = master["master_conclusions"]
    composer = master["composer_functional_swaps"]
    readout = _readout_evidence(master)
    routing = _routing_evidence(master)
    generalization = _generalization_evidence(master)
    f10_a = readout["dir_A"]["f10_node_states_only_rescue_rate"]
    f10_b = readout["dir_B"]["f10_node_states_only_rescue_rate"]
    pb_min, pb_max = routing["point_biserial_vs_rescue_range"]
    return {
        "source_of_truth": MASTER_NAME,
        "final_mechanistic_decision": conclusions["final_target_class"],
        "verdict": conclusions["final_verdict"],
        "evidence": {
            "composer_functional_swaps": composer,
            "readout_factorial": readout,
            "routing_associations": routing,
            "generalization_gap": generalization,
        },
        "justification": [
            (
                "Composer functional transfer is WHAT-dominant: WHAT-only "
                f"rescues {_percent(composer['v21_to_v22']['what_only_rescue_rate']):.2f}% "
                "in direction A and "
                f"{_percent(composer['v22_to_v21']['what_only_rescue_rate']):.2f}% "
                "in direction B, transmitting contextual Pixel-GNN information."
            ),
            (
                "Readout transfer is node-state dominant: node states alone rescue "
                f"{_percent(f10_a):.2f}% in direction A and "
                f"{_percent(f10_b):.2f}% in direction B."
            ),
            (
                "The v2.1 Train-Public gap expands from "
                f"{_percent(generalization['v21']['R1']['train_public_gap']):.2f} pp "
                "at PRE to "
                f"{_percent(generalization['v21']['R3']['train_public_gap']):.2f} pp "
                "at L2; v2.2 shows the same early-depth pattern."
            ),
            (
                "Routing divergence is not an isolated rescue lever: point-biserial "
                f"r ranges from {pb_min:+.3f} to {pb_max:+.3f}, and all associations "
                "are non-significant."
            ),
            (
                "The v2.3 intervention must therefore target representation "
                "preservation across Motif Layers 1-2 without changing routing, "
                "TYPE, readout, or classifier design."
            ),
        ],
        "strictly_deprioritized_interventions": [
            "Graph topology density tuning or Top-K schedules.",
            "Classifier or motif-readout redesign as the primary intervention.",
            "Prototype-count, TYPE-projection, or isolated TYPE-loss tuning.",
        ],
    }


def write_derived_artifacts(audit_dir: Path = AUDIT_DIR) -> None:
    master = json.loads((audit_dir / MASTER_NAME).read_text(encoding="utf-8"))
    documents = {
        "a6r2_hypothesis_decisions.json": build_hypothesis_document(master),
        "a6r2_final_target.json": build_target_document(master),
    }
    for name, document in documents.items():
        (audit_dir / name).write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Wrote {audit_dir / name}")


if __name__ == "__main__":
    write_derived_artifacts()
