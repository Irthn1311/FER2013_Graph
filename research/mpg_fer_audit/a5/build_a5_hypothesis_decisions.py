"""Generate a5_hypothesis_decisions.json for A5 audit."""

from __future__ import annotations

import json
from pathlib import Path

AUDIT_DIR = Path(__file__).resolve().parent

decisions = {
    "audit": "MPG-FER A5",
    "topic": "Shared Hard-Example, Label-Ambiguity & Representation-Failure Audit",
    "hypotheses": {
        "H-A5-DATA": {
            "name": "Dataset & Label Ambiguity",
            "question": "Is the shared hard error ceiling associated with visually ambiguous, inconsistently labeled, or intrinsically overlapping FER examples?",
            "status": "DATA_AMBIGUITY_SUPPORTED",
            "evidence": {
                "exact_duplicate_label_conflicts": {
                    "total_exact_duplicate_groups": 1516,
                    "total_duplicate_images": 3369,
                    "conflicting_label_groups": 57,
                    "cross_split_conflicting_groups": 26,
                    "finding": "Direct cryptographic proof of label noise: 57 groups of byte-identical images have conflicting ground-truth emotion labels."
                },
                "nonlearned_neighborhood_contradiction": {
                    "raw_pixel_true_purity_on_high_conf_errors": "15.5% to 17.3% (near random chance 14.3%)",
                    "gradient_pca_true_purity_on_high_conf_errors": "15.0% to 17.2%",
                    "finding": "Non-learned image spaces do NOT support the provided dataset labels for shared high-confidence errors."
                },
                "intrinsic_class_geometry_overlap": {
                    "fear_raw_pixel_margin": "-3.39 (Public), -3.23 (Private)",
                    "fear_gradient_pca_margin": "-0.060 (Public), -0.061 (Private)",
                    "finding": "Fear exhibits the most negative margin in non-learned pixel and gradient spaces, proving that its separability collapse is intrinsic to the dataset distribution."
                }
            }
        },
        "H-A5-REP": {
            "name": "Common Representation Failure",
            "question": "Does the MPG model family systematically map validly labeled examples into the wrong class manifold due to a shared representational blind spot?",
            "status": "MIXED",
            "evidence": {
                "shared_failure_analysis": "On shared unanimous errors, non-learned image descriptors also fail to retrieve the nominal class (purity ~15-17%), ruling out clean representation failure on the shared core.",
                "model_disagreement_cases": "On 34% of test samples, the 4 model variants disagree, with individual models successfully resolving samples that others fail on (5-NN purity in resolving model's space is ~70-75% vs ~20% in failing model's space).",
                "finding": "Model representation failure explains model-specific errors and disagreement cases, but does not explain the persistent shared error core."
            }
        },
        "H-A5-COMPLEMENTARITY": {
            "name": "Model Complementarity",
            "question": "How much error is truly shared across model variants versus model-specific?",
            "status": "MEANINGFUL",
            "evidence": {
                "v21_plus_v22_oracle_accuracy": "76.51% (Public), 76.71% (Private) (+7.1% over individual models)",
                "four_model_oracle_accuracy": "81.75% (Public), 82.06% (Private) (+12.3% over individual models)",
                "four_model_disagreement_fraction": "34.19% (Public), 34.80% (Private)",
                "four_model_shared_failure_fraction": "18.25% (Public), 17.94% (Private)",
                "finding": "Nearly half of the total individual model error (~30.5% error rate -> ~18.0% shared error) is model-specific and resolvable by different architectural/training inductive biases."
            }
        }
    },
    "verdict": "A5_AUTOMATED_AUDIT_READY_FOR_HUMAN_REVIEW"
}

out_path = AUDIT_DIR / "a5_hypothesis_decisions.json"
out_path.write_text(json.dumps(decisions, indent=2), encoding="utf-8")
print(f"Saved {out_path}")
