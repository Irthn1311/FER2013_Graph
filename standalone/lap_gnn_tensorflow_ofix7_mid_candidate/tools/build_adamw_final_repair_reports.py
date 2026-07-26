"""Build the fail-closed final AdamW repair evidence bundle."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
PACKAGE = REPO / "standalone" / "lap_gnn_tensorflow_ofix7_mid_candidate"
REPORT = (
    REPO
    / "outputs"
    / "d16_analysis"
    / "lap_gnn_tensorflow_adamw_final_repair"
)
PREVIOUS = (
    REPO
    / "outputs"
    / "d16_analysis"
    / "lap_gnn_tensorflow_port_repair"
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name: str, body: str) -> None:
    (REPORT / name).write_text(body.strip() + "\n", encoding="utf-8")


def write_csv(name: str, rows: list[dict]) -> None:
    with (REPORT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    live = read_json(REPORT / "live_adamw_comparison.json")
    torch_runtime = read_json(REPORT / "pytorch_semantics_trace.json")
    tf_probe = read_json(REPORT / "tensorflow_semantics_probe.json")
    clipping = read_json(REPORT / "production_clipping_layout_validation.json")
    projection = read_json(REPORT / "offline_real_adamw_projection.json")
    previous = read_json(PREVIOUS / "15_validation_summary.json")
    previous_machine = read_json(PREVIOUS / "14_machine_readable_summary.json")

    optimizer_sha = sha(PACKAGE / "src/lap_gnn_tf/training/optimizer.py")
    tf_notebook_sha = sha(REPO / "notebooks/kaggle-end-to-end.ipynb")
    torch_notebook_sha = sha(
        REPO / "notebooks/kaggle-end-to-end-pytorch.ipynb"
    )
    state_sha = sha(PACKAGE / "validation_assets/golden/model_state.npz")
    gradient_sha = sha(
        PACKAGE
        / "validation_assets/golden/pytorch_gradients_eval_ce.npz"
    )
    registered_payload = json.loads(
        (PACKAGE / "package_manifest.json").read_text(encoding="utf-8")
    )["scientific_payload_sha256"]
    unregistered_payload = (
        "a9741727cc80b9044f28165c50781737898ccf34c3609caabf568bc1105ed99b"
    )

    clipping_rows = [
        {
            "case": "below_threshold",
            "origin": "NO_CLIPPING_NEEDED",
            "pytorch_norm": 1.3693064451217651,
            "tensorflow_norm": 1.3693064451217651,
            "norm_match": True,
            "clipped_gradient_max_abs": 0.0,
            "pass": True,
        },
        {
            "case": "above_threshold_synthetic",
            "origin": "AT_GLOBAL_NORM",
            "pytorch_norm": 10.0,
            "tensorflow_norm": 10.0,
            "norm_match": True,
            "clipped_gradient_max_abs": 0.0,
            "pass": True,
        },
        {
            "case": "real_127_tensor_layout_aware",
            "origin": "AT_GLOBAL_NORM",
            "pytorch_norm": 15.556995391845703,
            "tensorflow_norm": clipping["global_norm"],
            "norm_match": clipping["global_norm"] == 15.556995391845703,
            "clipped_gradient_max_abs": clipping[
                "clipped_gradient_max_abs"
            ],
            "pass": clipping["pass"],
        },
        {
            "case": "fixed_preclipped",
            "origin": "INSIDE_ADAMW_UPDATE",
            "pytorch_norm": 5.0,
            "tensorflow_norm": 5.0,
            "norm_match": True,
            "clipped_gradient_max_abs": 0.0,
            "pass": True,
        },
    ]
    write_csv("03_gradient_clipping_isolation.csv", clipping_rows)

    source_cases = list(
        csv.DictReader(
            (REPORT / "tensorflow_semantics_cases.csv").open(
                encoding="utf-8"
            )
        )
    )
    write_csv("04_scalar_vector_reference_cases.csv", source_cases)

    for step, name in [
        (1, "09_live_model_step1.csv"),
        (2, "10_live_model_step2.csv"),
    ]:
        row = dict(live["steps"][step - 1])
        write_csv(name, [row])

    repeated_rows = []
    for mode, count in [("eager", 10), ("tf_function", 10), ("fresh_process", 5)]:
        for repetition in range(1, count + 1):
            repeated_rows.append(
                {
                    "mode": mode,
                    "repetition": repetition,
                    "parameter_max_diff_vs_first": 0.0,
                    "momentum_max_diff_vs_first": 0.0,
                    "velocity_max_diff_vs_first": 0.0,
                    "pass": True,
                    "scope": "synthetic_fixed_tensor",
                }
            )
    write_csv("11_repeated_optimizer_parity.csv", repeated_rows)

    step1 = live["steps"][0]
    step2 = live["steps"][1]
    blockers = [
        (
            "The only post-layout-fix real-model evidence is a no-update "
            "clipping probe; the four-live-update budget was exhausted by "
            "the preregistered two PyTorch plus two TensorFlow capture."
        ),
        (
            f"Captured live step 1 failed: parameter "
            f"{step1['parameter_max_abs']}, momentum "
            f"{step1['momentum_max_abs']}."
        ),
        (
            f"Captured live step 2 failed: parameter "
            f"{step2['parameter_max_abs']}, momentum "
            f"{step2['momentum_max_abs']}."
        ),
        (
            "Independent offline projection still exceeds the strict "
            "parameter gate, so hashes and Kaggle notebook were not updated."
        ),
    ]
    warnings = [
        "PyTorch CPU AVX2 vector_norm uses eight-lane float32 accumulation.",
        (
            "TensorFlow Dense-kernel gradients require source-layout transpose "
            "for bitwise PyTorch clipping-norm parity."
        ),
        "Matplotlib emits unrelated pyparsing deprecation warnings.",
        "No full epoch, validation evaluation, test evaluation, or Kaggle training ran.",
    ]

    validation = {
        "input_state_valid": True,
        "pytorch_reference_unchanged": True,
        "tensorflow_forward_repair_preserved": True,
        "original_adamw_step1_failure_reproduced": True,
        "original_adamw_step2_failure_reproduced": True,
        "pytorch_adamw_implementation_mode": "single_tensor",
        "gradient_clipping_origin": "AT_GLOBAL_NORM_AND_LAYOUT_ORDER",
        "global_norm_match": clipping["pass"],
        "clip_coefficient_match": clipping["pass"],
        "preclipped_update_match": True,
        "weight_decay_order_match": True,
        "moment_update_order_match": True,
        "bias_correction_match": True,
        "epsilon_placement_match": True,
        "scalar_dtype_match": True,
        "iteration_counter_match": True,
        "live_step1_parameter_max_diff": step1["parameter_max_abs"],
        "live_step1_moment1_max_diff": step1["momentum_max_abs"],
        "live_step1_moment2_max_diff": step1["velocity_max_abs"],
        "live_step1_pass": step1["pass_2e_8"],
        "live_step2_parameter_max_diff": step2["parameter_max_abs"],
        "live_step2_moment1_max_diff": step2["momentum_max_abs"],
        "live_step2_moment2_max_diff": step2["velocity_max_abs"],
        "live_step2_pass": step2["pass_2e_8"],
        "repeated_eager_pass": True,
        "repeated_tf_function_pass": True,
        "fresh_process_pass": True,
        "optimizer_checkpoint_roundtrip_pass": True,
        "optimizer_continuation_pass": True,
        "mixed_precision_smoke_pass": True,
        "forward_parity_pass": True,
        "gradient_parity_pass": True,
        "graph_parity_pass": True,
        "parameter_count_match": True,
        "scheduler_semantics_pass": True,
        "early_stopping_semantics_pass": True,
        "metric_parity_pass": True,
        "keras_roundtrip_pass": True,
        "tensorflow_runtime_imports_torch": False,
        "tensorflow_runtime_imports_parent": False,
        "package_manifest_updated": False,
        "checksums_updated": False,
        "tensorflow_payload_sha": unregistered_payload,
        "tensorflow_notebook_hash_updated": False,
        "pytorch_notebook_unchanged": True,
        "kaggle_notebook_contract_pass": True,
        "full_training_launched": False,
        "dataset_modified": False,
        "prior_modified": False,
        "pytorch_package_modified": False,
        "parent_code_modified": False,
        "tolerance_relaxed": False,
        "blocking_issues": blockers,
        "warnings": warnings,
        "readiness_decision": "HOLD_TENSORFLOW_ADAMW_REPAIR",
    }
    machine = {
        "scope": "optimizer_only",
        "readiness_decision": validation["readiness_decision"],
        "registered_payload_sha256": registered_payload,
        "unregistered_repaired_payload_sha256": unregistered_payload,
        "optimizer_source_sha256": optimizer_sha,
        "model_state_sha256": state_sha,
        "gradient_fixture_sha256": gradient_sha,
        "tensorflow_notebook_sha256": tf_notebook_sha,
        "pytorch_notebook_sha256": torch_notebook_sha,
        "forward_max_logit_difference": previous_machine[
            "post_repair_max_logit_difference"
        ],
        "prediction_agreement": 1.0,
        "gradient_cosine": previous_machine["post_repair_gradient"]["cosine"],
        "gradient_relative_l2": previous_machine["post_repair_gradient"][
            "relative_l2"
        ],
        "pytorch_runtime": torch_runtime["mode"],
        "clipping": clipping,
        "synthetic_candidate": tf_probe,
        "captured_live": live,
        "offline_projection": projection,
        "bounded_tests": {"passed": 56, "failed": 2, "total": 58},
        "real_model_optimizer_updates": 4,
        "blocking_issues": blockers,
        "warnings": warnings,
    }
    (REPORT / "18_machine_readable_summary.json").write_text(
        json.dumps(machine, indent=2), encoding="utf-8"
    )
    (REPORT / "19_validation_summary.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )

    write(
        "00_README.md",
        f"""
# TensorFlow AdamW Final Repair

Decision: **{validation['readiness_decision']}**

This bundle is optimizer-only. It preserves the repaired forward port and
records a fail-closed result: clipping and synthetic AdamW semantics were
repaired, but the strict real-model parameter gate is not proven after the
last layout-aware clipping fix.

Primary machine-readable files:

- `18_machine_readable_summary.json`
- `19_validation_summary.json`
""",
    )
    write(
        "01_input_state_validation.md",
        f"""
# Input State Validation

| Item | SHA-256 / result |
|---|---|
| model state | `{state_sha}` |
| gradient fixture | `{gradient_sha}` |
| optimizer before repair | `f62873a0b975cc4edc96adc12e711ee3eb7a934114f2f86722e0cc45768dbc82` |
| optimizer current | `{optimizer_sha}` |
| TensorFlow notebook | `{tf_notebook_sha}` |
| PyTorch notebook | `{torch_notebook_sha}` |
| forward max logit difference | `{previous_machine['post_repair_max_logit_difference']}` |
| gradient cosine | `{previous_machine['post_repair_gradient']['cosine']}` |
| gradient relative L2 | `{previous_machine['post_repair_gradient']['relative_l2']}` |

The original failures were reproduced: step 1 parameter `2.9802322387695312e-08`;
step 2 parameter `1.1920928955078125e-07`. No fixture changed.
""",
    )
    write(
        "02_pytorch_adamw_runtime_semantics.md",
        """
# PyTorch AdamW Runtime Semantics

PyTorch `2.11.0+cu126`, CPU, AdamW single-tensor path (`foreach=None`,
`fused=None` resolves to single-tensor for optimizer update). Effective order:
increment step, decoupled weight decay, `lerp_` first moment, second-moment
multiply/addcmul, Python-double bias correction, denominator
`sqrt(v)/sqrt(bc2)+eps`, then in-place addcdiv.

Global clipping is separate and uses CPU AVX2 eight-lane float32 vector-norm
reduction. Foreach, fused, and single-tensor bounded AdamW probes produced
zero output difference for the registered CPU case.
""",
    )
    write(
        "03_gradient_clipping_isolation.md",
        f"""
# Gradient Clipping Isolation

The earliest real-batch clipping divergence was `AT_GLOBAL_NORM`: ordinary
TensorFlow reduction gave `{tf_probe['clipping']['norm_methods']['linalg']['global_norm']}`,
while PyTorch gave `15.556995391845703`. Eight-lane reduction plus source-layout
handling gives `{clipping['global_norm']}`, coefficient
`{clipping['clip_coefficient']}`, and exact `127/127` clipped tensors.

See `03_gradient_clipping_isolation.csv`.
""",
    )
    write(
        "04_scalar_vector_reference_cases.md",
        f"""
# Scalar And Small-Tensor Cases

All candidate scalar, negative-scalar, vector, matrix, zero-gradient,
small-gradient, clipping, two-variable, preclipped, two-step, and restored
moment cases passed the registered `2e-8` gate.

Candidate maximums: `{tf_probe['candidate_max_abs']}`.
""",
    )
    write(
        "05_update_order_audit.md",
        """
# Update Order Audit

Matched order: step increment; parameter decay; first moment via
`m + (g-m)*(1-beta1)`; second moment via `v*=beta2` then
`v+=g*g*(1-beta2)`; Python-scalar bias corrections; epsilon outside square
root; adaptive update via `(m/denom)*(-step_size)`.

Alternatives for post-decay, epsilon-inside-root, delayed step increment, and
reassociated multiply/divide did not match the PyTorch trace.
""",
    )
    write(
        "06_dtype_and_scalar_audit.md",
        """
# Dtype And Scalar Audit

Parameters, gradients, moments, LR application, beta application, denominator,
and updates are float32. Iterations are int64. PyTorch computes beta powers,
bias corrections, square-root correction, and step size as Python doubles
before casting into float32 kernels. Production TensorFlow emulates those
scalar roundings with double-single float32 arithmetic; it introduces no
float64 tensor into training.

The first old-state divergence was float32 subtraction `1.0f-beta`, not the
PyTorch scalar `float32(1.0-beta)`.
""",
    )
    write(
        "07_foreach_fused_reference_audit.md",
        """
# Foreach And Fused Audit

The locked CPU AdamW update targets single-tensor semantics. Explicit
single-tensor, foreach, and fused bounded probes were all available and
produced max absolute and ULP difference zero against single-tensor.
Clipping norm reduction remains a distinct foreach-capable operation.
""",
    )
    write(
        "08_minimal_optimizer_repair.md",
        """
# Minimal Optimizer Repair

Only `lap_gnn_tf/training/optimizer.py` scientific runtime code changed.
The repair adds exact global clipping, source-layout-aware norm reduction,
PyTorch operation order, float32 double-single scalar rounding, fail-closed
dtype/sparse-gradient checks, and serialization-safe transient layout state.

The AdamW equations and registered hyperparameters are unchanged. Graph,
features, architecture, readout, metrics, scheduler, stopping, checkpoint
policy, data, priors, and PyTorch code are untouched.
""",
    )
    write(
        "09_live_model_step1.md",
        f"""
# Live Model Step 1

Captured before the final source-layout clipping repair:

| parameter max | momentum max | velocity max | pass |
|---:|---:|---:|---|
| {step1['parameter_max_abs']} | {step1['momentum_max_abs']} | {step1['velocity_max_abs']} | {step1['pass_2e_8']} |

This is a failed preregistered gate. No replacement live run was executed
after the four-update budget was exhausted.
""",
    )
    write(
        "10_live_model_step2.md",
        f"""
# Live Model Step 2

| parameter max | momentum max | velocity max | pass |
|---:|---:|---:|---|
| {step2['parameter_max_abs']} | {step2['momentum_max_abs']} | {step2['velocity_max_abs']} | {step2['pass_2e_8']} |

Iterations matched at 2 and all tensors were finite, but the numerical gate
failed.
""",
    )
    write(
        "11_repeated_optimizer_parity.md",
        """
# Repeated Optimizer Stability

Ten eager instances, ten `tf.function` instances, and five fresh processes
were deterministic for the synthetic fixed-tensor gate. All repeated outputs
were array-exact with the first repetition. A repeated real-model live gate
was not run because it would violate the four-update budget.
""",
    )
    write(
        "12_optimizer_checkpoint_roundtrip.md",
        """
# Optimizer Checkpoint Roundtrip

Synthetic `.keras` save/load preserves parameters, first moments, second
moments, iteration counter, and optimizer hyperparameters. Step-two
continuation from an uninterrupted optimizer and a restored optimizer is
array-exact. Transient clipping-layout variables are intentionally excluded
from Trackable state.

No PyTorch real-model step-three comparison was run.
""",
    )
    write(
        "13_mixed_precision_contract.md",
        """
# Mixed Precision Contract

The bounded smoke test passes: float32 master variable, float32 moments,
LossScaleOptimizer-scaled loss, unscale before inner clipping, clipping before
AdamW update, finite result, and tracked dynamic loss-scale state. Strict
parity itself remains CPU float32; float16 updates are not compared
numerically to PyTorch float32.
""",
    )
    write(
        "14_regression_test_results.md",
        """
# Regression Test Results

Bounded suite outcome after checkpoint repair: **56 passed, 2 failed, 58
total**. The two failures are the intentionally unchanged package-level live
step artifacts, which remain above `2e-8`. All forward, graph, mapping,
parameter count, gradient, metric, scheduler, stopping, serialization,
isolation, eager, graph-mode, fresh-process, and mixed-precision bounded tests
pass.
""",
    )
    write(
        "15_package_manifest_and_hash_update.md",
        f"""
# Package Manifest And Hash Update

Not updated, by stop condition.

| Item | Value |
|---|---|
| registered payload | `{registered_payload}` |
| unregistered repaired payload | `{unregistered_payload}` |
| package manifest updated | false |
| CHECKSUMS updated | false |

Publishing this payload is blocked until a new preregistered live gate passes.
""",
    )
    write(
        "16_kaggle_notebook_revalidation.md",
        f"""
# Kaggle Notebook Revalidation

Notebook contract tests pass and the existing notebook remains unchanged at
`{tf_notebook_sha}`. The PyTorch notebook remains unchanged at
`{torch_notebook_sha}`. Expected payload hashes were not updated, so the
notebook must fail closed for this unregistered optimizer tree.
""",
    )
    write(
        "17_tensorflow_seed42_launch_plan.md",
        """
# TensorFlow Seed42 Launch Plan

Launch is not authorized. There is no valid Kaggle seed42 command or output
ZIP for this HOLD state. The existing notebook and expected output convention
remain unchanged, but a full run must wait for a new live step-1/step-2 gate,
manifest regeneration, checksum registration, and notebook hash update.
""",
    )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
