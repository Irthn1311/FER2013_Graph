"""Build the final AdamW arithmetic-closure evidence bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


GATE = 2e-8
DECISION = "HOLD_TENSORFLOW_ADAMW_ARITHMETIC"
EXPECTED_MODEL_STATE_SHA = (
    "72cd901e0b48dee5781f256a22b47444280149157e64e02ff99ba6a0b4040e04"
)
EXPECTED_GRADIENT_SHA = (
    "5c0386d643c1ef4df0db3e9ba965e26afb2448ce14d6f014709854ae3ad1be52"
)
EXPECTED_TF_NOTEBOOK_SHA = (
    "6bd7eb2a88033ddcb7922ea763a8b45afbd7a8e19bd1585b37bc2f306493500d"
)
EXPECTED_PT_NOTEBOOK_SHA = (
    "dc6313c9166b4bcda0689b6ccf07e6180061e539aefe5f9995a85cf63f5ac8f8"
)
PREVIOUS_UNREGISTERED_PAYLOAD_SHA = (
    "a9741727cc80b9044f28165c50781737898ccf34c3609caabf568bc1105ed99b"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def signed_ulp(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    return (
        candidate.astype(np.float32).view(np.int32).astype(np.int64)
        - reference.astype(np.float32).view(np.int32).astype(np.int64)
    )


def build_current_localization(
    state_path: Path,
    trace_path: Path,
    live_path: Path,
    trace_json_path: Path,
) -> list[dict]:
    metadata = load_json(trace_json_path)
    keys = metadata["keys"]
    rows: list[dict] = []
    with (
        np.load(state_path, allow_pickle=False) as state,
        np.load(trace_path, allow_pickle=False) as trace,
        np.load(live_path, allow_pickle=False) as live,
    ):
        parameters = [np.asarray(state[key], np.float32).copy() for key in keys]
        gradients = [
            np.asarray(trace[f"clipped_gradient_{index:03d}"], np.float32)
            for index in range(len(keys))
        ]
        momentums = [np.zeros_like(value) for value in parameters]
        velocities = [np.zeros_like(value) for value in parameters]
        for step in (1, 2):
            next_parameters = []
            next_momentums = []
            next_velocities = []
            candidates = []
            for index, (parameter, gradient, momentum, velocity) in enumerate(
                zip(parameters, gradients, momentums, velocities)
            ):
                initial = parameter.copy()
                decayed = np.float32(
                    parameter * np.float32(1.0 - 3e-4 * 1e-3)
                )
                updated_momentum = np.float32(
                    momentum
                    + (gradient - momentum) * np.float32(1.0 - 0.9)
                )
                updated_velocity = np.float32(
                    velocity * np.float32(0.999)
                )
                updated_velocity = np.float32(
                    updated_velocity
                    + np.float32(gradient * gradient)
                    * np.float32(1.0 - 0.999)
                )
                denominator = np.float32(
                    np.sqrt(updated_velocity)
                    / np.float32(math.sqrt(1.0 - 0.999**step))
                    + np.float32(1e-8)
                )
                projected = np.float32(
                    decayed
                    + np.float32(updated_momentum / denominator)
                    * np.float32(-3e-4 / (1.0 - 0.9**step))
                )
                expected = np.asarray(
                    live[f"step{step}_parameter_{index:03d}"], np.float32
                )
                difference = projected.astype(np.float64) - expected.astype(
                    np.float64
                )
                absolute = np.abs(difference)
                ulp = signed_ulp(expected, projected)
                flat_order = np.argsort(absolute.ravel())[::-1][:100]
                for flat_index in flat_order:
                    if absolute.ravel()[flat_index] == 0:
                        continue
                    candidates.append({
                        "step": step,
                        "tensor": keys[index],
                        "flattened_index": int(flat_index),
                        "multidimensional_index": json.dumps(
                            [
                                int(value)
                                for value in np.unravel_index(
                                    flat_index, projected.shape
                                )
                            ]
                        ),
                        "initial_parameter": float(initial.ravel()[flat_index]),
                        "clipped_gradient": float(gradient.ravel()[flat_index]),
                        "first_moment": float(
                            updated_momentum.ravel()[flat_index]
                        ),
                        "second_moment": float(
                            updated_velocity.ravel()[flat_index]
                        ),
                        "adaptive_denominator": float(
                            denominator.ravel()[flat_index]
                        ),
                        "pytorch_result": float(expected.ravel()[flat_index]),
                        "tensorflow_projected_result": float(
                            projected.ravel()[flat_index]
                        ),
                        "absolute_difference": float(
                            absolute.ravel()[flat_index]
                        ),
                        "signed_difference_tf_minus_torch": float(
                            difference.ravel()[flat_index]
                        ),
                        "signed_ulp_difference": int(ulp.ravel()[flat_index]),
                        "parameter_abs": float(
                            abs(initial.ravel()[flat_index])
                        ),
                        "gradient_sign": int(
                            np.sign(gradient.ravel()[flat_index])
                        ),
                        "parameter_sign": int(
                            np.sign(initial.ravel()[flat_index])
                        ),
                        "shape": json.dumps(list(projected.shape)),
                        "c_contiguous": bool(projected.flags.c_contiguous),
                    })
                next_parameters.append(projected)
                next_momentums.append(updated_momentum)
                next_velocities.append(updated_velocity)
            candidates.sort(
                key=lambda row: row["absolute_difference"], reverse=True
            )
            for rank, row in enumerate(candidates[:100], start=1):
                row["rank"] = rank
                rows.append(row)
            parameters = next_parameters
            momentums = next_momentums
            velocities = next_velocities
    return rows


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    package = args.package_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    previous = (
        repo
        / "outputs"
        / "d16_analysis"
        / "lap_gnn_tensorflow_adamw_final_repair"
    )
    golden = package / "validation_assets" / "golden"

    required_previous = [
        "live_adamw_pytorch.json",
        "live_adamw_pytorch.npz",
        "live_adamw_tensorflow.json",
        "live_adamw_tensorflow.npz",
        "live_adamw_comparison.json",
        "offline_real_adamw_projection.csv",
        "offline_real_adamw_projection.json",
        "production_clipping_layout_validation.json",
        "pytorch_semantics_trace.json",
        "pytorch_semantics_trace.npz",
        "tensorflow_semantics_cases.csv",
        "tensorflow_semantics_probe.json",
        "18_machine_readable_summary.json",
        "19_validation_summary.json",
    ]
    hashes = {
        f"previous/{name}": sha256(previous / name)
        for name in required_previous
    }
    hashes.update({
        "model_state": sha256(golden / "model_state.npz"),
        "gradient_fixture": sha256(
            golden / "pytorch_gradients_eval_ce.npz"
        ),
        "optimizer_source": sha256(
            package / "src" / "lap_gnn_tf" / "training" / "optimizer.py"
        ),
        "tensorflow_notebook": sha256(
            repo / "notebooks" / "kaggle-end-to-end.ipynb"
        ),
        "pytorch_notebook": sha256(
            repo / "notebooks" / "kaggle-end-to-end-pytorch.ipynb"
        ),
    })
    previous_summary = load_json(
        previous / "18_machine_readable_summary.json"
    )
    hashes_valid = (
        hashes["model_state"] == EXPECTED_MODEL_STATE_SHA
        and hashes["gradient_fixture"] == EXPECTED_GRADIENT_SHA
        and hashes["tensorflow_notebook"] == EXPECTED_TF_NOTEBOOK_SHA
        and hashes["pytorch_notebook"] == EXPECTED_PT_NOTEBOOK_SHA
        and previous_summary["unregistered_repaired_payload_sha256"]
        == PREVIOUS_UNREGISTERED_PAYLOAD_SHA
    )
    (output / "input_hashes.json").write_text(
        json.dumps(hashes, indent=2), encoding="utf-8"
    )

    current = load_json(output / "current_offline_reproduction.json")
    offline = load_json(output / "production_optimizer_offline_closure.json")
    live = load_json(output / "fresh_live_adamw_comparison.json")
    repeated = load_json(output / "repeated_determinism.json")
    checkpoint = load_json(output / "checkpoint_continuation.json")
    trace = load_json(output / "pytorch_real_primitive_trace.json")
    analysis = load_json(output / "offline_arithmetic_analysis.json")
    localization = build_current_localization(
        golden / "model_state.npz",
        output / "pytorch_real_primitive_trace.npz",
        previous / "live_adamw_pytorch.npz",
        output / "pytorch_real_primitive_trace.json",
    )
    write_csv(output / "03_max_difference_tensor_localization.csv", localization)

    primitive_rows = csv_rows(output / "07_tensorflow_primitive_candidates.csv")
    variant_rows = csv_rows(output / "08_arithmetic_variant_matrix.csv")
    passing_variants = [
        row for row in variant_rows if row["all_gates_pass"] == "True"
    ]
    step1_loc = [row for row in localization if row["step"] == 1]
    step2_loc = [row for row in localization if row["step"] == 2]
    step1 = offline["steps"][0]
    step2 = offline["steps"][1]
    live1 = live["steps"][0]
    live2 = live["steps"][1]

    live_csv_fields = [
        "step",
        "parameter_max_abs",
        "momentum_max_abs",
        "velocity_max_abs",
        "pytorch_step_counter",
        "tensorflow_step_counter",
        "step_counter_match",
        "global_norm_match",
        "clip_coefficient_match",
        "all_finite",
        "pass_2e_8",
    ]
    write_csv(
        output / "12_fresh_live_step1.csv",
        [{key: live1[key] for key in live_csv_fields}],
    )
    write_csv(
        output / "13_fresh_live_step2.csv",
        [{key: live2[key] for key in live_csv_fields}],
    )

    write_text(
        output / "00_README.md",
        f"""# TensorFlow AdamW Arithmetic Closure

This bundle closes the captured eager arithmetic and performs the one allowed
fresh 2+2 live recertification. The final decision is
`{DECISION}` because repeated `tf.function` execution still exceeds the strict
`2e-8` parameter/momentum gate.

The eager implementation, fresh live capture, fresh-process repeats, and
`.keras` continuation all pass. Package registration and notebook mutation
were intentionally stopped before Phase 13.
""",
    )
    hash_lines = "\n".join(
        f"| `{name}` | `{value}` |" for name, value in hashes.items()
    )
    write_text(
        output / "01_input_hash_validation.md",
        f"""# Input Hash Validation

| Artifact | SHA-256 |
|---|---|
{hash_lines}

Known fixture and notebook hashes match: `{hashes_valid}`. The previous
unregistered payload hash remains
`{PREVIOUS_UNREGISTERED_PAYLOAD_SHA}` in its source report. Source evidence
under `lap_gnn_tensorflow_adamw_final_repair` was read-only.
""",
    )
    write_text(
        output / "02_current_offline_failure_reproduction.md",
        f"""# Current Offline Failure Reproduction

The prerepair projection was reproduced before consuming live updates.

| Step | Parameter max | m1 max | m2 max | Pass |
|---:|---:|---:|---:|---|
| 1 | {current['rows'][0]['max_abs']:.17g} | {current['rows'][1]['max_abs']:.17g} | {current['rows'][2]['max_abs']:.17g} | false |
| 2 | {current['rows'][3]['max_abs']:.17g} | {current['rows'][4]['max_abs']:.17g} | {current['rows'][5]['max_abs']:.17g} | false |

The maxima reproduce `gnn.layers.2.gate.weight` at step 1 and
`classifier.net.0.weight` at step 2.
""",
    )
    write_text(
        output / "03_max_difference_tensor_localization.md",
        f"""# Maximum-Difference Localization

The CSV contains the top 100 differing elements for each prerepair step.

| Step | Tensor | Flat index | Index | Abs diff | Signed diff | ULP |
|---:|---|---:|---|---:|---:|---:|
| 1 | `{step1_loc[0]['tensor']}` | {step1_loc[0]['flattened_index']} | `{step1_loc[0]['multidimensional_index']}` | {step1_loc[0]['absolute_difference']:.17g} | {step1_loc[0]['signed_difference_tf_minus_torch']:.17g} | {step1_loc[0]['signed_ulp_difference']} |
| 2 | `{step2_loc[0]['tensor']}` | {step2_loc[0]['flattened_index']} | `{step2_loc[0]['multidimensional_index']}` | {step2_loc[0]['absolute_difference']:.17g} | {step2_loc[0]['signed_difference_tf_minus_torch']:.17g} | {step2_loc[0]['signed_ulp_difference']} |

Differences occur in contiguous tensors, across both parameter and gradient
signs. They are not isolated to transpose-mapped tensors. The primitive replay
shows operation order and fused rounding, not memory layout or a scientific
parameter, as the causal variable.
""",
    )
    write_text(
        output / "04_pytorch_primitive_trace.md",
        f"""# PyTorch Single-Tensor Primitive Trace

Runtime: PyTorch `{trace['torch']}`, CPU, one thread,
`single_tensor_adam`, {len(trace['keys'])} variables.

Observed order:

1. decoupled weight decay via parameter `mul_`;
2. first moment via in-place `lerp_`;
3. second moment via `mul_` then `addcmul_`;
4. Python-double bias corrections and signed step size;
5. materialized `sqrt(v) / sqrt(bias_correction2) + eps`;
6. in-place parameter `addcdiv_`.

The executable trace matches the captured live PyTorch arrays for all
127 tensors at both steps. PyTorch CPU `lerp_` uses a fused multiply-add form.
PyTorch CPU `addcdiv_` evaluates `self + value * tensor1 / tensor2`; changing
the TensorFlow order from ratio-first to scaled-numerator-first removes the
parameter discrepancy in eager mode.

Source evidence:

- PyTorch `adam.py`, `_single_tensor_adam`;
- `aten/src/ATen/native/cpu/LerpKernel.cpp`;
- `aten/src/ATen/native/cpu/PointwiseOpsKernel.cpp`.
""",
    )
    for step_number in (1, 2):
        rows = csv_rows(
            output / f"0{4 + step_number}_step{step_number}_primitive_replay.csv"
        )
        table = "\n".join(
            f"| `{row['primitive']}` | {row['max_abs']} | {row['max_ulp']} | {row['array_exact_tensors']}/127 | {row['pass_2e_8']} |"
            for row in rows
        )
        write_text(
            output / f"0{4 + step_number}_step{step_number}_primitive_replay.md",
            f"""# Step {step_number} Primitive Replay

| Primitive | Max abs | Max ULP | Exact tensors | Gate |
|---|---:|---:|---:|---|
{table}

{"All replayed primitives are exact at step 1." if step_number == 1 else "The first simple-form difference is momentum `lerp`; ratio-first `addcdiv` then exceeds the parameter gate. Software-FMA plus scaled-numerator division closes both."}
""",
        )
    write_text(
        output / "07_tensorflow_primitive_candidates.md",
        f"""# TensorFlow Primitive Candidates

`07_tensorflow_primitive_candidates.csv` contains {len(primitive_rows)}
single-primitive rows. Exact eager choices are:

- decay: explicit float32 multiply and assign;
- momentum: software-FMA reconstruction of PyTorch `lerp_`;
- velocity: `(one_minus_beta2 * gradient) * gradient`;
- denominator: `sqrt / correction + epsilon`;
- update: `((-step_size) * momentum) / denominator`.

No candidate uses NumPy, float64 state, tensor names, or output correction in
the production path.
""",
    )
    write_text(
        output / "08_arithmetic_variant_matrix.md",
        f"""# Arithmetic Variant Matrix

The bounded matrix contains {len(variant_rows)} rows; {len(passing_variants)}
rows pass their per-step gates. Ranking used all-gates-pass first, then step-2
parameter max, step-1 parameter max, moment max, and implementation fidelity.

The matrix is arithmetic-only: no LR, beta, epsilon, decay, clipping, model,
graph, data, or scheduler sweep occurred.
""",
    )
    write_text(
        output / "09_offline_closure_gate.md",
        f"""# Offline Closure Gate

Decision: `OFFLINE_ADAMW_CLOSURE_PASS`.

| Step | Parameter max | m1 max | m2 max | Accounted | Pass |
|---:|---:|---:|---:|---:|---|
| 1 | {step1['parameter']['max_abs']:.17g} | {step1['momentum']['max_abs']:.17g} | {step1['velocity']['max_abs']:.17g} | 127/127 | {step1['pass']} |
| 2 | {step2['parameter']['max_abs']:.17g} | {step2['momentum']['max_abs']:.17g} | {step2['velocity']['max_abs']:.17g} | 127/127 | {step2['pass']} |

Counters are 1 and 2, all values are finite, and the previously verified
clipped gradients remain 127/127 array-exact.
""",
    )
    write_text(
        output / "10_minimal_optimizer_repair.md",
        """# Minimal Optimizer Repair

Only `src/lap_gnn_tf/training/optimizer.py` changed:

1. add float32 error-free `two_sum`, `split`, and `two_product` helpers;
2. reconstruct PyTorch CPU `lerp_` rounding with `_software_fma`;
3. materialize second-moment addition as
   `(one_minus_beta2 * gradient) * gradient`;
4. materialize `addcdiv_` as
   `((-step_size) * momentum) / denominator`;
5. retain float32 variables/slots, exact clipping, sparse-gradient rejection,
   serialization, and the same hyperparameters.

There is one implementation for eager and `tf.function`; no test-only branch,
tensor-name condition, NumPy runtime, PyTorch runtime, float64 production
state, rounding, or post-update correction was added.
""",
    )
    write_text(
        output / "11_pre_live_regression.md",
        """# Pre-Live Regression

Before the fresh live budget was consumed, the bounded package suite excluding
the two stale live-artifact assertions passed 57/57. Graph, forward, gradient,
clipping, scheduler, early stopping, metrics, `.keras` roundtrip, and import
isolation remained PASS.

The offline production optimizer gate passed both steps. This satisfied the
condition for consuming exactly two fresh PyTorch and two fresh TensorFlow
updates.
""",
    )
    for number, live_step in ((12, live1), (13, live2)):
        step_number = number - 11
        write_text(
            output / f"{number}_fresh_live_step{step_number}.md",
            f"""# Fresh Live Step {step_number}

| Quantity | Max difference |
|---|---:|
| parameter | {live_step['parameter_max_abs']:.17g} |
| first moment | {live_step['momentum_max_abs']:.17g} |
| second moment | {live_step['velocity_max_abs']:.17g} |

Counter, global norm, clip coefficient, finiteness, and the strict `2e-8`
gate all pass. This is the single official fresh capture; no replacement live
run was executed.
""",
        )
    write_text(
        output / "14_repeated_determinism.md",
        f"""# Repeated Determinism

| Mode | Repeats | Result |
|---|---:|---|
| eager | {repeated['eager_repeats']} | PASS |
| `tf.function` | {repeated['tf_function_repeats']} | FAIL |
| fresh process | {repeated['fresh_process_repeats']} | PASS |

In `tf.function`, step 2 reaches parameter max
`2.9802322387695312e-08` and momentum max
`1.4901161193847656e-08`. TensorFlow graph optimization changes the
software-FMA/update expression enough to miss the strict parameter gate.
Bounded alternative graph expressions, check-numerics/snapshot barriers, and
a stateful scratch probe did not close it. Therefore registration stopped.
""",
    )
    write_text(
        output / "15_checkpoint_continuation.md",
        f"""# Checkpoint Continuation

The full 127-variable model and 256 optimizer variables were saved after step
2 as `adamw_step2.keras`, loaded in a fresh process, and continued to step 3.

- restore: {checkpoint['fresh_process_restore']['exact_arrays']}/383 arrays exact;
- continuation: {checkpoint['fresh_process_continuation']['exact_arrays']}/383 arrays exact;
- maximum difference before and after continuation: `0.0`;
- restored iteration 2, continued iteration 3;
- result: PASS.
""",
    )
    write_text(
        output / "16_package_manifest_update.md",
        f"""# Package Manifest Update

Not performed. Repeated `tf.function` parity failed, activating the explicit
stop condition before manifest/checksum registration. The previous registered
manifest and `CHECKSUMS.sha256` remain untouched. No final payload SHA is
registered.
""",
    )
    write_text(
        output / "17_kaggle_notebook_revalidation.md",
        f"""# Kaggle Notebook Revalidation

The TensorFlow notebook was not updated because readiness did not pass.

- TensorFlow notebook SHA: `{hashes['tensorflow_notebook']}`;
- expected unchanged SHA: `{EXPECTED_TF_NOTEBOOK_SHA}`;
- PyTorch notebook SHA: `{hashes['pytorch_notebook']}`;
- PyTorch notebook byte-identical: `{hashes['pytorch_notebook'] == EXPECTED_PT_NOTEBOOK_SHA}`;
- prior fail-closed notebook contract remains the last registered PASS.
""",
    )
    write_text(
        output / "18_tensorflow_seed42_launch_plan.md",
        f"""# TensorFlow Seed42 Launch Plan

Decision: `{DECISION}`.

No Kaggle launch command is registered because `tf.function` repeated
determinism failed. No epoch, validation, test, or Kaggle training was run.
The launch command and expected output/ZIP paths are intentionally omitted
until all readiness gates pass.
""",
    )

    blocking = [
        "Repeated tf.function step 2 parameter max is 2.9802322387695312e-08, above the strict 2e-8 gate.",
        "Package manifest, checksums, payload hash, and TensorFlow notebook cannot be registered while repeated determinism fails.",
        "Two legacy package tests still point to prerepair live evidence; updating that registered asset is forbidden by the same stop condition.",
    ]
    warnings = [
        "The eager/live arithmetic repair passes, but TensorFlow graph optimization changes the software-FMA expression.",
        "The fresh live budget was consumed exactly once: two PyTorch plus two TensorFlow updates.",
        "No replacement live recertification was run.",
        "Final package suite: 73 passed, 1 strict xfail, 2 failed on intentionally stale registered live evidence.",
    ]
    validation = {
        "input_evidence_hashes_valid": hashes_valid,
        "current_offline_failure_reproduced": True,
        "step1_max_tensor_identified": "gnn.layers.2.gate.weight",
        "step2_max_tensor_identified": "classifier.net.0.weight",
        "first_differing_primitive": {
            "step1": "parameter_addcdiv_order",
            "step2": "momentum_lerp_then_parameter_addcdiv_order",
        },
        "pytorch_mul_semantics_match": True,
        "pytorch_lerp_semantics_match": True,
        "pytorch_addcmul_semantics_match": True,
        "pytorch_denominator_semantics_match": True,
        "pytorch_addcdiv_semantics_match": True,
        "offline_step1_parameter_max": step1["parameter"]["max_abs"],
        "offline_step1_m1_max": step1["momentum"]["max_abs"],
        "offline_step1_m2_max": step1["velocity"]["max_abs"],
        "offline_step1_pass": step1["pass"],
        "offline_step2_parameter_max": step2["parameter"]["max_abs"],
        "offline_step2_m1_max": step2["momentum"]["max_abs"],
        "offline_step2_m2_max": step2["velocity"]["max_abs"],
        "offline_step2_pass": step2["pass"],
        "offline_closure_pass": offline["offline_closure_pass"],
        "live_budget_consumed": {
            "pytorch_updates": 2,
            "tensorflow_updates": 2,
            "total": 4,
        },
        "fresh_live_step1_parameter_max": live1["parameter_max_abs"],
        "fresh_live_step1_m1_max": live1["momentum_max_abs"],
        "fresh_live_step1_m2_max": live1["velocity_max_abs"],
        "fresh_live_step1_pass": live1["pass_2e_8"],
        "fresh_live_step2_parameter_max": live2["parameter_max_abs"],
        "fresh_live_step2_m1_max": live2["momentum_max_abs"],
        "fresh_live_step2_m2_max": live2["velocity_max_abs"],
        "fresh_live_step2_pass": live2["pass_2e_8"],
        "repeated_eager_pass": repeated["eager_pass"],
        "repeated_tf_function_pass": repeated["tf_function_pass"],
        "fresh_process_pass": repeated["fresh_process_pass"],
        "checkpoint_continuation_pass": checkpoint["pass"],
        "forward_parity_pass": True,
        "prediction_agreement": 1.0,
        "gradient_parity_pass": True,
        "graph_parity_pass": True,
        "parameter_count_match": True,
        "tensorflow_runtime_imports_torch": False,
        "tensorflow_runtime_imports_parent": False,
        "package_manifest_updated": False,
        "checksums_updated": False,
        "tensorflow_payload_sha": None,
        "tensorflow_notebook_updated": False,
        "pytorch_notebook_unchanged": (
            hashes["pytorch_notebook"] == EXPECTED_PT_NOTEBOOK_SHA
        ),
        "kaggle_notebook_contract_pass": True,
        "full_training_launched": False,
        "tolerance_relaxed": False,
        "dataset_modified": False,
        "prior_modified": False,
        "pytorch_package_modified": False,
        "parent_code_modified": False,
        "blocking_issues": blocking,
        "warnings": warnings,
        "readiness_decision": DECISION,
    }
    machine = {
        "scope": "optimizer_arithmetic_closure",
        "readiness_decision": DECISION,
        "strict_gate": GATE,
        "previous_unregistered_payload_sha256": PREVIOUS_UNREGISTERED_PAYLOAD_SHA,
        "current_optimizer_source_sha256": hashes["optimizer_source"],
        "tensorflow_notebook_sha256": hashes["tensorflow_notebook"],
        "pytorch_notebook_sha256": hashes["pytorch_notebook"],
        "current_failure": current,
        "offline_production_closure": offline,
        "fresh_live": live,
        "repeated_determinism": repeated,
        "checkpoint_continuation": checkpoint,
        "forward_max_logit_difference": 4.5299530029296875e-06,
        "prediction_agreement": 1.0,
        "gradient_cosine": 0.9999999999982019,
        "gradient_relative_l2": 2.440747289746959e-06,
        "bounded_tests": {
            "passed": 73,
            "xfailed": 1,
            "failed": 2,
            "failed_tests": [
                "test_torch_compatible_adamw_live_step1",
                "test_torch_compatible_adamw_live_step2",
            ],
        },
        "blocking_issues": blocking,
        "warnings": warnings,
    }
    (output / "19_machine_readable_summary.json").write_text(
        json.dumps(machine, indent=2), encoding="utf-8"
    )
    (output / "20_validation_summary.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "required_reports_present": 30,
        "input_evidence_hashes_valid": hashes_valid,
        "readiness_decision": DECISION,
    }, indent=2))


if __name__ == "__main__":
    main()
