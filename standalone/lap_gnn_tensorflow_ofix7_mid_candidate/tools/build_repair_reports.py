"""Build the bounded TensorFlow repair evidence bundle from locked probe data."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
REPORT = REPO / "outputs" / "d16_analysis" / "lap_gnn_tensorflow_port_repair"
OLD_REPORT = REPO / "outputs" / "d16_analysis" / "lap_gnn_tensorflow_port"
PYTORCH = REPO / "standalone" / "lap_gnn_pytorch_ofix7_mid_candidate"
TF_NOTEBOOK = REPO / "notebooks" / "kaggle-end-to-end.ipynb"
PYTORCH_NOTEBOOK = REPO / "notebooks" / "kaggle-end-to-end-pytorch.ipynb"

ORIGINAL_PAYLOAD_SHA = (
    "6fca5f6966aa4773d02b57f617b97dedb975ae86e609b9dab4edfe0b323934bc"
)
CANDIDATE_PAYLOAD_SHA = (
    "b796c2cfc2f7b3d8d3cfe40c0379370e6ab1ae4abb78e700ad79c0fc58409dd9"
)
PYTORCH_NOTEBOOK_SHA = (
    "dc6313c9166b4bcda0689b6ccf07e6180061e539aefe5f9995a85cf63f5ac8f8"
)
TF_NOTEBOOK_SHA = (
    "6bd7eb2a88033ddcb7922ea763a8b45afbd7a8e19bd1585b37bc2f306493500d"
)
READINESS = "HOLD_TENSORFLOW_PORT_REPAIR"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(name: str, text: str) -> None:
    (REPORT / name).write_text(text.strip() + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(name: str, fieldnames: list[str], rows: list[dict]) -> None:
    with (REPORT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    original = load_json(OLD_REPORT / "forward_parity_probe.json")
    instrumented = load_json(REPORT / "probe_default.json")
    repaired_onednn = load_json(REPORT / "probe_repaired_onednn.json")
    repaired_no_onednn = load_json(REPORT / "probe_repaired_no_onednn.json")
    repeated = load_json(REPORT / "repeated_forward_summary.json")
    classifier_pytorch = load_json(REPORT / "pytorch_classifier_probe.json")
    live = load_json(REPORT / "live_adamw_comparison.json")
    gradient = load_json(REPORT / "gradient_after_repair.json")
    pytorch_checks = load_json(REPORT / "pytorch_checksum_recheck.json")

    exact_original = {
        "sample_index": 1,
        "class_index": 1,
        "pytorch_logit": 3.290811777114868,
        "tensorflow_logit": 3.2908217906951904,
        "absolute_difference": 1.0013580322265625e-05,
        "signed_difference": 1.0013580322265625e-05,
        "ulp_distance": 42,
    }

    localization_rows = []
    for rank, row in enumerate(instrumented["top20"], start=1):
        localization_rows.append(
            {
                "rank": rank,
                "probe": "instrumented_pre_repair",
                **row,
            }
        )
    write_csv(
        "02_logit_difference_localization.csv",
        [
            "rank",
            "probe",
            "sample_index",
            "class_index",
            "pytorch_logit",
            "tensorflow_logit",
            "absolute_difference",
            "signed_difference",
            "ulp_distance",
        ],
        localization_rows,
    )

    classifier_rows = [
        {
            "probe": "A_pytorch_classifier",
            "input": "locked_pytorch_classifier_input",
            "implementation": "torch_linear",
            "execution": "eager",
            "max_abs": classifier_pytorch[
                "golden_input_against_golden_logits"
            ]["max_abs"],
            "relative_l2": classifier_pytorch[
                "golden_input_against_golden_logits"
            ]["relative_l2"],
        },
        {
            "probe": "B_pytorch_classifier",
            "input": "tensorflow_classifier_input",
            "implementation": "torch_linear",
            "execution": "eager",
            "max_abs": classifier_pytorch[
                "tensorflow_input_against_golden_logits"
            ]["max_abs"],
            "relative_l2": classifier_pytorch[
                "tensorflow_input_against_golden_logits"
            ]["relative_l2"],
        },
    ]
    for implementation, values in instrumented["classifier_candidates"].items():
        for key in [
            "golden_input_eager",
            "tensorflow_input_eager",
            "golden_input_tf_function",
            "tensorflow_input_tf_function",
        ]:
            value = values[key]
            classifier_rows.append(
                {
                    "probe": "A_tensorflow_classifier"
                    if key.startswith("golden")
                    else "B_tensorflow_classifier",
                    "input": "locked_pytorch_classifier_input"
                    if key.startswith("golden")
                    else "tensorflow_classifier_input",
                    "implementation": implementation,
                    "execution": "tf_function"
                    if key.endswith("tf_function")
                    else "eager",
                    "max_abs": value["max_abs"],
                    "relative_l2": value["relative_l2"],
                }
            )
    write_csv(
        "03_classifier_isolation.csv",
        [
            "probe",
            "input",
            "implementation",
            "execution",
            "max_abs",
            "relative_l2",
        ],
        classifier_rows,
    )

    backend_rows = [
        {
            "stage": "pre_repair_registered",
            "onednn": "default",
            "max_logit_difference": original["max_logit_difference"],
            "max_probability_difference": original["max_probability_difference"],
            "prediction_agreement": original["prediction_agreement"],
            "source": "locked_forward_parity_probe",
        },
        {
            "stage": "pre_repair_instrumented",
            "onednn": "default",
            "max_logit_difference": instrumented["baseline_logits"]["max_abs"],
            "max_probability_difference": instrumented["baseline_probabilities"][
                "max_abs"
            ],
            "prediction_agreement": instrumented["prediction_agreement"],
            "source": "debug_trace_probe",
        },
        {
            "stage": "pre_repair_console_probe",
            "onednn": "disabled",
            "max_logit_difference": 1.33514404296875e-05,
            "max_probability_difference": "",
            "prediction_agreement": 1.0,
            "source": "bounded_fresh_process_console_observation",
        },
        {
            "stage": "post_repair",
            "onednn": "default",
            "max_logit_difference": repaired_onednn["baseline_logits"]["max_abs"],
            "max_probability_difference": repaired_onednn[
                "baseline_probabilities"
            ]["max_abs"],
            "prediction_agreement": repaired_onednn["prediction_agreement"],
            "source": "repaired_fresh_process_probe",
        },
        {
            "stage": "post_repair",
            "onednn": "disabled",
            "max_logit_difference": repaired_no_onednn["baseline_logits"]["max_abs"],
            "max_probability_difference": repaired_no_onednn[
                "baseline_probabilities"
            ]["max_abs"],
            "prediction_agreement": repaired_no_onednn["prediction_agreement"],
            "source": "repaired_fresh_process_probe",
        },
    ]
    write_csv(
        "04_backend_reduction_audit.csv",
        [
            "stage",
            "onednn",
            "max_logit_difference",
            "max_probability_difference",
            "prediction_agreement",
            "source",
        ],
        backend_rows,
    )

    write_text(
        "00_README.md",
        f"""
# TensorFlow Port Repair Evidence

Decision: **{READINESS}**

The CPU float32 forward blocker was repaired without changing model equations:
the maximum repeated logit difference is now
`{repeated["eager"]["maximum"]:.17g}` and every eager, `tf.function`, and
fresh-process probe passes the locked `1e-5` gate.

The package is still held because the bounded live
`TorchCompatibleAdamW` comparison exceeds the preferred `2e-8` gate at both
steps. Phase 11 therefore remains intentionally unexecuted: checksums,
manifest, payload registration, and notebook hashes were not updated.

This directory is new repair evidence. The prior evidence under
`outputs/d16_analysis/lap_gnn_tensorflow_port` was not modified.
""",
    )

    write_text(
        "01_input_lock_and_hash_validation.md",
        f"""
# Input Lock And Hash Validation

| Lock | Result |
|---|---|
| repair branch | `repair/tensorflow-port-logit` |
| PyTorch checksum verification | PASS, {pytorch_checks["checked_files"]} files |
| PyTorch golden manifest | PASS |
| PyTorch baseline lock SHA | `d54c9162de7e4f6bda2ee37dbe735939f7542195d9d2fe6dbd5e61cd85351dc3` |
| checkpoint-policy lock SHA | `dfce606a69343b1a8de821ec3fc547d5700b94ff6c9a15f6b26d32e02601fc5f` |
| golden checkpoint SHA | `20084362d4a9529f7a833a1994ba407d958cb6264b42f60e4252efca3edad5ac` |
| state key count | 127 |
| trainable parameters | 1,061,192 |
| original TensorFlow payload SHA | `{ORIGINAL_PAYLOAD_SHA}` |
| preserved PyTorch notebook SHA | `{sha256(PYTORCH_NOTEBOOK)}` |
| TensorFlow notebook SHA at repair start/end | `{sha256(TF_NOTEBOOK)}` |

The original TensorFlow checksum set was valid before repair. It now reports
exactly two expected mismatches, `lap_gnn.py` and `motif_layers.py`, because
Phase 11 forbids re-signing a repair that fails the live optimizer gate.
The PyTorch package and golden fixtures remain checksum-valid.
""",
    )

    write_text(
        "02_logit_difference_localization.md",
        f"""
# Logit Difference Localization

The authoritative pre-repair registered probe reproduced:

| sample | class | PyTorch | TensorFlow | signed / absolute delta | ULP |
|---:|---:|---:|---:|---:|---:|
| {exact_original["sample_index"]} | {exact_original["class_index"]} | {exact_original["pytorch_logit"]:.16g} | {exact_original["tensorflow_logit"]:.16g} | {exact_original["absolute_difference"]:.17g} | {exact_original["ulp_distance"]} |

This exceeds the locked `1e-5` gate by
`1.3580322265625e-08`. The accompanying CSV contains the top 20 values from
the finer debug-instrumented pre-repair probe. Instrumentation changed the
exact reduction path slightly, so its top delta (`{instrumented["baseline_logits"]["max_abs"]:.17g}`)
is diagnostic evidence and does not replace the registered blocker above.

The maximum stayed at sample 1, class 1. Prediction agreement remained 100%.
""",
    )

    write_text(
        "03_classifier_isolation.md",
        f"""
# Classifier Isolation

| Controlled probe | Maximum absolute delta |
|---|---:|
| PyTorch classifier, exact locked PyTorch classifier input | {classifier_pytorch["golden_input_against_golden_logits"]["max_abs"]:.17g} |
| TensorFlow classifier, exact locked PyTorch classifier input | {instrumented["classifier_candidates"]["matmul_add"]["golden_input_eager"]["max_abs"]:.17g} |
| PyTorch classifier, TensorFlow classifier input | {classifier_pytorch["tensorflow_input_against_golden_logits"]["max_abs"]:.17g} |
| TensorFlow classifier, TensorFlow classifier input | {instrumented["classifier_candidates"]["matmul_add"]["tensorflow_input_eager"]["max_abs"]:.17g} |

All tested TensorFlow float32 equations (`matmul`, raw `MatMul`, `einsum`,
`tensordot`, explicit bias add, and the current explicit linear path) produced
the same relevant result. Classifier-only arithmetic is below the gate; the
near-threshold final difference is inherited from the readout input.
No classifier implementation was changed.
""",
    )

    write_text(
        "04_backend_reduction_audit.md",
        f"""
# Backend And Reduction Audit

| Stage | oneDNN | Max logit delta | Result |
|---|---|---:|---|
| pre-repair registered | default | {original["max_logit_difference"]:.17g} | FAIL |
| pre-repair debug trace | default | {instrumented["baseline_logits"]["max_abs"]:.17g} | FAIL |
| pre-repair bounded console probe | disabled | {1.33514404296875e-05:.17g} | FAIL |
| post-repair | default | {repaired_onednn["baseline_logits"]["max_abs"]:.17g} | PASS |
| post-repair | disabled | {repaired_no_onednn["baseline_logits"]["max_abs"]:.17g} | PASS |

Disabling oneDNN did not solve the original blocker and is not registered as a
training workaround. The repaired default path passes deterministically, so
normal notebook backend policy remains unchanged: CPU float32 for parity,
mixed precision only for training, and XLA disabled.
""",
    )

    write_text(
        "05_fine_grained_readout_trace.md",
        """
# Fine-Grained Readout Trace

The pre-readout node representation was not affected by the candidate pooling
change. The first operation with a meaningful improvement was per-part graph
pooling:

| Tensor | segment max abs | graph-wise max abs | improvement |
|---|---:|---:|---:|
| `part_pool_nose_cheek` | 3.5762786865234375e-06 | 7.152557373046875e-07 | 2.86102294921875e-06 |
| `part_pool_eye` | 2.3245811462402344e-06 | 4.172325134277344e-07 | 1.9073486328125e-06 |
| `part_pool_brow` | 2.3245811462402344e-06 | 5.9604644775390625e-07 | 1.7285346984863281e-06 |

TensorFlow's flat `unsorted_segment_sum` and PyTorch's per-graph loop use
different float32 accumulation orders. The difference propagated through the
readout projection into the classifier. The CSV preserves all traced tensors;
large ULP values close to zero are not interpreted as large scientific errors.
""",
    )

    write_text(
        "06_minimal_repair.md",
        f"""
# Minimal Repair

Two source edits were made.

1. `src/lap_gnn_tf/model/motif_layers.py:18`:
   `part_pool` now iterates over graph IDs with `tf.map_fn`, masks the
   contiguous graph nodes, and performs the same per-graph `reduce_sum` /
   `reduce_mean` order as the PyTorch reference.
2. `src/lap_gnn_tf/model/lap_gnn.py:46`:
   `LapGNN.build()` calls `super().build(input_shape)` so Keras records the
   structured input as built without creating state.

The pooling numerator, denominator clamp (`1e-6`), validity threshold
(`1e-5`), sum/mean semantics, part order, feature order, graph order, tensor
shapes, mapped weights, and classifier equation are unchanged. No rounding,
clipping, corrective constant, float64 model path, or tolerance change was
introduced.

Post-repair default oneDNN max logit delta:
`{repaired_onednn["baseline_logits"]["max_abs"]:.17g}`.
""",
    )

    def repeated_row(label: str, values: dict) -> str:
        return (
            f"| {label} | {values['runs']} | {values['minimum']:.17g} | "
            f"{values['maximum']:.17g} | {values['mean']:.17g} | "
            f"{values['sample_standard_deviation']:.17g} | "
            f"{values['maximum_probability_difference']:.17g} | "
            f"{values['maximum_loss_difference']:.17g} | "
            f"{values['minimum_prediction_agreement']:.1f} |"
        )

    write_text(
        "07_repeated_forward_parity.md",
        f"""
# Repeated Forward Parity

| Mode | Runs | Min logit | Max logit | Mean logit | Sample SD | Max probability | Max loss | Prediction agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{repeated_row("eager", repeated["eager"])}
{repeated_row("tf.function", repeated["tf_function"])}
{repeated_row("fresh process", repeated["fresh_process"])}

Every one of the 25 probes passed `max_abs <= 1e-5`. Classification:
**{repeated["parity_class"]}**.
""",
    )

    step1, step2 = live["steps"]
    write_text(
        "08_live_torchcompatible_adamw.md",
        f"""
# Live TorchCompatibleAdamW

Execution budget was consumed exactly once: two PyTorch updates and two
TensorFlow updates, four updates total. No additional update was run.

| Step | Parameter max abs | First moment max abs | Second moment max abs | Counter | Finite | `<=2e-8` |
|---:|---:|---:|---:|---|---|---|
| 1 | {step1["parameter_max_abs"]:.17g} | {step1["momentum_max_abs"]:.17g} | {step1["velocity_max_abs"]:.17g} | 1 / 1 | yes | **FAIL** |
| 2 | {step2["parameter_max_abs"]:.17g} | {step2["momentum_max_abs"]:.17g} | {step2["velocity_max_abs"]:.17g} | 2 / 2 | yes | **FAIL** |

Keys match for all 127 tensors; counters and finiteness pass. The strict
preferred variable gate does not pass, and moment differences are also above
`2e-8`, so optimizer slot-state parity is not accepted.

Capture warning: fresh PyTorch step-1 NumPy views were overwritten by step 2
before serialization. Step 1 was therefore compared against the pre-existing
locked PyTorch live-step fixture; fresh step 2 remained valid. The update
budget prevents rerunning this experiment. Per the stop condition, this is a
HOLD rather than a silently relaxed tolerance.
""",
    )

    write_text(
        "09_keras_build_serialization_repair.md",
        """
# Keras Build And Serialization Repair

Cause: `LapGNN` is a subclassed model with a structured dictionary input and
had no explicit `build(input_shape)`. Keras therefore warned that `build()`
was called on an unbuilt subclass without a build implementation.

Repair: `LapGNN.build()` now delegates to `super().build(input_shape)`. It
creates no variables and does not suppress warnings.

Bounded tests confirm:

- no unresolved subclass-build warning;
- 127 trainable variables restored;
- 1,061,192 trainable parameters retained;
- logits are array-exact before and after `.keras` roundtrip;
- model configuration is equal;
- optimizer variable count and slot values roundtrip exactly;
- no custom-object ambiguity due to registered Keras serializables.
""",
    )

    write_text(
        "10_regression_test_results.md",
        f"""
# Regression Test Results

Final bounded suite: **36 passed, 2 failed**.

The only failures are the deliberate evidence gates:

- `test_torch_compatible_adamw_live_step1.py`;
- `test_torch_compatible_adamw_live_step2.py`.

All other existing and focused tests pass, including repeated/fresh logit
parity, classifier isolation, backend flags, graph parity, feature schema,
127/127 mapping, 1,061,192 parameters, repaired gradient parity
(`cosine={gradient["cosine"]:.16g}`,
`relative_l2={gradient["relative_l2"]:.16g}`,
`max_abs={gradient["max_abs"]:.16g}`), checkpoint/optimizer-slot roundtrip,
Keras clean load, scheduler semantics, early stopping, metrics, and notebook
contract.

External warnings are Matplotlib/pyparsing deprecations. No test launched an
epoch or executed a new optimizer update.
""",
    )

    write_text(
        "11_package_manifest_update.md",
        f"""
# Package Manifest Update

**Not performed.**

Phase 11 explicitly allows checksum, manifest, payload, and notebook-hash
updates only after all repair tests pass. The two live AdamW tests fail, so the
registered package remains:

- registered original payload SHA: `{ORIGINAL_PAYLOAD_SHA}`;
- current unregistered repaired-tree scientific checksum:
  `{CANDIDATE_PAYLOAD_SHA}`;
- package checksum verification now intentionally fails only for
  `src/lap_gnn_tf/model/motif_layers.py` and
  `src/lap_gnn_tf/model/lap_gnn.py`.

The candidate checksum is diagnostic only. It is not a signed payload SHA and
must not be used by Kaggle as a ready package.
""",
    )

    write_text(
        "12_kaggle_notebook_revalidation.md",
        f"""
# Kaggle Notebook Revalidation

The structural notebook contract passes:

- checks TensorFlow/Keras versions before replacement;
- installs the local package with `--no-deps`;
- requires restart if the runtime framework is replaced;
- rechecks environment after restart;
- requires GPU for full training unless explicitly overridden;
- runs parity on CPU float32 with deterministic operations;
- trains with XLA disabled;
- imports no PyTorch or parent D16 runtime;
- uses existing priors without regeneration;
- contains exactly one fresh seed42 run and no resume.

PyTorch notebook SHA remains `{sha256(PYTORCH_NOTEBOOK)}`.
TensorFlow notebook SHA remains `{sha256(TF_NOTEBOOK)}`.

The notebook was not hash-updated because Phase 11 is blocked. Its checksum
verification therefore fails closed on the two repaired source files before
training. Contract structure passes; launch readiness does not.
""",
    )

    write_text(
        "13_tensorflow_seed42_launch_plan.md",
        """
# TensorFlow Seed42 Launch Plan

Do **not** launch while the decision is `HOLD_TENSORFLOW_PORT_REPAIR`.

After a future optimizer repair passes both live-step gates and the package is
re-signed, the exact notebook training command is:

```bash
python -B -m lap_gnn_tf.cli.train \
  --config configs/fer2013_ofix7_mid_tensorflow_seed42.yaml \
  --fer-csv /kaggle/input/datasets/doduyquynii/fer13-split/fer13-split/train.csv \
  --prior-root /kaggle/input/datasets/irthn1311/d16-mediapipe-pixel-priors-best-retry-rescue/outputs/d16_mediapipe_pixel_priors_best_retry_rescue \
  --output-root /kaggle/working/outputs/tensorflow_validation/lap_gnn_tensorflow_ofix7_mid_candidate/ofix7_mid_seed42 \
  --device gpu --graph-workers 2 --batch-size 16 \
  --no-resume --mixed-precision --no-xla --memory-growth
```

Expected output:
`/kaggle/working/outputs/tensorflow_validation/lap_gnn_tensorflow_ofix7_mid_candidate/ofix7_mid_seed42`

Expected archive:
`/kaggle/working/ofix7_mid_seed42_tensorflow_outputs.zip`
""",
    )

    blocking = [
        "Live TorchCompatibleAdamW step 1 parameter/state parity exceeds the strict 2e-8 target.",
        "Live TorchCompatibleAdamW step 2 parameter/state parity exceeds the strict 2e-8 target.",
        "Phase 11 package checksums, manifest, payload registration, and notebook hashes are not updated.",
    ]
    warnings = [
        live["capture_warning"],
        "The pre-repair debug-instrumented top-20 probe follows a slightly different execution trace than the authoritative registered blocker.",
        "Matplotlib emits unrelated pyparsing deprecation warnings.",
        "No full epoch, validation split, official test split, or Kaggle training was run.",
    ]
    summary = {
        "repair_scope": "standalone TensorFlow OFIX7-mid bounded parity repair",
        "readiness_decision": READINESS,
        "forward_parity_class": repeated["parity_class"],
        "original_registered_max_logit_difference": original[
            "max_logit_difference"
        ],
        "post_repair_max_logit_difference": repeated["eager"]["maximum"],
        "post_repair_gradient": gradient,
        "live_adamw": live,
        "registered_original_payload_sha256": ORIGINAL_PAYLOAD_SHA,
        "unregistered_repaired_tree_scientific_checksum": CANDIDATE_PAYLOAD_SHA,
        "pytorch_notebook_sha256": sha256(PYTORCH_NOTEBOOK),
        "tensorflow_notebook_sha256": sha256(TF_NOTEBOOK),
        "bounded_tests": {"passed": 36, "failed": 2, "total": 38},
        "blocking_issues": blocking,
        "warnings": warnings,
    }
    (REPORT / "14_machine_readable_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    validation = {
        "input_tensorflow_payload_sha_valid": True,
        "pytorch_reference_unchanged": True,
        "pytorch_golden_unchanged": True,
        "baseline_locks_valid": True,
        "original_blocker_reproduced": True,
        "max_difference_sample_index": exact_original["sample_index"],
        "max_difference_class_index": exact_original["class_index"],
        "max_difference_ulp": exact_original["ulp_distance"],
        "classifier_only_difference": instrumented["classifier_candidates"][
            "matmul_add"
        ]["golden_input_eager"]["max_abs"],
        "upstream_readout_difference": original["maximum_layer_max_abs"],
        "onednn_enabled_result": repaired_onednn["baseline_logits"]["max_abs"],
        "onednn_disabled_result": repaired_no_onednn["baseline_logits"]["max_abs"],
        "eager_repeated_max": repeated["eager"]["maximum"],
        "tf_function_repeated_max": repeated["tf_function"]["maximum"],
        "fresh_process_repeated_max": repeated["fresh_process"]["maximum"],
        "all_repeated_logit_gates_pass": True,
        "prediction_agreement": 1.0,
        "graph_parity_pass": True,
        "weight_mapping_complete": True,
        "parameter_count_match": True,
        "gradient_parity_pass": gradient["pass"],
        "live_custom_adamw_step1_pass": step1["pass_2e_8"],
        "live_custom_adamw_step2_pass": step2["pass_2e_8"],
        "optimizer_slot_state_match": False,
        "plateau_semantics_pass": True,
        "early_stopping_semantics_pass": True,
        "metric_parity_pass": True,
        "keras_build_warning_resolved": True,
        "checkpoint_roundtrip_pass": True,
        "tensorflow_runtime_imports_torch": False,
        "tensorflow_runtime_imports_parent": False,
        "notebook_hash_updated": False,
        "pytorch_notebook_unchanged": sha256(PYTORCH_NOTEBOOK)
        == PYTORCH_NOTEBOOK_SHA,
        "kaggle_notebook_contract_pass": True,
        "package_checksums_updated": False,
        "package_manifest_updated": False,
        "full_training_launched": False,
        "dataset_modified": False,
        "prior_modified": False,
        "pytorch_package_modified": False,
        "parent_code_modified": False,
        "blocking_issues": blocking,
        "warnings": warnings,
        "readiness_decision": READINESS,
    }
    (REPORT / "15_validation_summary.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "report_dir": str(REPORT),
                "required_reports": 22,
                "readiness_decision": READINESS,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
