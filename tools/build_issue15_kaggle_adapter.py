"""Build the preregistered Issue #15 Kaggle T4 pre-run adapter notebook."""

from __future__ import annotations

import copy
import importlib.util
import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ISSUE11_BUILDER = ROOT / "tools" / "build_issue11_kaggle_adapter.py"
NOTEBOOK_PATH = ROOT / "notebooks" / "kaggle-issue15-direct-part-decomposition.ipynb"


def _source(text: str) -> list[str]:
    return textwrap.dedent(text).lstrip("\n").splitlines(keepends=True)


def _set_source(cell: dict, text: str) -> None:
    cell["source"] = _source(text)


def _replace_required(cell: dict, old: str, new: str) -> None:
    source = "".join(cell["source"])
    if old not in source:
        raise RuntimeError(f"Issue #15 adapter template drift; missing {old!r}")
    cell["source"] = source.replace(old, new).splitlines(keepends=True)


def _load_issue11_builder():
    spec = importlib.util.spec_from_file_location("_issue11_adapter_template", ISSUE11_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load adapter template: {ISSUE11_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_notebook() -> dict:
    """Return a self-contained, deterministic, deliberately unexecuted notebook."""

    notebook = copy.deepcopy(_load_issue11_builder().build_notebook())
    cells = notebook["cells"]

    _set_source(
        cells[0],
        """
        # Issue #15: fixed-checkpoint direct-part pathway decomposition adapter

        This is the dedicated **pre-run adapter** for the preregistered Step 8
        experiment. It is intentionally unexecuted in this PR. After research-lead
        approval, it runs the reviewed Step 7 D0-D5 harness exactly once on Kaggle
        GPU T4, using the exact Issue #7 epoch-31 checkpoint and validation assets only.

        Required Kaggle Inputs and resolved reads:

        - FER split input: `/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split`;
          only `val.csv` is opened.
        - MediaPipe prior input:
          `/kaggle/input/datasets/irthn1311/d16-mediapipe-pixel-priors-best-retry-rescue/outputs/d16_mediapipe_pixel_priors_best_retry_rescue`;
          only `val/*.npz` and shared root schema/name metadata required by the frozen
          loader are used.
        - Clean graph cache input:
          `/kaggle/input/datasets/irthn1311/ofix7-mid-seed42-records`; only
          `val/index.json`, its referenced validation shards, and root
          `CACHE_COMPLETE.json` are used. The completion marker is disclosed as
          shared non-sample aggregate metadata and may summarize other splits.
        - Issue #7 artifact input: attach one separate read-only Kaggle Input
          containing `best_val_accuracy.keras`, `best_val_accuracy.metadata.json`,
          and `resolved_config.json`. Its mount name is not trusted; each artifact is
          located outside the public sample inputs by basename and exact SHA-256, and
          ambiguous matches are rejected.

        Internet is required only to clone the exact registered scientific commit and,
        if the Kaggle image is incompatible, install the registered dependencies. All
        datasets, cache records, and checkpoint/config artifacts are offline inputs.

        Compact outputs:

        - report: `/kaggle/working/tf_step8_direct_part_decomposition.md`
        - archive: `/kaggle/working/tf_step8_direct_part_decomposition_kaggle_t4.zip`

        D1-D4 are nonlinear functional sensitivity diagnostics conditional on the
        fixed official MediaPipe-derived scaffold. They are non-additive, not causal
        contributions, and not evidence of a prior-free graph.
        """,
    )
    _set_source(cells[1], "## 1. Preregistered constants and immutable paths\n")
    _set_source(
        cells[2],
        """
        from pathlib import Path

        REPO_URL = "https://github.com/Irthn1311/FER2013_Graph.git"
        EXPECTED_SCIENTIFIC_BASE_COMMIT = "d14e7e1e3eec2ffbd5339b5a3bd0d5db5ab3de8b"
        EXPECTED_EXECUTION_COMMIT = EXPECTED_SCIENTIFIC_BASE_COMMIT
        TF_PACKAGE_RELATIVE = Path("standalone/lap_gnn_tensorflow_ofix7_mid_candidate")
        PROBE_TOOL_RELATIVE = TF_PACKAGE_RELATIVE / "tools/evaluate_fixed_checkpoint_direct_part_decomposition_probe.py"
        SUPPORT_TOOL_RELATIVE = TF_PACKAGE_RELATIVE / "tools/evaluate_fixed_checkpoint_prior_probe.py"

        EXPECTED_PROBE_TOOL_SHA256 = "e611b74ac143c50149326c9761b35177183a09b3cf44b52ab018b01ed3d87ffd"
        EXPECTED_SUPPORT_TOOL_SHA256 = "3a033c977a29e102cfed75282ae7c1062f41feac8bef1b955ae425ec7e4004b3"
        EXPECTED_SCIENTIFIC_PAYLOAD_SHA256 = "286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e"
        EXPECTED_EXECUTION_CONTRACT_SHA256 = "14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22"
        EXPECTED_CONFIG_HASH = "a4038682bf09c03786e86119001cf5f81ac5fec25d09062e6a0866484c32a3cf"
        EXPECTED_GRAPH_SIGNATURE = "1c7597b170fd8604056ab7787fd2880d6e84f3025962fc4b6c8fb3e3faf8e1e8"
        EXPECTED_FEATURE_SIGNATURE = "752538062fa2e40d9615c650c529e9f4117f33a030b74d281b5b21fa573731fc"
        EXPECTED_PRIOR_SIGNATURE = "ea888bab9c003af9b279719025da7c39f90537179411326c2c3119fc8c3f0824"
        EXPECTED_DATASET_SPLIT_SIGNATURE = "fer2013_train28709_val3589_test3589"

        EXPECTED_ARTIFACTS = {
            "checkpoint": {
                "basename": "best_val_accuracy.keras",
                "sha256": "9ec11bb819f97e4fbda432f68da76c1201b8a3f9e06fae9eb30489a528d6ac16",
            },
            "checkpoint_metadata": {
                "basename": "best_val_accuracy.metadata.json",
                "sha256": "e62cf8c86f0d6a56c3041911de6397d18f47276f79f03c6a50ca71fa47300a37",
            },
            "resolved_config": {
                "basename": "resolved_config.json",
                "sha256": "3c028dd2f32ebed3a252544e170220b150b5e29920cea865924dddce6aef5a32",
            },
        }

        EXPECTED_CHECKPOINT_EPOCH = 31
        EXPECTED_SEED = 42
        EXPECTED_VALIDATION_SAMPLES = 3589
        CONDITION_D0 = "official_manual_forward"
        CONDITION_D1 = "context_local_prior_neutralized"
        CONDITION_D2 = "readout_local_prior_neutralized"
        CONDITION_D3 = "local_part_residual_zero"
        CONDITION_D4 = "local_motif_validity_off"
        CONDITION_D5 = "full_direct_part_zero_anchor"
        CONDITIONS = (CONDITION_D0, CONDITION_D1, CONDITION_D2, CONDITION_D3, CONDITION_D4, CONDITION_D5)
        D0_REFERENCE = {"accuracy": 0.63137364168292, "macro_f1": 0.5932591901893336, "loss": 1.1537981724317095}
        D5_REFERENCE = {"accuracy": 0.27751462803009197, "macro_f1": 0.19745892656222366, "loss": 1.757720434560185}
        REFERENCE_TOLERANCE = {"accuracy": 0.001, "macro_f1": 0.001, "loss": 0.005}
        NATIVE_MANUAL_TOLERANCE = {"prediction_agreement": 1.0, "max_abs_logit_difference": 1e-5, "max_abs_probability_difference": 1e-6}

        KAGGLE_INPUT_ROOT = Path("/kaggle/input")
        FER_DATASET_MOUNT = Path("/kaggle/input/datasets/doduyquynii/fer13-split")
        PRIOR_DATASET_MOUNT = Path("/kaggle/input/datasets/irthn1311/d16-mediapipe-pixel-priors-best-retry-rescue")
        CACHE_DATASET_MOUNT = Path("/kaggle/input/datasets/irthn1311/ofix7-mid-seed42-records")
        FER_SPLIT_ROOT = FER_DATASET_MOUNT / "fer13-split"
        FER_VAL_CSV = FER_SPLIT_ROOT / "val.csv"
        PRIOR_ROOT = PRIOR_DATASET_MOUNT / "outputs/d16_mediapipe_pixel_priors_best_retry_rescue"
        PRIOR_VAL_DIR = PRIOR_ROOT / "val"
        GRAPH_CACHE_ROOT = CACHE_DATASET_MOUNT
        GRAPH_CACHE_VAL_DIR = GRAPH_CACHE_ROOT / "val"
        CACHE_COMPLETE_PATH = GRAPH_CACHE_ROOT / "CACHE_COMPLETE.json"
        PUBLIC_SAMPLE_INPUT_ROOTS = (FER_DATASET_MOUNT, PRIOR_DATASET_MOUNT, CACHE_DATASET_MOUNT)

        WORKING = Path("/kaggle/working")
        PROJECT_PATH = WORKING / "FER2013_Graph"
        TF_PACKAGE_PATH = PROJECT_PATH / TF_PACKAGE_RELATIVE
        PROBE_TOOL_PATH = PROJECT_PATH / PROBE_TOOL_RELATIVE
        SUPPORT_TOOL_PATH = PROJECT_PATH / SUPPORT_TOOL_RELATIVE
        RUN_ROOT = WORKING / "tf_step8_direct_part_decomposition"
        PROBE_OUTPUT_ROOT = RUN_ROOT / "probe"
        ADAPTER_METADATA_ROOT = RUN_ROOT / "adapter_metadata"
        REPORT_PATH = WORKING / "tf_step8_direct_part_decomposition.md"
        ARCHIVE_PATH = WORKING / "tf_step8_direct_part_decomposition_kaggle_t4.zip"

        EVAL_BATCH_SIZE = 32
        GRAPH_WORKERS = 2
        GRAPH_CACHE_SIZE = 64
        TESTED_TENSORFLOW = "2.18.1"
        TESTED_KERAS = "3.15.0"
        """,
    )

    _replace_required(
        cells[4],
        'run_checked([\n    "git", "clone", "--branch", REPO_BRANCH, "--single-branch",\n    clone_url, PROJECT_PATH,\n])',
        'run_checked(["git", "clone", "--no-checkout", clone_url, PROJECT_PATH])',
    )
    _replace_required(cells[4], '    PROBE_TOOL_PATH,\n', '    PROBE_TOOL_PATH,\n    SUPPORT_TOOL_PATH,\n')
    _replace_required(
        cells[4],
        'if sha256(PROBE_TOOL_PATH) != EXPECTED_PROBE_TOOL_SHA256:\n    raise RuntimeError("Reviewed Step 5 probe tool drift")',
        'if sha256(PROBE_TOOL_PATH) != EXPECTED_PROBE_TOOL_SHA256:\n    raise RuntimeError("Reviewed Step 7 probe tool drift")\nif sha256(SUPPORT_TOOL_PATH) != EXPECTED_SUPPORT_TOOL_SHA256:\n    raise RuntimeError("Reviewed Step 6 support-tool drift")',
    )
    _replace_required(
        cells[4],
        '    "probe_tool_sha256": sha256(PROBE_TOOL_PATH),\n',
        '    "step7_probe_tool_sha256": sha256(PROBE_TOOL_PATH),\n    "step6_support_tool_sha256": sha256(SUPPORT_TOOL_PATH),\n',
    )
    _replace_required(cells[6], "Issue #11 requires Kaggle GPU T4", "Issue #15 requires Kaggle GPU T4")

    _replace_required(cells[8], "Issue #7 reference validation metrics drift", "Issue #7 metadata validation metrics drift")
    reference_block = '''metadata_validation_metrics = checkpoint_metadata.get("validation_metrics") or {}
reference_metric_mismatches = {
    key: {"expected": expected, "actual": metadata_validation_metrics.get(key)}
    for key, expected in C0_REFERENCE.items()
    if metadata_validation_metrics.get(key) != expected
}
if reference_metric_mismatches:
    raise RuntimeError(
        f"Issue #7 metadata validation metrics drift: {reference_metric_mismatches}"
    )
'''
    _replace_required(cells[8], reference_block, "")

    for index in (12,):
        _replace_required(cells[index], "Issue #11", "Issue #15")
    _replace_required(cells[10], '    "issue": 11,', '    "issue": 15,')
    _replace_required(
        cells[10],
        '    "probe_tool_sha256": sha256(PROBE_TOOL_PATH),\n',
        '    "step7_probe_tool_sha256": sha256(PROBE_TOOL_PATH),\n    "step6_support_tool_sha256": sha256(SUPPORT_TOOL_PATH),\n',
    )
    _replace_required(cells[10], "READY_FOR_ISSUE11_REGISTERED_PROBE", "READY_FOR_ISSUE15_REGISTERED_PROBE")
    _replace_required(cells[10], '"registered_full_run": True,', '"registered_full_run": True,\n    "registered_condition_order": list(CONDITIONS),')

    _set_source(
        cells[11],
        """
        ## 6. One registered full validation probe

        **Pre-run review gate:** do not execute this cell until this draft PR is
        approved by the research lead. It invokes the SHA-verified Step 7 harness
        exactly once and deliberately supplies no `--limit-val-batches` argument.
        """,
    )
    _replace_required(cells[12], "Reviewed probe tool must be invoked exactly once", "Reviewed Step 7 harness must be invoked exactly once")
    _set_source(cells[13], "## 7. Preserve and verify the harness Gate A/B/C evidence\n")
    _set_source(
        cells[14],
        """
        probe_manifest_path = PROBE_OUTPUT_ROOT / "probe_manifest.json"
        equivalence_path = PROBE_OUTPUT_ROOT / "native_manual_d0_equivalence.json"
        integrity_path = PROBE_OUTPUT_ROOT / "intervention_integrity.json"
        probe_manifest = json.loads(probe_manifest_path.read_text(encoding="utf-8"))
        native_manual_equivalence = json.loads(equivalence_path.read_text(encoding="utf-8"))
        integrity = json.loads(integrity_path.read_text(encoding="utf-8"))

        expected_output_names = {
            *(f"validation_metrics_{condition}.json" for condition in CONDITIONS),
            "paired_validation_predictions.csv",
            "native_manual_d0_equivalence.json",
            "intervention_integrity.json",
            "probe_manifest.json",
        }
        actual_output_names = {path.name for path in PROBE_OUTPUT_ROOT.iterdir() if path.is_file()}
        if actual_output_names != expected_output_names:
            raise RuntimeError(
                f"Step 7 output inventory drift: expected={sorted(expected_output_names)}, actual={sorted(actual_output_names)}"
            )

        condition_metrics = {}
        for condition in CONDITIONS:
            payload = json.loads(
                (PROBE_OUTPUT_ROOT / f"validation_metrics_{condition}.json").read_text(encoding="utf-8")
            )
            if payload.get("split") != "val" or payload.get("condition") != condition:
                raise RuntimeError(f"Condition output identity mismatch: {condition}")
            condition_metrics[condition] = payload["metrics"]

        paired_path = PROBE_OUTPUT_ROOT / "paired_validation_predictions.csv"
        with paired_path.open("r", encoding="utf-8", newline="") as stream:
            paired_reader = csv.DictReader(stream)
            paired_header = tuple(paired_reader.fieldnames or ())
            paired_sample_count = sum(1 for _ in paired_reader)
        expected_paired_columns = {"sample_id", "label"}
        for condition in CONDITIONS:
            expected_paired_columns.add(f"{condition}_prediction")
            expected_paired_columns.update(f"{condition}_probability_{index}" for index in range(7))
        if set(paired_header) != expected_paired_columns:
            raise RuntimeError("Paired prediction schema drift")
        if paired_sample_count != EXPECTED_VALIDATION_SAMPLES:
            raise RuntimeError(f"Expected {EXPECTED_VALIDATION_SAMPLES} paired samples, got {paired_sample_count}")

        gates = probe_manifest.get("registered_gates_and_diagnostics") or {}
        gate_a = gates.get("gate_a_native_manual_equivalence") or {}
        gate_b = gates.get("gate_b_d0_reference") or {}
        gate_c = gates.get("gate_c_d5_anchor") or {}
        if gates.get("status") != "VALID_REGISTERED_DECOMPOSITION":
            raise RuntimeError(f"Registered decomposition gates did not pass: {gates.get('status')}")
        if gate_a.get("status") != "PASS" or gate_a.get("pass") is not True:
            raise RuntimeError("Gate A native/manual equivalence failed")
        if gate_b.get("status") != "PASS" or gate_c.get("status") != "PASS":
            raise RuntimeError("Gate B/C reference reproduction failed")
        if gate_a.get("tolerances") != NATIVE_MANUAL_TOLERANCE:
            raise RuntimeError("Gate A tolerance drift")
        if (
            gate_a.get("prediction_agreement") != 1.0
            or float(gate_a.get("max_abs_logit_difference", float("inf"))) > NATIVE_MANUAL_TOLERANCE["max_abs_logit_difference"]
            or float(gate_a.get("max_abs_probability_difference", float("inf"))) > NATIVE_MANUAL_TOLERANCE["max_abs_probability_difference"]
        ):
            raise RuntimeError("Gate A numeric evidence failed")

        def verify_reference_gate(gate, reference, name):
            checks = gate.get("checks") or {}
            for metric, expected in reference.items():
                check = checks.get(metric) or {}
                if (
                    check.get("reference") != expected
                    or check.get("tolerance") != REFERENCE_TOLERANCE[metric]
                    or check.get("pass") is not True
                ):
                    raise RuntimeError(f"{name} locked {metric} evidence drift")

        if (
            gate_b.get("sample_count") != EXPECTED_VALIDATION_SAMPLES
            or gate_b.get("required_sample_count") != EXPECTED_VALIDATION_SAMPLES
            or gate_b.get("sample_count_exact") is not True
        ):
            raise RuntimeError("Gate B sample-count evidence drift")
        verify_reference_gate(gate_b, D0_REFERENCE, "Gate B")
        verify_reference_gate(gate_c, D5_REFERENCE, "Gate C")
        if gates.get("per_path_diagnostics") is None or gates.get("overall_decision") is None:
            raise RuntimeError("Registered diagnostics missing after passed gates")

        expected_manifest_identity = {
            "split": "val",
            "sample_count": EXPECTED_VALIDATION_SAMPLES,
            "limit_val_batches": None,
            "bounded_smoke_only": False,
            "condition_order": list(CONDITIONS),
            "scientific_payload_sha256": EXPECTED_SCIENTIFIC_PAYLOAD_SHA256,
        }
        manifest_mismatches = {
            key: {"expected": expected, "actual": probe_manifest.get(key)}
            for key, expected in expected_manifest_identity.items()
            if probe_manifest.get(key) != expected
        }
        if manifest_mismatches:
            raise RuntimeError(f"Probe manifest identity drift: {manifest_mismatches}")
        required_integrity = {
            "source_batches_mutated": False,
            "message_passing_inputs_changed": False,
            "node_edge_features_changed": False,
            "topology_changed": False,
            "paired_original_batch_evaluation": True,
            "checkpoint_unchanged": True,
            "model_weights_unchanged": True,
            "training_or_optimizer_state_created": False,
            "test_split_constructed": False,
        }
        integrity_mismatches = {
            key: {"expected": expected, "actual": integrity.get(key)}
            for key, expected in required_integrity.items()
            if integrity.get(key) != expected
        }
        if integrity_mismatches:
            raise RuntimeError(f"Intervention integrity gate failed: {integrity_mismatches}")
        checks_per_condition = integrity.get("checks_per_condition") or {}
        if (
            set(checks_per_condition) != set(CONDITIONS)
            or any(int(checks_per_condition[condition]) <= 0 for condition in CONDITIONS)
        ):
            raise RuntimeError("Intervention checks-per-condition evidence drift")
        if not probe_manifest.get("checkpoint", {}).get("unchanged"):
            raise RuntimeError("Checkpoint file changed during the registered run")
        if not probe_manifest.get("checkpoint", {}).get("model_weights_unchanged"):
            raise RuntimeError("Model weights changed during the registered run")

        final_evidence = {
            "issue": 15,
            "scientific_base_commit": EXPECTED_SCIENTIFIC_BASE_COMMIT,
            "execution_commit": actual_commit,
            "step7_probe_tool_sha256": sha256(PROBE_TOOL_PATH),
            "step6_support_tool_sha256": sha256(SUPPORT_TOOL_PATH),
            "scientific_payload_sha256": package_manifest["scientific_payload_sha256"],
            "artifact_hashes_before": artifact_hashes_before,
            "artifact_hashes_after": artifact_hashes_after,
            "checkpoint_epoch": EXPECTED_CHECKPOINT_EPOCH,
            "seed": EXPECTED_SEED,
            "environment": environment_payload,
            "sample_count": probe_manifest["sample_count"],
            "batch_count": probe_manifest["batch_count"],
            "condition_metrics": condition_metrics,
            "registered_gates_and_diagnostics": gates,
            "paired_diagnostics": probe_manifest["paired_diagnostics"],
            "native_manual_d0_equivalence": native_manual_equivalence,
            "intervention_integrity": integrity,
            "probe_runtime_resources": probe_manifest["resources"],
            "test_isolation": validation_asset_evidence,
            "training_performed": False,
            "interpretation_boundary": probe_manifest["interpretation_boundary"],
        }
        final_evidence_path = ADAPTER_METADATA_ROOT / "final_evidence.json"
        final_evidence_path.write_text(
            json.dumps(final_evidence, indent=2, sort_keys=True, default=str) + "\\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "gate_a": gate_a["status"],
            "gate_b": gate_b["status"],
            "gate_c": gate_c["status"],
            "overall_decision": gates["overall_decision"],
            "sample_count": probe_manifest["sample_count"],
            "batch_count": probe_manifest["batch_count"],
        }, indent=2))
        """,
    )
    _set_source(cells[15], "## 8. Write the compact Step 8 report\n")
    _set_source(
        cells[16],
        """
        def metric_line(condition):
            metrics = condition_metrics[condition]
            return (
                f"- `{condition}`: accuracy `{metrics['accuracy']}`, macro-F1 "
                f"`{metrics['macro_f1']}`, loss `{metrics['loss']}`, per-class F1 "
                f"`{json.dumps(metrics['per_class_f1'])}`."
            )

        report_lines = [
            "# TensorFlow Step 8 direct-part pathway decomposition",
            "",
            "## Provenance",
            "",
            "- Issue: #15.",
            f"- Frozen scientific base/execution commit: `{EXPECTED_SCIENTIFIC_BASE_COMMIT}`.",
            f"- Step 7 probe-tool SHA-256: `{sha256(PROBE_TOOL_PATH)}`.",
            f"- Step 6 support-tool SHA-256: `{sha256(SUPPORT_TOOL_PATH)}`.",
            f"- Frozen scientific payload SHA-256: `{package_manifest['scientific_payload_sha256']}`.",
            f"- Checkpoint SHA-256: `{artifact_hashes_after['checkpoint']}`.",
            f"- Checkpoint metadata SHA-256: `{artifact_hashes_after['checkpoint_metadata']}`.",
            f"- Resolved config SHA-256: `{artifact_hashes_after['resolved_config']}`.",
            f"- Checkpoint epoch/seed: `{EXPECTED_CHECKPOINT_EPOCH}` / `{EXPECTED_SEED}`.",
            f"- Runtime: Kaggle T4, TensorFlow `{environment_payload['tensorflow']}`, Keras `{environment_payload['keras']}`.",
            f"- GPU memory growth: requested `{probe_manifest['resources']['memory_growth_requested']}`, status `{probe_manifest['resources']['memory_growth_status']}`.",
            "",
            "## Mandatory gates",
            "",
            f"- Gate A native/manual D0 equivalence: `{json.dumps(gate_a, sort_keys=True)}`.",
            f"- Gate B D0 reference reproduction: `{json.dumps(gate_b, sort_keys=True)}`.",
            f"- Gate C D5 anchor reproduction: `{json.dumps(gate_c, sort_keys=True)}`.",
            "",
            "## D0-D5 validation metrics and per-class F1",
            "",
            *(metric_line(condition) for condition in CONDITIONS),
            "",
            "## Preregistered primary diagnostics",
            "",
            f"- Per-path D1-D4 deltas and labels: `{json.dumps(gates['per_path_diagnostics'], sort_keys=True)}`.",
            f"- Overall decision: `{gates['overall_decision']}`.",
            f"- Non-additivity warning: {gates['non_additivity_warning']}",
            "",
            "## Paired diagnostics",
            "",
            f"- D1-D5 paired prediction/correctness diagnostics versus D0: `{json.dumps(probe_manifest['paired_diagnostics'], sort_keys=True)}`.",
            "",
            "## Integrity and boundaries",
            "",
            f"- Sample/batch count: `{probe_manifest['sample_count']}` / `{probe_manifest['batch_count']}`.",
            f"- Native/manual D0 evidence: `{json.dumps(native_manual_equivalence, sort_keys=True)}`.",
            f"- Intervention integrity and checks per condition: `{json.dumps(integrity, sort_keys=True)}`.",
            f"- Checkpoint file SHA before/after: `{probe_manifest['checkpoint']['sha256_before']}` / `{probe_manifest['checkpoint']['sha256_after']}`.",
            f"- Model-weight SHA before/after: `{probe_manifest['checkpoint']['model_weights_sha256_before']}` / `{probe_manifest['checkpoint']['model_weights_sha256_after']}`.",
            "- Test CSV/prior/cache records/labels/predictions/metrics/inference accessed: `false`.",
            "- Shared cache-root `CACHE_COMPLETE.json` accessed: `true`, solely as required non-sample aggregate loader metadata; it may summarize other splits.",
            "- Training, fine-tuning, optimizer/gradient steps, raw-prior corruption, graph rebuild, topology changes, and node/edge feature changes: `false`.",
            "- D1-D4 effects are nonlinear functional sensitivity diagnostics under the fixed official MediaPipe-derived scaffold. They must not be summed or treated as additive causal contributions.",
            "- The result is not proof of shortcut learning, not causal proof of the Issue #7 generalization gap, not model selection, and not a prior-free/MediaPipe-free graph.",
        ]
        REPORT_PATH.write_text("\\n".join(report_lines) + "\\n", encoding="utf-8")
        print("report:", REPORT_PATH)
        """,
    )
    _set_source(cells[17], "## 9. Archive compact validation-only evidence\n")
    for index, cell in enumerate(cells):
        cell["id"] = f"issue15-{index:02d}"
    return notebook


def main() -> None:
    NOTEBOOK_PATH.write_text(
        json.dumps(build_notebook(), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
