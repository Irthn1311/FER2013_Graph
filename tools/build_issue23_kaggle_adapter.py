"""Build the preregistered Issue #23 Kaggle T4 pre-run adapter notebook."""

from __future__ import annotations

import copy
import importlib.util
import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ISSUE15_BUILDER = ROOT / "tools" / "build_issue15_kaggle_adapter.py"
NOTEBOOK_PATH = (
    ROOT / "notebooks" / "kaggle-issue23-local-residual-slot-decomposition.ipynb"
)


def _source(text: str) -> list[str]:
    return textwrap.dedent(text).lstrip("\n").splitlines(keepends=True)


def _set_source(cell: dict, text: str) -> None:
    cell["source"] = _source(text)


def _replace_required(cell: dict, old: str, new: str) -> None:
    source = "".join(cell["source"])
    if old not in source:
        raise RuntimeError(f"Issue #23 adapter template drift; missing {old!r}")
    cell["source"] = source.replace(old, new).splitlines(keepends=True)


def _load_issue15_builder():
    spec = importlib.util.spec_from_file_location(
        "_issue23_issue15_adapter_template", ISSUE15_BUILDER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load adapter template: {ISSUE15_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _transform_inherited_sources(cells: list[dict]) -> None:
    replacements = (
        ("Issue #15", "Issue #23"),
        ("issue15", "issue23"),
        ("tf_step8_direct_part_decomposition", "tf_step9_local_residual_slot_decomposition"),
        ("create_step8_archive", "create_step9_archive"),
        ("step7_subprocess.log", "step9_subprocess.log"),
        ("PROBE_TOOL_RELATIVE", "STEP9_TOOL_RELATIVE"),
        ("PROBE_TOOL_PATH", "STEP9_TOOL_PATH"),
        ("EXPECTED_PROBE_TOOL_SHA256", "EXPECTED_STEP9_TOOL_SHA256"),
        ("SUPPORT_TOOL_RELATIVE", "STEP6_SUPPORT_RELATIVE"),
        ("SUPPORT_TOOL_PATH", "STEP6_SUPPORT_PATH"),
        ("EXPECTED_SUPPORT_TOOL_SHA256", "EXPECTED_STEP6_SUPPORT_SHA256"),
        ("step7_probe_tool_sha256", "step9_harness_sha256"),
        ("step6_support_tool_sha256", "step6_support_sha256"),
        ("CONDITION_D0", "CONDITION_S0"),
        ("CONDITION_D1", "CONDITION_S1"),
        ("CONDITION_D2", "CONDITION_S2"),
        ("CONDITION_D3", "CONDITION_S3"),
        ("CONDITION_D4", "CONDITION_S4"),
        ("CONDITION_D5", "CONDITION_S5"),
        ("D0_REFERENCE", "S0_REFERENCE"),
        ("D5_REFERENCE", "S5_REFERENCE"),
        ("native_manual_d0_equivalence", "native_manual_s0_equivalence"),
        ("gate_a_native_manual_equivalence", "gate_a_native_manual_s0_equivalence"),
        ("gate_b_d0_reference", "gate_b_s0_reference"),
        ("gate_c_d5_anchor", "gate_c_s5_d3_anchor"),
        ("VALID_REGISTERED_DECOMPOSITION", "VALID_REGISTERED_SLOT_DECOMPOSITION"),
        ("per_path_diagnostics", "per_slot_diagnostics"),
        ("per-path", "per-slot"),
        ("D0-D5", "S0-S5"),
        ("D1-D5", "S1-S5"),
        ("D1-D4", "S1-S4"),
        ("D0", "S0"),
        ("D5", "S5"),
        ("Step 8 direct-part pathway decomposition", "Step 9 local residual-slot decomposition"),
        ("direct-part pathway decomposition", "local residual-slot decomposition"),
        ("Step-7 subprocess", "Step-9 subprocess"),
        ("Step-7 SHA-256", "Step-9 SHA-256"),
        ("Step 7 output", "Step 9 output"),
        ("Step 7 harness", "Step 9 harness"),
        ("Reviewed Step 7", "Reviewed Step 9"),
        ("registered Step 7", "registered Step 9"),
        ("Step 7 probe-tool", "Step 9 harness"),
        ("Unknown Step-7 wrapper status", "Unknown Step-9 wrapper status"),
        ("Step 8 report", "Step 9 report"),
        ("Per-path S1-S4", "Per-slot S1-S4"),
        ("READY_FOR_ISSUE15_REGISTERED_PROBE", "READY_FOR_ISSUE23_REGISTERED_PROBE"),
        ('"issue": 15', '"issue": 23'),
        ("- Issue: #15.", "- Issue: #23."),
    )
    for cell in cells:
        source = "".join(cell.get("source", []))
        for old, new in replacements:
            source = source.replace(old, new)
        cell["source"] = source.splitlines(keepends=True)


def build_notebook() -> dict:
    """Return a deterministic, self-contained and deliberately unexecuted notebook."""

    notebook = copy.deepcopy(_load_issue15_builder().build_notebook())
    cells = notebook["cells"]
    _transform_inherited_sources(cells)

    _set_source(
        cells[0],
        """
        # Issue #23: fixed-checkpoint local residual-slot decomposition adapter

        This dedicated **pre-run adapter** implements the preregistered Step 9
        execution contract. It is intentionally unexecuted. After explicit
        research-lead approval it will invoke the SHA-verified Step-9 S0-S5 harness
        exactly once on Kaggle GPU T4, using only the exact Issue #7 epoch-31
        checkpoint and validation assets.

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
          `CACHE_COMPLETE.json` are used. The completion marker is disclosed shared
          non-sample aggregate metadata and may summarize other splits.
        - Issue #7 artifact input: attach one separate read-only Kaggle Input
          containing `best_val_accuracy.keras`, `best_val_accuracy.metadata.json`,
          and `resolved_config.json`. Its mount name is not trusted; each artifact is
          located outside public sample inputs by basename plus exact SHA-256, and
          zero or ambiguous matches are rejected.

        Internet is required only to clone the exact reviewed execution source and,
        if the Kaggle image differs, install the registered dependencies. FER,
        priors, cache, checkpoint, metadata, and resolved config are offline inputs.

        Compact outputs:

        - report: `/kaggle/working/tf_step9_local_residual_slot_decomposition.md`
        - archive: `/kaggle/working/tf_step9_local_residual_slot_decomposition_kaggle_t4.zip`

        S1-S4 are nonlinear functional sensitivities of learned pooled local
        residual slots under the fixed official MediaPipe-derived scaffold. They
        must not be summed or divided by S5, and they are not causal percentage
        contributions or MediaPipe-prior isolation.

        The first authorized Issue #23 run remains classified as a
        **PRE-INTERVENTION TECHNICAL HARNESS FAILURE**. Its reviewed archive SHA-256
        is `ff19925fc4ad6f6d8144512979dd2f725355cacc31303a848bd77037d4a41b17`;
        `scientific_result_valid=false`, scientific interpretation is null, and no
        S0-S5 scientific outcome exists from that attempt.
        """,
    )
    _set_source(cells[1], "## 1. Preregistered constants and immutable paths\n")
    _set_source(
        cells[2],
        """
        from pathlib import Path

        REPO_URL = "https://github.com/Irthn1311/FER2013_Graph.git"
        EXPECTED_SCIENTIFIC_BASE_COMMIT = "753ae1a27b9e4467d11c5d68cb416df63de29ff5"
        EXPECTED_EXECUTION_COMMIT = "73a5bd6fe1210b379287ca9e0048526ff682e7a9"
        TF_PACKAGE_RELATIVE = Path("standalone/lap_gnn_tensorflow_ofix7_mid_candidate")
        STEP9_TOOL_RELATIVE = TF_PACKAGE_RELATIVE / "tools/evaluate_fixed_checkpoint_local_residual_slot_decomposition_probe.py"
        STEP7_TOOL_RELATIVE = TF_PACKAGE_RELATIVE / "tools/evaluate_fixed_checkpoint_direct_part_decomposition_probe.py"
        STEP6_SUPPORT_RELATIVE = TF_PACKAGE_RELATIVE / "tools/evaluate_fixed_checkpoint_prior_probe.py"

        EXPECTED_STEP9_TOOL_SHA256 = "50a310f622cdf9dccf13eff4edf6394f1d39b8ccf315dce5ede07d0a45bdd77a"
        EXPECTED_STEP7_TOOL_SHA256 = "c0b1df778e469665dd6437c58831d29dcc34fbde44231db75894c5469a1ade78"
        EXPECTED_STEP6_SUPPORT_SHA256 = "3a033c977a29e102cfed75282ae7c1062f41feac8bef1b955ae425ec7e4004b3"
        EXPECTED_SCIENTIFIC_PAYLOAD_SHA256 = "286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e"
        EXPECTED_EXECUTION_CONTRACT_SHA256 = "14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22"
        EXPECTED_CONFIG_HASH = "a4038682bf09c03786e86119001cf5f81ac5fec25d09062e6a0866484c32a3cf"
        EXPECTED_GRAPH_SIGNATURE = "1c7597b170fd8604056ab7787fd2880d6e84f3025962fc4b6c8fb3e3faf8e1e8"
        EXPECTED_FEATURE_SIGNATURE = "752538062fa2e40d9615c650c529e9f4117f33a030b74d281b5b21fa573731fc"
        EXPECTED_PRIOR_SIGNATURE = "ea888bab9c003af9b279719025da7c39f90537179411326c2c3119fc8c3f0824"
        EXPECTED_DATASET_SPLIT_SIGNATURE = "fer2013_train28709_val3589_test3589"
        FAILED_ATTEMPT_CLASSIFICATION = "PRE-INTERVENTION TECHNICAL HARNESS FAILURE"
        FAILED_ATTEMPT_ARCHIVE = "tf_step9_local_residual_slot_decomposition_kaggle_t4.zip"
        FAILED_ATTEMPT_ARCHIVE_SHA256 = "ff19925fc4ad6f6d8144512979dd2f725355cacc31303a848bd77037d4a41b17"
        FAILED_ATTEMPT_STEP9_TOOL_SHA256 = "a35893cc90c4179d31c101f7db026c4c41eaf2509e9c3b0e19a0c53bc8887645"
        FAILED_ATTEMPT_ERROR = "LocalResidualSlotProbeError: Frozen execution contract drift"

        EXPECTED_ARTIFACTS = {
            "checkpoint": {"basename": "best_val_accuracy.keras", "sha256": "9ec11bb819f97e4fbda432f68da76c1201b8a3f9e06fae9eb30489a528d6ac16"},
            "checkpoint_metadata": {"basename": "best_val_accuracy.metadata.json", "sha256": "e62cf8c86f0d6a56c3041911de6397d18f47276f79f03c6a50ca71fa47300a37"},
            "resolved_config": {"basename": "resolved_config.json", "sha256": "3c028dd2f32ebed3a252544e170220b150b5e29920cea865924dddce6aef5a32"},
        }

        EXPECTED_CHECKPOINT_EPOCH = 31
        EXPECTED_SEED = 42
        EXPECTED_VALIDATION_SAMPLES = 3589
        CONDITION_S0 = "official_manual_forward"
        CONDITION_S1 = "mouth_local_residual_zero"
        CONDITION_S2 = "eye_local_residual_zero"
        CONDITION_S3 = "brow_local_residual_zero"
        CONDITION_S4 = "nose_cheek_local_residual_zero"
        CONDITION_S5 = "all_local_residuals_zero_anchor"
        CONDITIONS = (CONDITION_S0, CONDITION_S1, CONDITION_S2, CONDITION_S3, CONDITION_S4, CONDITION_S5)
        S0_REFERENCE = {"accuracy": 0.63137364168292, "macro_f1": 0.5932591901893336, "loss": 1.1537981840361535}
        S5_REFERENCE = {"accuracy": 0.22596823627751464, "macro_f1": 0.1958426679087715, "loss": 1.883221954371022}
        REFERENCE_TOLERANCE = {"accuracy": 0.001, "macro_f1": 0.001, "loss": 0.005}
        NATIVE_MANUAL_TOLERANCE = {"prediction_agreement": 1.0, "max_abs_logit_difference": 1e-5, "max_abs_probability_difference": 3e-6}
        SLOT_SENSITIVITY_THRESHOLDS_PP = {"high_min": 10.0, "moderate_min": 5.0}
        SLOT_SENSITIVITY_LABELS = ("HIGH_SLOT_SENSITIVITY", "MODERATE_SLOT_SENSITIVITY", "LOW_SLOT_SENSITIVITY")
        OVERALL_DECISIONS = ("SINGLE_HIGH_LOCAL_SLOT", "MULTIPLE_HIGH_LOCAL_SLOTS", "NO_SINGLE_HIGH_LOCAL_SLOT_WITH_JOINT_DEPENDENCY")

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
        STEP9_TOOL_PATH = PROJECT_PATH / STEP9_TOOL_RELATIVE
        STEP7_TOOL_PATH = PROJECT_PATH / STEP7_TOOL_RELATIVE
        STEP6_SUPPORT_PATH = PROJECT_PATH / STEP6_SUPPORT_RELATIVE
        RUN_ROOT = WORKING / "tf_step9_local_residual_slot_decomposition"
        PROBE_OUTPUT_ROOT = RUN_ROOT / "probe"
        ADAPTER_METADATA_ROOT = RUN_ROOT / "adapter_metadata"
        REPORT_PATH = WORKING / "tf_step9_local_residual_slot_decomposition.md"
        ARCHIVE_PATH = WORKING / "tf_step9_local_residual_slot_decomposition_kaggle_t4.zip"
        SUBPROCESS_LOG_PATH = RUN_ROOT / "step9_subprocess.log"

        EVAL_BATCH_SIZE = 32
        GRAPH_WORKERS = 2
        GRAPH_CACHE_SIZE = 64
        TESTED_TENSORFLOW = "2.18.1"
        TESTED_KERAS = "3.15.0"
        """,
    )

    _replace_required(
        cells[4],
        "    STEP9_TOOL_PATH,\n    STEP6_SUPPORT_PATH,\n",
        "    STEP9_TOOL_PATH,\n    STEP7_TOOL_PATH,\n    STEP6_SUPPORT_PATH,\n",
    )
    _replace_required(
        cells[4],
        'if sha256(STEP9_TOOL_PATH) != EXPECTED_STEP9_TOOL_SHA256:\n    raise RuntimeError("Reviewed Step 9 probe tool drift")\n',
        'if sha256(STEP9_TOOL_PATH) != EXPECTED_STEP9_TOOL_SHA256:\n    raise RuntimeError("Reviewed Step 9 harness drift")\n'
        'if sha256(STEP7_TOOL_PATH) != EXPECTED_STEP7_TOOL_SHA256:\n    raise RuntimeError("Reviewed Step 7 harness drift")\n',
    )
    _replace_required(
        cells[4],
        '    "step9_harness_sha256": sha256(STEP9_TOOL_PATH),\n    "step6_support_sha256": sha256(STEP6_SUPPORT_PATH),\n',
        '    "step9_harness_sha256": sha256(STEP9_TOOL_PATH),\n'
        '    "step7_harness_sha256": sha256(STEP7_TOOL_PATH),\n'
        '    "step6_support_sha256": sha256(STEP6_SUPPORT_PATH),\n',
    )
    _replace_required(cells[6], "Issue #23 requires Kaggle GPU T4", "Issue #23 requires Kaggle GPU T4")

    _replace_required(
        cells[10],
        '    "step9_harness_sha256": sha256(STEP9_TOOL_PATH),\n    "step6_support_sha256": sha256(STEP6_SUPPORT_PATH),\n',
        '    "step9_harness_sha256": sha256(STEP9_TOOL_PATH),\n'
        '    "step7_harness_sha256": sha256(STEP7_TOOL_PATH),\n'
        '    "step6_support_sha256": sha256(STEP6_SUPPORT_PATH),\n',
    )
    _replace_required(
        cells[10],
        '    "scientific_payload_sha256": package_manifest["scientific_payload_sha256"],\n',
        '    "scientific_payload_sha256": package_manifest["scientific_payload_sha256"],\n'
        '    "execution_contract_sha256": package_manifest["execution_contract_sha256"],\n',
    )
    _replace_required(
        cells[10],
        '    "gate_a_native_manual_tolerance": dict(NATIVE_MANUAL_TOLERANCE),\n',
        '    "gate_a_native_manual_tolerance": dict(NATIVE_MANUAL_TOLERANCE),\n'
        '    "slot_sensitivity_thresholds_pp": dict(SLOT_SENSITIVITY_THRESHOLDS_PP),\n'
        '    "overall_decisions": list(OVERALL_DECISIONS),\n',
    )
    _replace_required(
        cells[10],
        '    "registered_full_run": True,\n',
        '    "previous_registered_attempt": {\n'
        '        "classification": FAILED_ATTEMPT_CLASSIFICATION,\n'
        '        "archive": FAILED_ATTEMPT_ARCHIVE,\n'
        '        "archive_sha256": FAILED_ATTEMPT_ARCHIVE_SHA256,\n'
        '        "step9_harness_sha256": FAILED_ATTEMPT_STEP9_TOOL_SHA256,\n'
        '        "failure": FAILED_ATTEMPT_ERROR,\n'
        '        "scientific_result_valid": False,\n'
        '        "scientific_interpretation": None,\n'
        '        "s0_s5_scientific_outcome": None,\n'
        '    },\n'
        '    "registered_full_run": True,\n',
    )
    forensic_block = '''    "reviewed_gate_a_forensic": {
        "archive_sha256": REVIEWED_GATE_A_FORENSIC_ARCHIVE_SHA256,
        "batch_count": REVIEWED_GATE_A_FORENSIC_BATCHES,
        "sample_count": REVIEWED_GATE_A_FORENSIC_SAMPLES,
        "scientific_decomposition_run": False,
    },
'''
    _replace_required(cells[10], forensic_block, "")

    _replace_required(
        cells[12],
        '        f"- Step-9 SHA-256: `{EXPECTED_STEP9_TOOL_SHA256}`.",\n        f"- Step-6 SHA-256: `{EXPECTED_STEP6_SUPPORT_SHA256}`.",\n',
        '        f"- Step-9 SHA-256: `{EXPECTED_STEP9_TOOL_SHA256}`.",\n'
        '        f"- Step-7 SHA-256: `{EXPECTED_STEP7_TOOL_SHA256}`.",\n'
        '        f"- Step-6 SHA-256: `{EXPECTED_STEP6_SUPPORT_SHA256}`.",\n',
    )
    _replace_required(
        cells[12],
        '        f"- Frozen scientific payload: `{EXPECTED_SCIENTIFIC_PAYLOAD_SHA256}`.",\n',
        '        f"- Frozen scientific payload: `{EXPECTED_SCIENTIFIC_PAYLOAD_SHA256}`.",\n'
        '        f"- Previous run: `{FAILED_ATTEMPT_CLASSIFICATION}`; archive SHA-256 `{FAILED_ATTEMPT_ARCHIVE_SHA256}`; no S0-S5 scientific outcome.",\n',
    )
    required_integrity_old = '''        "source_batches_mutated": False,
        "message_passing_inputs_changed": False,
        "node_edge_features_changed": False,
        "topology_changed": False,
        "paired_original_batch_evaluation": True,
        "checkpoint_unchanged": True,
        "model_weights_unchanged": True,
        "training_or_optimizer_state_created": False,
        "test_split_constructed": False,
'''
    required_integrity_new = '''        "source_batches_mutated": False,
        "message_passing_inputs_changed": False,
        "node_edge_features_changed": False,
        "coordinates_changed": False,
        "topology_changed": False,
        "labels_or_sample_ids_changed": False,
        "global_embedding_changed": False,
        "validity_flags_changed": False,
        "readout_part_soft_changed": False,
        "context_or_upstream_state_changed": False,
        "paired_original_batch_evaluation": True,
        "checkpoint_unchanged": True,
        "model_weights_unchanged": True,
        "training_or_optimizer_state_created": False,
        "test_split_constructed": False,
'''
    _replace_required(cells[14], required_integrity_old, required_integrity_new)
    gate_b_count_block = '''    if (
        gate_b.get("sample_count") != EXPECTED_VALIDATION_SAMPLES
        or gate_b.get("required_sample_count") != EXPECTED_VALIDATION_SAMPLES
        or gate_b.get("sample_count_exact") is not True
    ):
        raise RuntimeError("Gate B sample-count evidence drift")
'''
    both_gate_count_block = '''    for gate_name, gate in (("Gate B", gate_b), ("Gate C", gate_c)):
        if (
            gate.get("sample_count") != EXPECTED_VALIDATION_SAMPLES
            or gate.get("required_sample_count") != EXPECTED_VALIDATION_SAMPLES
            or gate.get("sample_count_exact") is not True
        ):
            raise RuntimeError(f"{gate_name} sample-count evidence drift")
'''
    _replace_required(cells[14], gate_b_count_block, both_gate_count_block)
    _replace_required(
        cells[14],
        '    if gates.get("per_slot_diagnostics") is None or gates.get("overall_decision") is None:\n        raise RuntimeError("Registered diagnostics missing after passed gates")\n',
        '    per_slot_diagnostics = gates.get("per_slot_diagnostics")\n'
        '    overall_decision = gates.get("overall_decision")\n'
        '    if not isinstance(per_slot_diagnostics, dict) or set(per_slot_diagnostics) != set(CONDITIONS[1:5]):\n'
        '        raise RuntimeError("Registered per-slot diagnostic inventory drift")\n'
        '    if any(item.get("label") not in SLOT_SENSITIVITY_LABELS for item in per_slot_diagnostics.values()):\n'
        '        raise RuntimeError("Registered per-slot label drift")\n'
        '    if overall_decision not in OVERALL_DECISIONS:\n'
        '        raise RuntimeError("Registered overall-decision identity drift")\n',
    )
    _replace_required(
        cells[14],
        '        "step9_harness_sha256": sha256(STEP9_TOOL_PATH),\n        "step6_support_sha256": sha256(STEP6_SUPPORT_PATH),\n',
        '        "step9_harness_sha256": sha256(STEP9_TOOL_PATH),\n'
        '        "step7_harness_sha256": sha256(STEP7_TOOL_PATH),\n'
        '        "step6_support_sha256": sha256(STEP6_SUPPORT_PATH),\n',
    )
    _replace_required(
        cells[16],
        '        f"- Step 9 harness SHA-256: `{sha256(STEP9_TOOL_PATH)}`.",\n        f"- Step 6 support-tool SHA-256: `{sha256(STEP6_SUPPORT_PATH)}`.",\n',
        '        f"- Step 9 harness SHA-256: `{sha256(STEP9_TOOL_PATH)}`.",\n'
        '        f"- Step 7 harness SHA-256: `{sha256(STEP7_TOOL_PATH)}`.",\n'
        '        f"- Step 6 support-tool SHA-256: `{sha256(STEP6_SUPPORT_PATH)}`.",\n',
    )
    _replace_required(
        cells[16],
        '        f"- Frozen scientific payload SHA-256: `{package_manifest[\'scientific_payload_sha256\']}`.",\n',
        '        f"- Frozen scientific payload SHA-256: `{package_manifest[\'scientific_payload_sha256\']}`.",\n'
        '        f"- Previous run: `{FAILED_ATTEMPT_CLASSIFICATION}`; archive SHA-256 `{FAILED_ATTEMPT_ARCHIVE_SHA256}`; no S0-S5 scientific outcome.",\n',
    )
    _replace_required(
        cells[16],
        "S1-S4 effects are nonlinear functional sensitivity diagnostics under the fixed official MediaPipe-derived scaffold. They must not be summed or treated as additive causal contributions.",
        "S1-S4 learned pooled local residual-slot effects are nonlinear functional sensitivity diagnostics under the fixed official MediaPipe-derived scaffold. The harness diagnostics are reported verbatim; effects must not be summed, divided by S5, or treated as additive causal percentage contributions.",
    )

    for index, cell in enumerate(cells):
        cell["id"] = f"issue23-{index:02d}"
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
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
