import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    ROOT / "notebooks" / "kaggle-issue23-local-residual-slot-decomposition.ipynb"
)
BUILDER_PATH = ROOT / "tools" / "build_issue23_kaggle_adapter.py"
PACKAGE_ROOT = ROOT / "standalone" / "lap_gnn_tensorflow_ofix7_mid_candidate"
STEP9_TOOL = (
    PACKAGE_ROOT
    / "tools"
    / "evaluate_fixed_checkpoint_local_residual_slot_decomposition_probe.py"
)
STEP7_TOOL = (
    PACKAGE_ROOT
    / "tools"
    / "evaluate_fixed_checkpoint_direct_part_decomposition_probe.py"
)
STEP6_TOOL = PACKAGE_ROOT / "tools" / "evaluate_fixed_checkpoint_prior_probe.py"

EXPECTED_SCIENTIFIC_BASE = "753ae1a27b9e4467d11c5d68cb416df63de29ff5"
EXPECTED_EXECUTION = "73a5bd6fe1210b379287ca9e0048526ff682e7a9"
EXPECTED_STEP9_SHA = "50a310f622cdf9dccf13eff4edf6394f1d39b8ccf315dce5ede07d0a45bdd77a"
FAILED_STEP9_SHA = "a35893cc90c4179d31c101f7db026c4c41eaf2509e9c3b0e19a0c53bc8887645"
FAILED_ARCHIVE_SHA = "ff19925fc4ad6f6d8144512979dd2f725355cacc31303a848bd77037d4a41b17"
EXPECTED_STEP7_SHA = "c0b1df778e469665dd6437c58831d29dcc34fbde44231db75894c5469a1ade78"
EXPECTED_STEP6_SHA = "3a033c977a29e102cfed75282ae7c1062f41feac8bef1b955ae425ec7e4004b3"
EXPECTED_PAYLOAD_SHA = "286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e"
EXPECTED_CONTRACT_SHA = "14acc2750875a25922007459161a137158d8040805e616166be923f63658bf22"
EXPECTED_ARTIFACT_HASHES = {
    "9ec11bb819f97e4fbda432f68da76c1201b8a3f9e06fae9eb30489a528d6ac16",
    "e62cf8c86f0d6a56c3041911de6397d18f47276f79f03c6a50ca71fa47300a37",
    "3c028dd2f32ebed3a252544e170220b150b5e29920cea865924dddce6aef5a32",
}
CONDITIONS = (
    "official_manual_forward",
    "mouth_local_residual_zero",
    "eye_local_residual_zero",
    "brow_local_residual_zero",
    "nose_cheek_local_residual_zero",
    "all_local_residuals_zero_anchor",
)


def _notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _code_cells():
    return [
        "".join(cell.get("source", []))
        for cell in _notebook()["cells"]
        if cell.get("cell_type") == "code"
    ]


def _all_source():
    return "\n".join(
        "".join(cell.get("source", [])) for cell in _notebook()["cells"]
    )


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_notebook_is_deterministic_unexecuted_and_code_compiles():
    spec = importlib.util.spec_from_file_location("issue23_builder", BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert _notebook() == module.build_notebook()

    notebook = _notebook()
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 19
    assert len(_code_cells()) == 9
    for index, cell in enumerate(notebook["cells"]):
        assert cell["id"] == f"issue23-{index:02d}"
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            compile("".join(cell["source"]), f"cell-{index}", "exec")


def test_exact_source_payload_contract_tools_and_artifact_locks():
    source = _all_source()
    for value in (
        EXPECTED_SCIENTIFIC_BASE,
        EXPECTED_EXECUTION,
        EXPECTED_STEP9_SHA,
        EXPECTED_STEP7_SHA,
        EXPECTED_STEP6_SHA,
        EXPECTED_PAYLOAD_SHA,
        EXPECTED_CONTRACT_SHA,
        *EXPECTED_ARTIFACT_HASHES,
    ):
        assert value in source
    assert _sha256(STEP9_TOOL) == EXPECTED_STEP9_SHA
    assert _sha256(STEP7_TOOL) == EXPECTED_STEP7_SHA
    assert _sha256(STEP6_TOOL) == EXPECTED_STEP6_SHA
    manifest = json.loads((PACKAGE_ROOT / "package_manifest.json").read_text())
    assert manifest["scientific_payload_sha256"] == EXPECTED_PAYLOAD_SHA
    assert manifest["execution_contract_sha256"] == EXPECTED_CONTRACT_SHA
    assert subprocess.run(
        ["git", "cat-file", "-e", f"{EXPECTED_EXECUTION}^{{commit}}"], cwd=ROOT
    ).returncode == 0
    assert subprocess.run(
        ["git", "cat-file", "-e", f"{EXPECTED_SCIENTIFIC_BASE}^{{commit}}"],
        cwd=ROOT,
    ).returncode == 0
    assert (
        f'EXPECTED_SCIENTIFIC_BASE_COMMIT = "{EXPECTED_SCIENTIFIC_BASE}"'
        in source
    )
    assert f'EXPECTED_EXECUTION_COMMIT = "{EXPECTED_EXECUTION}"' in source
    assert '"git", "clone", "--no-checkout"' in source
    assert '"git", "checkout", "--detach", EXPECTED_EXECUTION_COMMIT' in source
    assert "actual_commit != EXPECTED_EXECUTION_COMMIT or dirty" in source
    assert 'TF_PACKAGE_PATH / "tools/verify_checksums.py"' in source
    for check in (
        "sha256(STEP9_TOOL_PATH) != EXPECTED_STEP9_TOOL_SHA256",
        "sha256(STEP7_TOOL_PATH) != EXPECTED_STEP7_TOOL_SHA256",
        "sha256(STEP6_SUPPORT_PATH) != EXPECTED_STEP6_SUPPORT_SHA256",
    ):
        assert check in source
    assert EXPECTED_STEP9_SHA != FAILED_STEP9_SHA
    assert f'EXPECTED_STEP9_TOOL_SHA256 = "{EXPECTED_STEP9_SHA}"' in source
    assert f'FAILED_ATTEMPT_STEP9_TOOL_SHA256 = "{FAILED_STEP9_SHA}"' in source


def test_first_authorized_failure_is_preserved_without_scientific_outcome():
    source = _all_source()
    for token in (
        "PRE-INTERVENTION TECHNICAL HARNESS FAILURE",
        "tf_step9_local_residual_slot_decomposition_kaggle_t4.zip",
        FAILED_ARCHIVE_SHA,
        "LocalResidualSlotProbeError: Frozen execution contract drift",
        '"scientific_result_valid": False',
        '"scientific_interpretation": None',
        '"s0_s5_scientific_outcome": None',
    ):
        assert token in source


def test_sha_locator_uses_identity_rejects_zero_or_ambiguous_and_excludes_samples(
    tmp_path,
):
    artifact_cell = _code_cells()[3]
    tree = ast.parse(artifact_cell)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = {"os": __import__("os"), "Path": Path, "sha256": _sha256}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "locator", "exec"), namespace)

    input_root = tmp_path / "input"
    public_root = input_root / "public-samples"
    artifact_root = input_root / "issue7-artifacts"
    public_root.mkdir(parents=True)
    artifact_root.mkdir(parents=True)
    basename = "best_val_accuracy.keras"
    content = b"registered-epoch-31-checkpoint"
    (public_root / basename).write_bytes(content)
    expected_path = artifact_root / basename
    expected_path.write_bytes(content)
    namespace["KAGGLE_INPUT_ROOT"] = input_root
    namespace["PUBLIC_SAMPLE_INPUT_ROOTS"] = (public_root,)
    artifact_spec = {"basename": basename, "sha256": _sha256(expected_path)}
    assert namespace["locate_unique_sha_artifact"](artifact_spec) == expected_path.resolve()

    expected_path.unlink()
    with pytest.raises(RuntimeError, match="Expected exactly one SHA-matched"):
        namespace["locate_unique_sha_artifact"](artifact_spec)
    expected_path.write_bytes(content)
    duplicate_root = input_root / "duplicate-artifacts"
    duplicate_root.mkdir()
    (duplicate_root / basename).write_bytes(content)
    with pytest.raises(RuntimeError, match="Expected exactly one SHA-matched"):
        namespace["locate_unique_sha_artifact"](artifact_spec)


def test_preflight_reads_validation_paths_and_disclosed_shared_metadata_only():
    pre_run_source = "\n".join(_code_cells()[:5])
    assert 'FER_VAL_CSV = FER_SPLIT_ROOT / "val.csv"' in pre_run_source
    assert 'PRIOR_VAL_DIR = PRIOR_ROOT / "val"' in pre_run_source
    assert 'GRAPH_CACHE_VAL_DIR = GRAPH_CACHE_ROOT / "val"' in pre_run_source
    assert '"sample_split": "val"' in pre_run_source
    assert "CACHE_COMPLETE_PATH" in pre_run_source
    assert "non-sample aggregate metadata" in pre_run_source
    for forbidden in (
        'FER_SPLIT_ROOT / "train.csv"',
        'FER_SPLIT_ROOT / "test.csv"',
        'PRIOR_ROOT / "train"',
        'PRIOR_ROOT / "test"',
        'GRAPH_CACHE_ROOT / "train"',
        'GRAPH_CACHE_ROOT / "test"',
        'split="test"',
        "split='test'",
    ):
        assert forbidden not in pre_run_source


def test_registered_command_invokes_step9_exactly_once_and_is_unbounded():
    run_cell = _code_cells()[5]
    tree = ast.parse(run_cell)
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "probe_command"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1
    command = assignments[0].value
    assert isinstance(command, ast.List)
    strings = [
        item.value
        for item in command.elts
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    ]
    for argument in (
        "--checkpoint",
        "--checkpoint-metadata",
        "--resolved-config",
        "--prior-root",
        "--clean-graph-cache-dir",
        "--output-root",
        "--eval-batch-size",
        "--graph-workers",
        "--graph-cache-size",
    ):
        assert argument in strings
    assert not any(argument.startswith("--limit-") for argument in strings)
    assert run_cell.count("run_probe_with_failure_archive(probe_command") == 1
    assert "run_checked(probe_command" not in run_cell
    assert "probe_command.count(STEP9_TOOL_PATH) != 1" in run_cell
    assert "STEP7_TOOL_PATH" not in run_cell
    assert "STEP6_SUPPORT_PATH" not in run_cell
    constants = _code_cells()[0]
    assert "EVAL_BATCH_SIZE = 32" in constants
    assert "GRAPH_WORKERS = 2" in constants
    assert "GRAPH_CACHE_SIZE = 64" in constants


def test_exit7_returns_normally_and_publishes_failure_archive(tmp_path):
    run_cell = _code_cells()[5]
    tree = ast.parse(run_cell)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    working = tmp_path / "working"
    run_root = working / "tf_step9_local_residual_slot_decomposition"
    probe_root = run_root / "probe"
    adapter_root = run_root / "adapter_metadata"
    probe_root.mkdir(parents=True)
    adapter_root.mkdir(parents=True)
    partial_probe = probe_root / "partial_probe_evidence.json"
    partial_probe.write_text(
        '{"status": "PARTIAL", "scientific_result_valid": false}\n',
        encoding="utf-8",
    )
    pre_run = adapter_root / "pre_run_manifest.json"
    pre_run.write_text('{"issue": 23}\n', encoding="utf-8")
    located = {}
    for name in ("checkpoint", "checkpoint_metadata", "resolved_config"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        located[name] = path
    artifact_hashes = {name: _sha256(path) for name, path in located.items()}
    namespace = {
        "Path": Path,
        "zipfile": zipfile,
        "subprocess": subprocess,
        "json": json,
        "WORKING": working,
        "RUN_ROOT": run_root,
        "PROBE_OUTPUT_ROOT": probe_root,
        "ADAPTER_METADATA_ROOT": adapter_root,
        "SUBPROCESS_LOG_PATH": run_root / "step9_subprocess.log",
        "REPORT_PATH": working / "tf_step9_local_residual_slot_decomposition.md",
        "ARCHIVE_PATH": working
        / "tf_step9_local_residual_slot_decomposition_kaggle_t4.zip",
        "located_artifacts": located,
        "artifact_hashes_before": artifact_hashes,
        "sha256": _sha256,
        "EXPECTED_SCIENTIFIC_BASE_COMMIT": EXPECTED_SCIENTIFIC_BASE,
        "EXPECTED_EXECUTION_COMMIT": EXPECTED_EXECUTION,
        "EXPECTED_STEP9_TOOL_SHA256": EXPECTED_STEP9_SHA,
        "EXPECTED_STEP7_TOOL_SHA256": EXPECTED_STEP7_SHA,
        "EXPECTED_STEP6_SUPPORT_SHA256": EXPECTED_STEP6_SHA,
        "EXPECTED_SCIENTIFIC_PAYLOAD_SHA256": EXPECTED_PAYLOAD_SHA,
        "FAILED_ATTEMPT_CLASSIFICATION": (
            "PRE-INTERVENTION TECHNICAL HARNESS FAILURE"
        ),
        "FAILED_ATTEMPT_ARCHIVE_SHA256": FAILED_ARCHIVE_SHA,
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), "wrapper", "exec"),
        namespace,
    )
    outcome = namespace["run_probe_with_failure_archive"](
        [
            sys.executable,
            "-c",
            "import sys; print('synthetic Step-9 exit-7 evidence'); sys.exit(7)",
        ],
        tmp_path,
    )
    assert outcome["result"].returncode == 7
    wrapper = outcome["wrapper_execution"]
    assert wrapper["status"] == "TECHNICAL_OR_GATE_FAILURE"
    assert wrapper["subprocess_return_code"] == 7
    assert wrapper["artifact_hashes_before"] == artifact_hashes
    assert wrapper["artifact_hashes_after"] == artifact_hashes
    assert wrapper["artifacts_unchanged"] is True
    assert wrapper["scientific_result_valid"] is False
    assert wrapper["scientific_interpretation"] is None
    assert wrapper["training"] is False
    assert wrapper["test_access"] is False

    namespace.update(
        {
            "wrapper_execution": wrapper,
            "wrapper_status": wrapper["status"],
            "initial_archive_names": outcome["archive_names"],
        }
    )
    for cell_index in (6, 7, 8):
        exec(
            compile(_code_cells()[cell_index], f"failure-cell-{cell_index}", "exec"),
            namespace,
        )
    assert namespace["wrapper_status"] == "TECHNICAL_OR_GATE_FAILURE"
    archive_path = namespace["ARCHIVE_PATH"]
    assert archive_path.is_file()
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        assert any(name.endswith("step9_subprocess.log") for name in names)
        assert any(name.endswith("adapter_metadata/pre_run_manifest.json") for name in names)
        assert any(name.endswith("adapter_metadata/wrapper_execution.json") for name in names)
        assert any(name.endswith("adapter_metadata/failure_status.json") for name in names)
        assert any(name.endswith("probe/partial_probe_evidence.json") for name in names)
        assert "tf_step9_local_residual_slot_decomposition.md" in names
        assert not any(name.endswith("final_evidence.json") for name in names)
        assert not any(name.endswith(".keras") for name in names)
        log_name = "tf_step9_local_residual_slot_decomposition/step9_subprocess.log"
        assert "synthetic Step-9 exit-7 evidence" in archive.read(log_name).decode()
    assert not (adapter_root / "final_evidence.json").exists()
    report = namespace["REPORT_PATH"].read_text(encoding="utf-8")
    assert "TECHNICAL_OR_GATE_FAILURE" in report
    assert "Scientific result valid: `false`" in report
    assert "Scientific interpretation: `null`" in report


def test_failure_and_success_branches_remain_explicit_and_fail_scientifically_closed():
    run_cell, verify_cell, report_cell, archive_cell = _code_cells()[5:9]
    assert '"status": "TECHNICAL_OR_GATE_FAILURE"' in run_cell
    assert '"scientific_result_valid": False' in run_cell
    assert '"scientific_interpretation": None' in run_cell
    assert 'wrapper_status == "SUBPROCESS_COMPLETE_PENDING_VERIFICATION"' in verify_cell
    assert 'wrapper_status == "TECHNICAL_OR_GATE_FAILURE"' in verify_cell
    assert "verify_complete_probe_evidence()" in verify_cell
    assert "record_failure_and_archive(" in verify_cell
    assert 'wrapper_status == "COMPLETE"' in report_cell
    assert "verify_failure_archive(initial_archive_names)" in report_cell
    assert 'wrapper_status == "COMPLETE"' in archive_cell
    assert "verify_failure_archive(archived_names)" in archive_cell


def test_t4_versions_fresh_outputs_conditions_and_registered_gates_are_locked():
    source = _all_source()
    assert 'all("T4" in name for name in gpu_names)' in source
    assert 'TESTED_TENSORFLOW = "2.18.1"' in source
    assert 'TESTED_KERAS = "3.15.0"' in source
    assert "Registered output must be fresh" in source
    assert "/kaggle/working/tf_step9_local_residual_slot_decomposition.md" in source
    assert (
        "/kaggle/working/tf_step9_local_residual_slot_decomposition_kaggle_t4.zip"
        in source
    )
    for condition in CONDITIONS:
        assert condition in source
    assert (
        'NATIVE_MANUAL_TOLERANCE = {"prediction_agreement": 1.0, '
        '"max_abs_logit_difference": 1e-5, '
        '"max_abs_probability_difference": 3e-6}'
    ) in source
    for reference in (
        "0.63137364168292",
        "0.5932591901893336",
        "1.1537981840361535",
        "0.22596823627751464",
        "0.1958426679087715",
        "1.883221954371022",
    ):
        assert reference in source
    for token in (
        "gate_a_native_manual_s0_equivalence",
        "gate_b_s0_reference",
        "gate_c_s5_d3_anchor",
        "VALID_REGISTERED_SLOT_DECOMPOSITION",
        "per_slot_diagnostics",
        "SINGLE_HIGH_LOCAL_SLOT",
        "MULTIPLE_HIGH_LOCAL_SLOTS",
        "NO_SINGLE_HIGH_LOCAL_SLOT_WITH_JOINT_DEPENDENCY",
    ):
        assert token in source
    post_run = _code_cells()[6]
    assert 'verify_reference_gate(gate_b, S0_REFERENCE, "Gate B")' in post_run
    assert 'verify_reference_gate(gate_c, S5_REFERENCE, "Gate C")' in post_run
    assert "if delta" not in post_run
    assert "delta >=" not in post_run


def test_success_verification_requires_inventory_integrity_immutability_and_isolation():
    source = _all_source()
    for token in (
        "actual_output_names != expected_output_names",
        "paired_sample_count != EXPECTED_VALIDATION_SAMPLES",
        '"limit_val_batches": None',
        '"bounded_smoke_only": False',
        '"condition_order": list(CONDITIONS)',
        '"source_batches_mutated": False',
        '"message_passing_inputs_changed": False',
        '"node_edge_features_changed": False',
        '"coordinates_changed": False',
        '"topology_changed": False',
        '"labels_or_sample_ids_changed": False',
        '"global_embedding_changed": False',
        '"validity_flags_changed": False',
        '"readout_part_soft_changed": False',
        '"context_or_upstream_state_changed": False',
        '"checkpoint_unchanged": True',
        '"model_weights_unchanged": True',
        '"training_or_optimizer_state_created": False',
        '"test_split_constructed": False',
    ):
        assert token in source
    assert 'name.endswith(".keras")' in source
    assert 'Path(name).name in {"train.csv", "test.csv"}' in source
    assert 'ADAPTER_METADATA_ROOT / "final_evidence.json"' in source


def test_no_training_test_graph_rebuild_selector_or_op_determinism_path():
    source = _all_source()
    for forbidden in (
        "lap_gnn_tf.cli.train",
        "train_validation_only.py",
        "model.fit(",
        "apply_gradients(",
        "GradientTape",
        "PixelPriorDataset._zero_prior",
        "shuffle_prior",
        "forced_fallback",
        "attenuate_prior",
        "manual_forward(",
        "GraphBatchGenerator(",
        "build_graph(",
        "enable_op_determinism",
        "--condition",
        "--slot",
        "--pair",
    ):
        assert forbidden not in source
    assert '"training": False' in source
    assert '"test_access": False' in source


def test_documented_inputs_internet_zip_and_pre_run_stop():
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in _notebook()["cells"]
        if cell.get("cell_type") == "markdown"
    )
    for path in (
        "/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split",
        "/kaggle/input/datasets/irthn1311/d16-mediapipe-pixel-priors-best-retry-rescue/outputs/d16_mediapipe_pixel_priors_best_retry_rescue",
        "/kaggle/input/datasets/irthn1311/ofix7-mid-seed42-records",
        "/kaggle/working/tf_step9_local_residual_slot_decomposition.md",
        "/kaggle/working/tf_step9_local_residual_slot_decomposition_kaggle_t4.zip",
    ):
        assert path in markdown
    assert "Internet is required only" in markdown
    assert "attach one separate read-only Kaggle Input" in markdown
    assert "Pre-run review gate" in markdown
    assert "intentionally unexecuted" in markdown
