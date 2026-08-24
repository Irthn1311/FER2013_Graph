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
NOTEBOOK_PATH = ROOT / "notebooks" / "kaggle-issue15-direct-part-decomposition.ipynb"
BUILDER_PATH = ROOT / "tools" / "build_issue15_kaggle_adapter.py"
PACKAGE_ROOT = ROOT / "standalone" / "lap_gnn_tensorflow_ofix7_mid_candidate"
PROBE_TOOL = PACKAGE_ROOT / "tools" / "evaluate_fixed_checkpoint_direct_part_decomposition_probe.py"
SUPPORT_TOOL = PACKAGE_ROOT / "tools" / "evaluate_fixed_checkpoint_prior_probe.py"

EXPECTED_BASE = "d14e7e1e3eec2ffbd5339b5a3bd0d5db5ab3de8b"
EXPECTED_EXECUTION = "27c366a955648764386fe48e489a6a1e94a479a1"
EXPECTED_PROBE_SHA = "c0b1df778e469665dd6437c58831d29dcc34fbde44231db75894c5469a1ade78"
EXPECTED_SUPPORT_SHA = "3a033c977a29e102cfed75282ae7c1062f41feac8bef1b955ae425ec7e4004b3"
EXPECTED_PAYLOAD_SHA = "286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e"
EXPECTED_FORENSIC_ARCHIVE_SHA = "bf693500078f170ceea094fad319f513c2a64d8610fa41c65ebac088ec954c8d"
EXPECTED_ARTIFACT_HASHES = {
    "9ec11bb819f97e4fbda432f68da76c1201b8a3f9e06fae9eb30489a528d6ac16",
    "e62cf8c86f0d6a56c3041911de6397d18f47276f79f03c6a50ca71fa47300a37",
    "3c028dd2f32ebed3a252544e170220b150b5e29920cea865924dddce6aef5a32",
}
CONDITIONS = (
    "official_manual_forward",
    "context_local_prior_neutralized",
    "readout_local_prior_neutralized",
    "local_part_residual_zero",
    "local_motif_validity_off",
    "full_direct_part_zero_anchor",
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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_notebook_is_deterministic_unexecuted_and_code_compiles():
    spec = importlib.util.spec_from_file_location("issue15_builder", BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert _notebook() == module.build_notebook()

    notebook = _notebook()
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 19
    for index, cell in enumerate(notebook["cells"]):
        assert cell["id"] == f"issue15-{index:02d}"
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            compile("".join(cell["source"]), f"cell-{index}", "exec")


def test_exact_base_payload_tool_and_artifact_locks_match_repository():
    source = _all_source()
    for value in (
        EXPECTED_BASE,
        EXPECTED_EXECUTION,
        EXPECTED_PROBE_SHA,
        EXPECTED_SUPPORT_SHA,
        EXPECTED_PAYLOAD_SHA,
    ):
        assert value in source
    for expected in EXPECTED_ARTIFACT_HASHES:
        assert expected in source
    assert _sha256(PROBE_TOOL) == EXPECTED_PROBE_SHA
    assert _sha256(SUPPORT_TOOL) == EXPECTED_SUPPORT_SHA
    manifest = json.loads((PACKAGE_ROOT / "package_manifest.json").read_text())
    assert manifest["scientific_payload_sha256"] == EXPECTED_PAYLOAD_SHA
    assert EXPECTED_EXECUTION != EXPECTED_BASE
    assert subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTED_BASE, EXPECTED_EXECUTION],
        cwd=ROOT,
    ).returncode == 0
    assert f'EXPECTED_EXECUTION_COMMIT = "{EXPECTED_EXECUTION}"' in source
    assert '"git", "checkout", "--detach", EXPECTED_EXECUTION_COMMIT' in source
    assert "actual_commit != EXPECTED_EXECUTION_COMMIT or dirty" in source
    assert "sha256(PROBE_TOOL_PATH) != EXPECTED_PROBE_TOOL_SHA256" in source
    assert "sha256(SUPPORT_TOOL_PATH) != EXPECTED_SUPPORT_TOOL_SHA256" in source
    assert 'TF_PACKAGE_PATH / "tools/verify_checksums.py"' in source


def test_sha_only_artifact_discovery_is_unique_and_excludes_sample_inputs():
    source = _all_source()
    for basename in (
        "best_val_accuracy.keras",
        "best_val_accuracy.metadata.json",
        "resolved_config.json",
    ):
        assert basename in source
    assert "def locate_unique_sha_artifact(spec):" in source
    assert 'actual_sha == spec["sha256"]' in source
    assert "if len(matches) != 1:" in source
    assert "PUBLIC_SAMPLE_INPUT_ROOTS" in source
    assert "Artifact candidate overlaps a sample input" in source
    assert "Artifact is not a read-only Kaggle Input" in source
    assert "artifacts_unchanged = hashes_after == artifact_hashes_before" in source
    assert "if not artifacts_unchanged" in source
    assert "EXPECTED_CHECKPOINT_EPOCH = 31" in source
    assert "EXPECTED_SEED = 42" in source
    assert "EXPECTED_CONFIG_HASH" in source


def test_sha_locator_uses_content_identity_and_rejects_ambiguity(tmp_path):
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

    duplicate_root = input_root / "duplicate-artifacts"
    duplicate_root.mkdir()
    (duplicate_root / basename).write_bytes(content)
    with pytest.raises(RuntimeError, match="Expected exactly one SHA-matched"):
        namespace["locate_unique_sha_artifact"](artifact_spec)


def test_adapter_preflight_reads_validation_sample_paths_only():
    pre_run_source = "\n".join(_code_cells()[:5])
    assert 'FER_VAL_CSV = FER_SPLIT_ROOT / "val.csv"' in pre_run_source
    assert 'PRIOR_VAL_DIR = PRIOR_ROOT / "val"' in pre_run_source
    assert 'GRAPH_CACHE_VAL_DIR = GRAPH_CACHE_ROOT / "val"' in pre_run_source
    assert '"sample_split": "val"' in pre_run_source
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
    assert "CACHE_COMPLETE_PATH" in pre_run_source
    assert "non-sample aggregate metadata" in pre_run_source


def test_registered_command_invokes_step7_once_on_validation_and_is_unbounded():
    run_cell = _code_cells()[5]
    tree = ast.parse(run_cell)
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "probe_command" for target in node.targets)
    ]
    assert len(assignments) == 1
    command = assignments[0].value
    assert isinstance(command, ast.List)
    string_arguments = [
        element.value
        for element in command.elts
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
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
        assert argument in string_arguments
    assert not any(argument.startswith("--limit-") for argument in string_arguments)
    assert "run_checked(probe_command" not in run_cell
    assert run_cell.count("run_probe_with_failure_archive(probe_command") == 1
    assert "subprocess.run(" in run_cell
    assert "probe_command.count(PROBE_TOOL_PATH) != 1" in run_cell
    assert "PROBE_TOOL_PATH" in run_cell
    assert "SUPPORT_TOOL_PATH" not in run_cell
    constants = _code_cells()[0]
    assert "EVAL_BATCH_SIZE = 32" in constants
    assert "GRAPH_WORKERS = 2" in constants
    assert "GRAPH_CACHE_SIZE = 64" in constants


def test_nonzero_step7_subprocess_returns_normally_with_failure_archive(tmp_path):
    run_cell = _code_cells()[5]
    tree = ast.parse(run_cell)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    working = tmp_path / "working"
    run_root = working / "tf_step8_direct_part_decomposition"
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
    pre_run.write_text('{"issue": 15}\n', encoding="utf-8")
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
        "SUBPROCESS_LOG_PATH": run_root / "step7_subprocess.log",
        "REPORT_PATH": working / "tf_step8_direct_part_decomposition.md",
        "ARCHIVE_PATH": working / "tf_step8_direct_part_decomposition_kaggle_t4.zip",
        "located_artifacts": located,
        "artifact_hashes_before": artifact_hashes,
        "sha256": _sha256,
        "EXPECTED_SCIENTIFIC_BASE_COMMIT": EXPECTED_BASE,
        "EXPECTED_EXECUTION_COMMIT": EXPECTED_EXECUTION,
        "EXPECTED_PROBE_TOOL_SHA256": EXPECTED_PROBE_SHA,
        "EXPECTED_SUPPORT_TOOL_SHA256": EXPECTED_SUPPORT_SHA,
        "EXPECTED_SCIENTIFIC_PAYLOAD_SHA256": EXPECTED_PAYLOAD_SHA,
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), "wrapper", "exec"),
        namespace,
    )
    outcome = namespace["run_probe_with_failure_archive"](
        [
            sys.executable,
            "-c",
            "import sys; print('synthetic Step-7 failure'); sys.exit(7)",
        ],
        tmp_path,
    )
    assert outcome["result"].returncode == 7
    wrapper_payload = outcome["wrapper_execution"]
    assert wrapper_payload["status"] == "TECHNICAL_OR_GATE_FAILURE"
    assert wrapper_payload["subprocess_return_code"] == 7
    assert wrapper_payload["scientific_result_valid"] is False
    assert wrapper_payload["scientific_interpretation"] is None
    assert wrapper_payload["training"] is False
    assert wrapper_payload["test_access"] is False
    assert wrapper_payload["artifact_hashes_before"] == artifact_hashes
    assert wrapper_payload["artifact_hashes_after"] == artifact_hashes

    namespace.update({
        "wrapper_execution": wrapper_payload,
        "wrapper_status": wrapper_payload["status"],
        "initial_archive_names": outcome["archive_names"],
    })
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
        assert any(name.endswith("step7_subprocess.log") for name in names)
        assert any(name.endswith("adapter_metadata/pre_run_manifest.json") for name in names)
        assert any(name.endswith("adapter_metadata/wrapper_execution.json") for name in names)
        assert any(name.endswith("adapter_metadata/failure_status.json") for name in names)
        assert any(name.endswith("probe/partial_probe_evidence.json") for name in names)
        assert "tf_step8_direct_part_decomposition.md" in names
        assert not any(name.endswith("final_evidence.json") for name in names)
        assert not any(name.endswith(".keras") for name in names)
        assert "synthetic Step-7 failure" in archive.read(
            "tf_step8_direct_part_decomposition/step7_subprocess.log"
        ).decode()
    assert not (adapter_root / "final_evidence.json").exists()
    report = namespace["REPORT_PATH"].read_text(encoding="utf-8")
    assert "TECHNICAL_OR_GATE_FAILURE" in report
    assert "Scientific result valid: `false`" in report
    assert "Scientific interpretation: `null`" in report


def test_notebook_branches_failure_and_success_evidence_paths():
    run_cell, verify_cell, report_cell, archive_cell = _code_cells()[5:9]
    assert '"status": "TECHNICAL_OR_GATE_FAILURE"' in run_cell
    assert '"scientific_result_valid": False' in run_cell
    assert 'wrapper_status == "SUBPROCESS_COMPLETE_PENDING_VERIFICATION"' in verify_cell
    assert 'wrapper_status == "TECHNICAL_OR_GATE_FAILURE"' in verify_cell
    assert "verify_complete_probe_evidence()" in verify_cell
    assert "record_failure_and_archive(" in verify_cell
    assert 'wrapper_status == "COMPLETE"' in report_cell
    assert "verify_failure_archive(initial_archive_names)" in report_cell
    assert 'wrapper_status == "COMPLETE"' in archive_cell
    assert "verify_failure_archive(archived_names)" in archive_cell


def test_adapter_requires_t4_registered_versions_and_fresh_compact_outputs():
    source = _all_source()
    assert 'all("T4" in name for name in gpu_names)' in source
    assert 'TESTED_TENSORFLOW = "2.18.1"' in source
    assert 'TESTED_KERAS = "3.15.0"' in source
    assert "Registered software/GPU mismatch" in source
    assert "Registered output must be fresh" in source
    assert "/kaggle/working/tf_step8_direct_part_decomposition.md" in source
    assert "/kaggle/working/tf_step8_direct_part_decomposition_kaggle_t4.zip" in source
    assert "memory_growth_requested" in source
    assert "memory_growth_status" in source


def test_gate_evidence_conditions_and_interpretation_are_harness_owned_and_preserved():
    source = _all_source()
    assert (
        'NATIVE_MANUAL_TOLERANCE = {"prediction_agreement": 1.0, '
        '"max_abs_logit_difference": 1e-5, '
        '"max_abs_probability_difference": 3e-6}'
    ) in source
    assert '"max_abs_probability_difference": 1e-6' not in source
    for condition in CONDITIONS:
        assert condition in source
    for token in (
        "gate_a_native_manual_equivalence",
        "gate_b_d0_reference",
        "gate_c_d5_anchor",
        "VALID_REGISTERED_DECOMPOSITION",
        "registered_gates_and_diagnostics",
        "per_path_diagnostics",
        "overall_decision",
        "paired_diagnostics",
        "native_manual_d0_equivalence",
        "intervention_integrity",
        "checks_per_condition",
        "D1-D4 effects are nonlinear",
        "must not be summed",
    ):
        assert token in source
    for reference in (
        "0.63137364168292",
        "0.5932591901893336",
        "1.1537981724317095",
        "0.27751462803009197",
        "0.19745892656222366",
        "1.757720434560185",
    ):
        assert reference in source
    post_run = _code_cells()[6]
    assert "verify_reference_gate(gate_b, D0_REFERENCE" in post_run
    assert "verify_reference_gate(gate_c, D5_REFERENCE" in post_run
    assert "if delta" not in post_run
    assert "HIGH_PATH_SENSITIVITY" not in post_run


def test_post_run_fails_closed_on_inventory_identity_immutability_and_isolation():
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
        '"topology_changed": False',
        '"checkpoint_unchanged": True',
        '"model_weights_unchanged": True',
        '"training_or_optimizer_state_created": False',
        '"test_split_constructed": False',
    ):
        assert token in source
    assert 'name.endswith(".keras")' in source
    assert 'Path(name).name in {"train.csv", "test.csv"}' in source


def test_no_training_test_lifecycle_or_scientific_intervention_logic():
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
        "enable_op_determinism",
    ):
        assert forbidden not in source
    assert '"training": False' in source
    assert '"test_access": False' in source


def test_documented_kaggle_inputs_internet_outputs_and_review_stop():
    markdown_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in _notebook()["cells"]
        if cell.get("cell_type") == "markdown"
    )
    for path in (
        "/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split",
        "/kaggle/input/datasets/irthn1311/d16-mediapipe-pixel-priors-best-retry-rescue/outputs/d16_mediapipe_pixel_priors_best_retry_rescue",
        "/kaggle/input/datasets/irthn1311/ofix7-mid-seed42-records",
        "/kaggle/working/tf_step8_direct_part_decomposition.md",
        "/kaggle/working/tf_step8_direct_part_decomposition_kaggle_t4.zip",
    ):
        assert path in markdown_source
    assert "Internet is required only" in markdown_source
    assert "attach one separate read-only Kaggle Input" in markdown_source
    assert "Pre-run review gate" in markdown_source
    assert "PRE-INTERVENTION TECHNICAL HARNESS FAILURE" in markdown_source
    assert "INVALID_MANUAL_FORWARD_EQUIVALENCE" in markdown_source
    assert "before any D1-D4 intervention outcome" in markdown_source
    assert "POST-HOTFIX PRE-INTERVENTION TECHNICAL HARNESS FAILURE" in markdown_source
    assert "Neither attempt produced a valid D1-D4 scientific outcome" in markdown_source
    assert "113` batches / `3589` samples" in markdown_source
    assert "original `1e-6` probability tolerance" in markdown_source
    assert EXPECTED_FORENSIC_ARCHIVE_SHA in markdown_source
    assert "not a D1-D4 scientific decomposition result" in markdown_source
