from __future__ import annotations

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
NOTEBOOK_PATH = ROOT / "notebooks" / "kaggle-issue15-gate-a-forensic.ipynb"
BUILDER_PATH = ROOT / "tools" / "build_issue15_gate_a_forensic_adapter.py"
PACKAGE_ROOT = ROOT / "standalone" / "lap_gnn_tensorflow_ofix7_mid_candidate"
FORENSIC_TOOL = PACKAGE_ROOT / "tools" / "evaluate_gate_a_forensic_probe.py"
STEP7_TOOL = PACKAGE_ROOT / "tools" / "evaluate_fixed_checkpoint_direct_part_decomposition_probe.py"
STEP6_SUPPORT = PACKAGE_ROOT / "tools" / "evaluate_fixed_checkpoint_prior_probe.py"

EXPECTED_BASE = "d14e7e1e3eec2ffbd5339b5a3bd0d5db5ab3de8b"
EXPECTED_HOTFIX = "a1b1d279bb9ec388f1d93ad86196e423dc750ad1"
EXPECTED_EXECUTION = "3cae1f6c78048cd6cd518d87cd0a5429d72f01e1"
EXPECTED_FORENSIC_SHA = "30c00fd6985810533cc09be05f66b64f7da5a794903aef493b9839b461eac7c0"
EXPECTED_STEP7_SHA = "fc60ece71caea14927c4840edfcd527d005737106f60d0bb475b9b1ba79eadd3"
EXPECTED_STEP6_SHA = "3a033c977a29e102cfed75282ae7c1062f41feac8bef1b955ae425ec7e4004b3"
EXPECTED_PAYLOAD_SHA = "286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e"
EXPECTED_ARTIFACT_HASHES = {
    "9ec11bb819f97e4fbda432f68da76c1201b8a3f9e06fae9eb30489a528d6ac16",
    "e62cf8c86f0d6a56c3041911de6397d18f47276f79f03c6a50ca71fa47300a37",
    "3c028dd2f32ebed3a252544e170220b150b5e29920cea865924dddce6aef5a32",
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def test_notebook_is_deterministic_unexecuted_and_compiles():
    spec = importlib.util.spec_from_file_location("forensic_builder", BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert _notebook() == module.build_notebook()

    notebook = _notebook()
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 19
    for index, cell in enumerate(notebook["cells"]):
        assert cell["id"] == f"issue15-forensic-{index:02d}"
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            compile("".join(cell["source"]), f"cell-{index}", "exec")


def test_exact_source_payload_tool_and_artifact_locks_match_repository():
    source = _all_source()
    for value in (
        EXPECTED_BASE,
        EXPECTED_HOTFIX,
        EXPECTED_EXECUTION,
        EXPECTED_FORENSIC_SHA,
        EXPECTED_STEP7_SHA,
        EXPECTED_STEP6_SHA,
        EXPECTED_PAYLOAD_SHA,
    ):
        assert value in source
    for value in EXPECTED_ARTIFACT_HASHES:
        assert value in source
    assert _sha256(FORENSIC_TOOL) == EXPECTED_FORENSIC_SHA
    assert _sha256(STEP7_TOOL) == EXPECTED_STEP7_SHA
    assert _sha256(STEP6_SUPPORT) == EXPECTED_STEP6_SHA
    manifest = json.loads((PACKAGE_ROOT / "package_manifest.json").read_text())
    assert manifest["scientific_payload_sha256"] == EXPECTED_PAYLOAD_SHA
    assert '"git", "checkout", "--detach", EXPECTED_EXECUTION_COMMIT' in source
    assert '"git", "merge-base", "--is-ancestor"' in source
    for ancestor in (EXPECTED_BASE, EXPECTED_HOTFIX):
        completed = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, EXPECTED_EXECUTION],
            cwd=ROOT,
        )
        assert completed.returncode == 0


def test_validation_only_inputs_are_preflighted_and_graph_rebuild_is_forbidden():
    source = _all_source()
    for required in (
        'FER_VAL_CSV = FER_SPLIT_ROOT / "val.csv"',
        'PRIOR_VAL_DIR = PRIOR_ROOT / "val"',
        'GRAPH_CACHE_VAL_DIR = GRAPH_CACHE_ROOT / "val"',
        'cache_val_index_path = GRAPH_CACHE_VAL_DIR / "index.json"',
        '"sample_split": "val"',
        '"graph_rebuild_allowed": False',
    ):
        assert required in source
    pre_run_source = "\n".join(_code_cells()[:5])
    for forbidden in (
        'FER_SPLIT_ROOT / "train.csv"',
        'FER_SPLIT_ROOT / "test.csv"',
        'PRIOR_ROOT / "train"',
        'PRIOR_ROOT / "test"',
        'GRAPH_CACHE_ROOT / "train"',
        'GRAPH_CACHE_ROOT / "test"',
    ):
        assert forbidden not in pre_run_source


def test_forensic_command_is_single_full_validation_and_never_invokes_d1_d5():
    run_cell = _code_cells()[5]
    tree = ast.parse(run_cell)
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "forensic_command"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1
    command = assignments[0].value
    assert isinstance(command, ast.List)
    strings = {
        element.value
        for element in command.elts
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    }
    for required in (
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
        assert required in strings
    assert not any(value.startswith("--limit-") for value in strings)
    assert run_cell.count("run_forensic_with_failure_archive(") == 2
    assert "FORENSIC_TOOL_PATH" in run_cell
    assert "STEP7_TOOL_PATH in forensic_command" in run_cell
    assert "STEP6_SUPPORT_PATH in forensic_command" in run_cell
    for forbidden in (
        "CONDITION_D1",
        "CONDITION_D2",
        "CONDITION_D3",
        "CONDITION_D4",
        "CONDITION_D5",
        "manual_forward(",
        "evaluate_conditions(",
        "model.fit(",
        "enable_op_determinism",
    ):
        assert forbidden not in run_cell


def test_reference_tolerances_dtype_evidence_and_non_scientific_boundary():
    source = _all_source()
    for token in (
        '"prediction_agreement": 1.0',
        '"max_abs_logit_difference": 1e-5',
        '"max_abs_probability_difference": 1e-6',
        "gate_a_tolerances_are_diagnostic_only",
        "stop_on_reference_exceedance",
        "native_1_vs_native_2",
        "manual_1_vs_manual_2",
        "native_1_vs_manual_1",
        "native_2_vs_manual_2",
        "outer_lap_gnn",
        "part_global_context",
        '"scientific_decomposition_run": False',
        '"intervention_conditions_executed": []',
        '"scientific_interpretation": None',
    ):
        assert token in source


def test_nonzero_subprocess_always_archives_partial_evidence_and_log(tmp_path):
    run_cell = _code_cells()[5]
    tree = ast.parse(run_cell)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    working = tmp_path / "working"
    run_root = working / "tf_step8_gate_a_forensic"
    forensic_root = run_root / "forensic"
    adapter_root = run_root / "adapter_metadata"
    forensic_root.mkdir(parents=True)
    adapter_root.mkdir(parents=True)
    partial = forensic_root / "batches/batch_00000.json"
    partial.parent.mkdir()
    partial.write_text('{"batch_index": 0}\n', encoding="utf-8")
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
        "FORENSIC_OUTPUT_ROOT": forensic_root,
        "ADAPTER_METADATA_ROOT": adapter_root,
        "SUBPROCESS_LOG_PATH": run_root / "forensic_subprocess.log",
        "REPORT_PATH": working / "tf_step8_gate_a_forensic.md",
        "ARCHIVE_PATH": working / "tf_step8_gate_a_forensic_kaggle_t4.zip",
        "located_artifacts": located,
        "artifact_hashes_before": artifact_hashes,
        "sha256": _sha256,
        "EXPECTED_SCIENTIFIC_BASE_COMMIT": EXPECTED_BASE,
        "EXPECTED_HOTFIX_ANCESTOR_COMMIT": EXPECTED_HOTFIX,
        "EXPECTED_EXECUTION_COMMIT": EXPECTED_EXECUTION,
        "EXPECTED_FORENSIC_TOOL_SHA256": EXPECTED_FORENSIC_SHA,
        "EXPECTED_STEP7_TOOL_SHA256": EXPECTED_STEP7_SHA,
        "EXPECTED_STEP6_SUPPORT_SHA256": EXPECTED_STEP6_SHA,
        "EXPECTED_SCIENTIFIC_PAYLOAD_SHA256": EXPECTED_PAYLOAD_SHA,
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), "wrapper", "exec"),
        namespace,
    )
    with pytest.raises(RuntimeError, match="partial forensic archive"):
        namespace["run_forensic_with_failure_archive"](
            [sys.executable, "-c", "import sys; sys.exit(7)"], tmp_path
        )

    archive_path = namespace["ARCHIVE_PATH"]
    assert archive_path.is_file()
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        assert any(name.endswith("forensic/batches/batch_00000.json") for name in names)
        assert any(name.endswith("forensic_subprocess.log") for name in names)
        assert any(name.endswith("adapter_metadata/wrapper_execution.json") for name in names)
        assert "tf_step8_gate_a_forensic.md" in names
    wrapper_payload = json.loads(
        (adapter_root / "wrapper_execution.json").read_text(encoding="utf-8")
    )
    assert wrapper_payload["returncode"] == 7
    assert wrapper_payload["status"] == "TECHNICAL_FORENSIC_FAILURE"
    assert wrapper_payload["artifacts_unchanged"] is True


def test_documented_inputs_internet_outputs_and_review_stop():
    markdown_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in _notebook()["cells"]
        if cell.get("cell_type") == "markdown"
    )
    for path in (
        "/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split",
        "/kaggle/input/datasets/irthn1311/d16-mediapipe-pixel-priors-best-retry-rescue/outputs/d16_mediapipe_pixel_priors_best_retry_rescue",
        "/kaggle/input/datasets/irthn1311/ofix7-mid-seed42-records",
        "/kaggle/working/tf_step8_gate_a_forensic.md",
        "/kaggle/working/tf_step8_gate_a_forensic_kaggle_t4.zip",
    ):
        assert path in markdown_source
    assert "Internet is required only" in markdown_source
    assert "separate read-only Kaggle Input" in markdown_source
    assert "Pre-run review gate" in markdown_source
    assert "intentionally unexecuted" in markdown_source
