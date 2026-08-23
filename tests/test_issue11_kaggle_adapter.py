import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "kaggle-issue11-fixed-topology-prior-probe.ipynb"
BUILDER_PATH = ROOT / "tools" / "build_issue11_kaggle_adapter.py"
PACKAGE_ROOT = ROOT / "standalone" / "lap_gnn_tensorflow_ofix7_mid_candidate"
PROBE_TOOL = PACKAGE_ROOT / "tools" / "evaluate_fixed_checkpoint_prior_probe.py"

EXPECTED_BASE = "69f4571c5069da9a7f8558ef3c01101635ee904a"
EXPECTED_TOOL_SHA = "564eab26b7cf683bd531fec08bf6539a1384d9ef370961b9484335726c7c2351"
EXPECTED_PAYLOAD_SHA = "286be711a53b76511bcf3b9bf949fad694f7c7d272392f9defc56f4914822c0e"
EXPECTED_ARTIFACT_HASHES = {
    "9ec11bb819f97e4fbda432f68da76c1201b8a3f9e06fae9eb30489a528d6ac16",
    "e62cf8c86f0d6a56c3041911de6397d18f47276f79f03c6a50ca71fa47300a37",
    "3c028dd2f32ebed3a252544e170220b150b5e29920cea865924dddce6aef5a32",
}


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
    spec = importlib.util.spec_from_file_location("issue11_builder", BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert _notebook() == module.build_notebook()

    notebook = _notebook()
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            compile("".join(cell["source"]), f"cell-{index}", "exec")


def test_exact_issue11_source_tool_payload_and_artifact_locks():
    source = _all_source()
    assert EXPECTED_BASE in source
    assert EXPECTED_TOOL_SHA in source
    assert EXPECTED_PAYLOAD_SHA in source
    for expected in EXPECTED_ARTIFACT_HASHES:
        assert expected in source
    assert _sha256(PROBE_TOOL) == EXPECTED_TOOL_SHA
    manifest = json.loads((PACKAGE_ROOT / "package_manifest.json").read_text())
    assert manifest["scientific_payload_sha256"] == EXPECTED_PAYLOAD_SHA
    assert "git\", \"checkout\", \"--detach\", EXPECTED_COMMIT" in source
    assert "sha256(PROBE_TOOL_PATH) != EXPECTED_PROBE_TOOL_SHA256" in source


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
    assert "artifact_hashes_after != artifact_hashes_before" in source
    assert "EXPECTED_CHECKPOINT_EPOCH = 31" in source
    assert "EXPECTED_SEED = 42" in source
    assert "EXPECTED_CONFIG_HASH" in source


def test_sha_locator_uses_content_identity_and_rejects_ambiguity(tmp_path):
    artifact_cell = _code_cells()[3]
    tree = ast.parse(artifact_cell)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = {
        "os": __import__("os"),
        "Path": Path,
        "sha256": _sha256,
    }
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
    spec = {"basename": basename, "sha256": _sha256(expected_path)}
    assert namespace["locate_unique_sha_artifact"](spec) == expected_path.resolve()

    duplicate_root = input_root / "duplicate-artifacts"
    duplicate_root.mkdir()
    (duplicate_root / basename).write_bytes(content)
    with pytest.raises(RuntimeError, match="Expected exactly one SHA-matched"):
        namespace["locate_unique_sha_artifact"](spec)


def test_adapter_reads_validation_sample_paths_only():
    code = _code_cells()
    pre_run_source = "\n".join(code[:6])
    assert 'FER_VAL_CSV = FER_SPLIT_ROOT / "val.csv"' in pre_run_source
    assert 'PRIOR_VAL_DIR = PRIOR_ROOT / "val"' in pre_run_source
    assert 'GRAPH_CACHE_VAL_DIR = GRAPH_CACHE_ROOT / "val"' in pre_run_source
    assert 'sample_split": "val"' in pre_run_source
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


def test_registered_command_is_single_validation_only_and_unbounded():
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
    assert "--checkpoint" in string_arguments
    assert "--checkpoint-metadata" in string_arguments
    assert "--resolved-config" in string_arguments
    assert "--prior-root" in string_arguments
    assert "--clean-graph-cache-dir" in string_arguments
    assert "--output-root" in string_arguments
    assert not any(argument.startswith("--limit-") for argument in string_arguments)
    assert run_cell.count("run_checked(probe_command") == 1
    assert "probe_command.count(PROBE_TOOL_PATH) != 1" in run_cell
    assert "test" not in " ".join(string_arguments).lower()


def test_adapter_requires_t4_registered_versions_and_fresh_outputs():
    source = _all_source()
    assert 'all("T4" in name for name in gpu_names)' in source
    assert 'TESTED_TENSORFLOW = "2.18.1"' in source
    assert 'TESTED_KERAS = "3.15.0"' in source
    assert "Registered software/GPU mismatch" in source
    assert "Registered output must be fresh" in source
    assert (
        '/kaggle/working/tf_step6_fixed_topology_prior_sensitivity_kaggle_t4.zip'
        in source
    )
    assert "/kaggle/working/tf_step6_fixed_topology_prior_sensitivity.md" in source


def test_c0_gate_and_registered_diagnostics_are_locked():
    source = _all_source()
    for token in (
        "0.6319308999721371",
        "0.5938407974340496",
        "1.1538367092081931",
        'C0_TOLERANCE = {"accuracy": 0.001, "macro_f1": 0.001, "loss": 0.005}',
        "INVALID_REFERENCE_REPRODUCTION",
        "HIGH_EXPLICIT_PRIOR_SENSITIVITY",
        "MODERATE_EXPLICIT_PRIOR_SENSITIVITY",
        "LOW_EXPLICIT_PRIOR_SENSITIVITY",
        "delta_f1_c1_pp",
        "delta_f1_c2_pp",
        "delta_f1_c2_minus_c1_pp",
        "accuracy_change_pp_c0_to_c1",
        "accuracy_change_pp_c0_to_c2",
        "prediction_disagreement_rate",
        "c0_correct_to_intervention_incorrect",
        "c0_incorrect_to_intervention_correct",
        "unchanged_correct",
        "unchanged_incorrect",
        "per_class_c0_minus_c1_f1_pp",
        "per_class_c0_minus_c2_f1_pp",
    ):
        assert token in source
    assert "if delta_f1_c2 >= 10.0:" in source
    assert "elif delta_f1_c2 >= 5.0:" in source
    assert "interpreted_diagnostics = None" in source


def test_no_training_test_lifecycle_or_checkpoint_packaging():
    source = _all_source()
    for forbidden in (
        "lap_gnn_tf.cli.train",
        "train_validation_only.py",
        "model.fit(",
        "apply_gradients(",
        "PixelPriorDataset._zero_prior",
        "shuffle_prior",
        "forced_fallback",
        "attenuate_prior",
    ):
        assert forbidden not in source
    assert '"training": False' in source
    assert '"test_access": False' in source
    assert 'name.endswith(".keras")' in source
    assert "C2 measures explicit semantic-prior/direct-part sensitivity conditional" in source
    assert "not a prior-free graph" in source


def test_documented_kaggle_inputs_internet_and_compact_outputs():
    markdown_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in _notebook()["cells"]
        if cell.get("cell_type") == "markdown"
    )
    for path in (
        "/kaggle/input/datasets/doduyquynii/fer13-split/fer13-split",
        "/kaggle/input/datasets/irthn1311/d16-mediapipe-pixel-priors-best-retry-rescue/outputs/d16_mediapipe_pixel_priors_best_retry_rescue",
        "/kaggle/input/datasets/irthn1311/ofix7-mid-seed42-records",
        "/kaggle/working/tf_step6_fixed_topology_prior_sensitivity.md",
        "/kaggle/working/tf_step6_fixed_topology_prior_sensitivity_kaggle_t4.zip",
    ):
        assert path in markdown_source
    assert "Internet is required only" in markdown_source
    assert "attach one separate read-only Kaggle Input" in markdown_source
    assert "Pre-run review gate" in markdown_source
