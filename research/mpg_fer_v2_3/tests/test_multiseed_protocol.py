from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src" / "mpg_fer_v2_3"
EXPECTED_SOURCE_SHA = (
    "1e63aadd13d53024c1b279dd4cc9bbc943048a6751899d8ecbabea3b12082f87"
)
SEEDS = (0, 1, 43, 123, 3047)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(SOURCE_ROOT.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def test_multiseed_notebook_is_generated_and_source_frozen() -> None:
    generator = _load(
        "sync_multiseed",
        ROOT / "tools" / "sync_multiseed_notebook.py",
    )
    expected = generator.build_notebook()
    actual = json.loads(
        (ROOT / "notebooks" / "MPG_FER_v2_3_MultiSeed_Kaggle_T4.ipynb")
        .read_text(encoding="utf-8")
    )
    assert actual == expected
    assert _source_hash() == EXPECTED_SOURCE_SHA
    assert actual["metadata"]["mpg_fer_v2_3_issue"] == 99
    assert actual["metadata"]["mpg_fer_v2_3_base_issue"] == 97
    assert actual["metadata"]["mpg_fer_multiseed_contract"]["seeds"] == [
        0, 1, 42, 43, 123, 3047
    ]


def test_multiseed_notebook_has_seed_only_controls_and_private_firewall() -> None:
    notebook = json.loads(
        (ROOT / "notebooks" / "MPG_FER_v2_3_MultiSeed_Kaggle_T4.ipynb")
        .read_text(encoding="utf-8")
    )
    control = "".join(notebook["cells"][1]["source"])
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "SEED = 42" in control
    assert "ACCOUNT_ALIAS = \"UNASSIGNED\"" in control
    assert "MPGConfig(seed=SEED" in code
    assert "CONFIG_HASH_MISMATCH" in code
    assert "private_evaluated_only_after_freeze" in code
    assert "FP32 evaluation refused: Private freeze firewall is absent" in code
    assert "canonical_fp32_no_autocast_tf32_disabled" in code
    assert "private_raw_predictions.csv" in code
    assert "RUN_MANIFEST.json" in code
    assert "SHA256SUMS.txt" in code
    assert "mpg_v2_3_seed_{SEED}_artifacts" in code
    assert 'cfg.batch_size = 8' not in code
    assert 'gradient_accumulation_steps = 4' not in code
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell-{index}", "exec")


def test_stage_tool_enforces_registered_seed_account_assignment() -> None:
    stage = _load(
        "stage_multiseed",
        ROOT / "tools" / "stage_multiseed_kernel.py",
    )
    assert stage.AUTHORIZED_ASSIGNMENTS == {
        "A": {0, 1},
        "B": {43, 123},
        "C": {3047},
    }
    hashes = {seed: stage._assert_seed_only_config_delta(seed) for seed in SEEDS}
    assert len(set(hashes.values())) == len(SEEDS)
    assert all(len(digest) == 64 for digest in hashes.values())
