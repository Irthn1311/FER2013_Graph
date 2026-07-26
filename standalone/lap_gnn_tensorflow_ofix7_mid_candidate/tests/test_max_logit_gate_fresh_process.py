import json
import os
from pathlib import Path
import subprocess
import sys


def test_max_logit_gate_fresh_process(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "fresh.json"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    environment["TF_DETERMINISTIC_OPS"] = "1"
    environment.pop("TF_ENABLE_ONEDNN_OPTS", None)
    subprocess.run(
        [
            sys.executable,
            "-B",
            root / "tools" / "repair_repeated_forward.py",
            "--package-root",
            root,
            "--worker-output",
            output,
        ],
        check=True,
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["max_logit_difference"] <= 1e-5
    assert result["prediction_agreement"] == 1.0
