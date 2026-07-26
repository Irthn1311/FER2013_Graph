import importlib.util
import json

import yaml

from _helpers import ROOT


def load_finalizer():
    path = ROOT / "tools" / "finalize_package.py"
    spec = importlib.util.spec_from_file_location(
        "lap_gnn_tf_finalize_package_test", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_runtime_signatures():
    path = ROOT / "src" / "lap_gnn_tf" / "signatures.py"
    spec = importlib.util.spec_from_file_location(
        "lap_gnn_tf_signatures_portability_test", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_finalizer_runtime_and_manifest_scientific_checksums_match():
    finalizer_checksum = load_finalizer().scientific_checksum()
    runtime_checksum = load_runtime_signatures().scientific_payload_checksum(ROOT)
    manifest = json.loads(
        (ROOT / "package_manifest.json").read_text(encoding="utf-8")
    )
    config = yaml.safe_load(
        (
            ROOT
            / "configs"
            / "fer2013_ofix7_mid_tensorflow_baseline.yaml"
        ).read_text(encoding="utf-8")
    )
    assert finalizer_checksum == runtime_checksum
    assert manifest["scientific_payload_sha256"] == runtime_checksum
    assert config["locked"]["package_checksum"] == runtime_checksum
