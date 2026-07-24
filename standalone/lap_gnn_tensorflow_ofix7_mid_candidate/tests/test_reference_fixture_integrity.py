import hashlib
import json

from _helpers import ROOT


def test_reference_fixture_integrity():
    manifest = json.loads((ROOT / "validation_assets" / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        path = ROOT / "validation_assets" / "golden" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected

