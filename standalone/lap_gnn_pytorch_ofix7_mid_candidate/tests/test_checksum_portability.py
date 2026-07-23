import hashlib
import importlib.util
from pathlib import Path


VERIFY_PATH = Path(__file__).resolve().parents[1] / "tools" / "verify_checksums.py"
SPEC = importlib.util.spec_from_file_location("standalone_verify_checksums", VERIFY_PATH)
assert SPEC is not None and SPEC.loader is not None
VERIFY_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY_MODULE)
canonical_bytes = VERIFY_MODULE.canonical_bytes


def test_text_checksums_are_independent_of_newline_style(tmp_path):
    windows = tmp_path / "windows.yaml"
    linux = tmp_path / "linux.yaml"
    windows.write_bytes(b"seed: 42\r\nrun_name: test\r\n")
    linux.write_bytes(b"seed: 42\nrun_name: test\n")

    windows_hash = hashlib.sha256(canonical_bytes(windows)).hexdigest()
    linux_hash = hashlib.sha256(canonical_bytes(linux)).hexdigest()

    assert windows_hash == linux_hash


def test_binary_checksums_preserve_raw_bytes(tmp_path):
    binary = tmp_path / "fixture.npy"
    payload = b"\x93NUMPY\r\n\x00\x01"
    binary.write_bytes(payload)

    assert canonical_bytes(binary) == payload
