from pathlib import Path

from _helpers import ROOT


def test_portable_paths():
    forbidden = ("D:\\\\SGU", "C:\\\\Users", "/kaggle/input/datasets/")
    for path in list((ROOT / "src").rglob("*.py")) + list((ROOT / "configs").rglob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        assert not any(item in text for item in forbidden)

