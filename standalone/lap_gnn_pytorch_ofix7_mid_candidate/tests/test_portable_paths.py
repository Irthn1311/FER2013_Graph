from pathlib import Path, PurePosixPath, PureWindowsPath


def test_windows_posix_and_space_paths_are_parseable(tmp_path):
    assert PureWindowsPath(r"C:\data with spaces\train.csv").name == "train.csv"
    assert PurePosixPath("/kaggle/input/data with spaces/train.csv").name == "train.csv"
    path = tmp_path / "output with spaces"
    path.mkdir()
    assert Path(path).is_dir()
