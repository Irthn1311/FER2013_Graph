from _adamw_closure_evidence import REPO_ROOT, sha256


def test_notebook_final_payload_hash():
    notebook = REPO_ROOT / "notebooks" / "kaggle-end-to-end.ipynb"
    assert sha256(notebook) == (
        "42f3f34aba67d02e7b318525b9c85447ad0b182fd898e4813b8c3f875e3d8482"
    )
