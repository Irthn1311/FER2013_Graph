from _adamw_closure_evidence import REPO_ROOT, sha256


def test_notebook_final_payload_hash():
    notebook = REPO_ROOT / "notebooks" / "kaggle-end-to-end.ipynb"
    assert sha256(notebook) == (
        "68a643f1b581dfa67f1ea5366c2a9af6e44e2bf60978d43e28eb25b43efaee01"
    )
