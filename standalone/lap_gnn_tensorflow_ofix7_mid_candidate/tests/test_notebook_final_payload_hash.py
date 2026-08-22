from _adamw_closure_evidence import REPO_ROOT, sha256


def test_notebook_final_payload_hash():
    notebook = REPO_ROOT / "notebooks" / "kaggle-end-to-end.ipynb"
    assert sha256(notebook) == (
        "1b023ef2b9b0f0d02d58779cb9b5a1d1777124e022e1e1c3e630ea3d30ed2499"
    )
