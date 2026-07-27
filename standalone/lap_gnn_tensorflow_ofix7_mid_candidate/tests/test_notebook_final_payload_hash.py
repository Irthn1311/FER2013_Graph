from _adamw_closure_evidence import REPO_ROOT, sha256


def test_notebook_final_payload_hash():
    notebook = REPO_ROOT / "notebooks" / "kaggle-end-to-end.ipynb"
    assert sha256(notebook) == (
        "aa431794d9f5893fc0b58b53f6b694fc229668002f5de2d3931b36cb03e0de2e"
    )
