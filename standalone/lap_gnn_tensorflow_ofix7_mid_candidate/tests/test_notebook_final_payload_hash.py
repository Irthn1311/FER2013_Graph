from _adamw_closure_evidence import REPO_ROOT, sha256


def test_notebook_final_payload_hash():
    notebook = REPO_ROOT / "notebooks" / "kaggle-end-to-end.ipynb"
    assert sha256(notebook) == (
        "62234f1bee6cd64b08fcfcf3655f8a03415f0db8e19b87f553e5fe887e5ca1bc"
    )
