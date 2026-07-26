from _adamw_closure_evidence import REPO_ROOT, sha256


def test_notebook_final_payload_hash():
    notebook = REPO_ROOT / "notebooks" / "kaggle-end-to-end.ipynb"
    assert sha256(notebook) == (
        "358c7a3ae390edda93e9474f5a25f63c3cae57d1ab4495b49ccee070a1af295b"
    )
