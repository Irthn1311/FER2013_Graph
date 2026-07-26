from _adamw_closure_evidence import REPO_ROOT, sha256


def test_notebook_final_payload_hash():
    notebook = REPO_ROOT / "notebooks" / "kaggle-end-to-end.ipynb"
    assert sha256(notebook) == (
        "d134878293c8591735e4de6c1ca380658ebf046820c95e33f9e5842adf9fc7c8"
    )
