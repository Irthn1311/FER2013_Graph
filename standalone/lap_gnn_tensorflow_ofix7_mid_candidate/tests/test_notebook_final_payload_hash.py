from _adamw_closure_evidence import REPO_ROOT, sha256


def test_notebook_final_payload_hash():
    notebook = REPO_ROOT / "notebooks" / "kaggle-end-to-end.ipynb"
    assert sha256(notebook) == (
        "cc2c6c2f77ed017626d7eb5a5bfa367495a954978db5800d0b08eb833ea1f5f9"
    )
