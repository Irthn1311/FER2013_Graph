from _adamw_closure_evidence import REPO_ROOT, sha256


def test_notebook_final_payload_hash():
    notebook = REPO_ROOT / "notebooks" / "kaggle-end-to-end.ipynb"
    assert sha256(notebook) == (
        "4b44c3c736e9aaa2a8244bf147707a89147630f888f50241ffb90583e8070d5f"
    )
