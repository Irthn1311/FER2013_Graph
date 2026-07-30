from _adamw_closure_evidence import REPO_ROOT, sha256


def test_notebook_final_payload_hash():
    notebook = REPO_ROOT / "notebooks" / "kaggle-end-to-end.ipynb"
    assert sha256(notebook) == (
        "1181b3366bcaa10ae978c5ed48461bea52486e03a88d9c3928fa38418fc8075c"
    )
