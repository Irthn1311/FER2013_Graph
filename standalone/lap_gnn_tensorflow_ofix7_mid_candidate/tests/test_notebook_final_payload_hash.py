from _adamw_closure_evidence import REPO_ROOT, sha256


def test_notebook_final_payload_hash():
    notebook = REPO_ROOT / "notebooks" / "kaggle-end-to-end.ipynb"
    assert sha256(notebook) == (
        "0bbf64d10695d7dd9287387d5f966aa1a2b47fa429ce337c12b707f49ec03a27"
    )
