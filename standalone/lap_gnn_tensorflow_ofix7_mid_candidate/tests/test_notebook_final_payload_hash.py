from _adamw_closure_evidence import REPO_ROOT, sha256


def test_notebook_final_payload_hash():
    notebook = REPO_ROOT / "notebooks" / "kaggle-end-to-end.ipynb"
    assert sha256(notebook) == (
        "3bd81a4aa1b7fc3f4863d6b7a4cd983bbfc225f4eb3e1ba85c89dd16b8e68cca"
    )
