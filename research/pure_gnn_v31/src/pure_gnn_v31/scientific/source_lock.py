"""Immutable source-lock verification for Pure-GNN scientific runs."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


class SourceLockError(RuntimeError):
    """Raised when repository state does not match the immutable reviewed source tag."""


def run_git(cmd: list[str], cwd: Path) -> str:
    res = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=True,
    )
    return res.stdout.strip()


def verify_immutable_source_lock(
    repo_root: Path,
    reviewed_source_tag: Optional[str],
) -> Dict[str, Any]:
    """Verifies that the repository is checked out at the exact immutable reviewed tag.

    Requirements:
    1. reviewed_source_tag must be a non-empty string.
    2. Tag must resolve to a peeled commit hash.
    3. HEAD must match the peeled tag commit (detached HEAD).
    4. Working tree must be completely clean (git status --porcelain is empty).
    """
    if not reviewed_source_tag:
        raise SourceLockError(
            "IMMUTABLE SOURCE LOCK FAILED: 'REVIEWED_SOURCE_TAG' is None or empty. "
            "Scientific execution requires an immutable reviewed source tag."
        )

    # 1. Resolve tag to peeled commit object
    try:
        peeled_commit = run_git(["git", "rev-parse", f"{reviewed_source_tag}^{{commit}}"], cwd=repo_root)
    except subprocess.CalledProcessError as exc:
        raise SourceLockError(
            f"IMMUTABLE SOURCE LOCK FAILED: Tag '{reviewed_source_tag}' cannot be resolved to a commit: {exc}"
        ) from exc

    # 2. Inspect HEAD commit
    try:
        current_head = run_git(["git", "rev-parse", "HEAD"], cwd=repo_root)
    except subprocess.CalledProcessError as exc:
        raise SourceLockError(f"Failed to inspect HEAD commit: {exc}") from exc

    # 3. If HEAD != peeled_commit, attempt checkout of the detached peeled commit
    if current_head != peeled_commit:
        try:
            run_git(["git", "checkout", "--detach", peeled_commit], cwd=repo_root)
            current_head = run_git(["git", "rev-parse", "HEAD"], cwd=repo_root)
        except subprocess.CalledProcessError as exc:
            raise SourceLockError(f"Failed to checkout tag commit {peeled_commit}: {exc}") from exc

    if current_head != peeled_commit:
        raise SourceLockError(
            f"IMMUTABLE SOURCE LOCK FAILED: HEAD ({current_head}) does not match peeled tag commit ({peeled_commit})."
        )

    # 4. Verify clean worktree
    status_out = run_git(["git", "status", "--porcelain"], cwd=repo_root)
    if status_out.strip():
        raise SourceLockError(
            f"IMMUTABLE SOURCE LOCK FAILED: Working tree is dirty:\n{status_out}"
        )

    return {
        "status": "SOURCE_LOCKED",
        "reviewed_source_tag": reviewed_source_tag,
        "peeled_commit": peeled_commit,
        "head_commit": current_head,
        "clean_worktree": True,
    }
