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
    2. Must be an exact tag ref: refs/tags/<reviewed_source_tag>
    3. Tag must resolve to a peeled commit hash.
    4. HEAD must match the peeled tag commit.
    5. HEAD must be DETACHED (symbolic-ref HEAD fails).
    6. Working tree must be completely clean (git status --porcelain is empty).
    """
    if not reviewed_source_tag or not isinstance(reviewed_source_tag, str):
        raise SourceLockError(
            "IMMUTABLE SOURCE LOCK FAILED: 'reviewed_source_tag' is None or empty. "
            "Scientific execution requires an immutable reviewed source tag."
        )

    exact_tag_ref = f"refs/tags/{reviewed_source_tag}"

    # If remote exists and tag is missing locally, attempt to fetch exact tag ref
    try:
        run_git(["git", "show-ref", "--verify", "--quiet", exact_tag_ref], cwd=repo_root)
    except subprocess.CalledProcessError:
        # Tag not found locally, try fetching from origin if remote exists
        try:
            run_git(["git", "fetch", "origin", f"{exact_tag_ref}:{exact_tag_ref}"], cwd=repo_root)
        except Exception:
            pass

    # 1. Require exact tag ref exists (rejects branches or arbitrary revs)
    try:
        run_git(["git", "show-ref", "--verify", "--quiet", exact_tag_ref], cwd=repo_root)
    except subprocess.CalledProcessError as exc:
        raise SourceLockError(
            f"IMMUTABLE SOURCE LOCK FAILED: Exact tag ref '{exact_tag_ref}' does not exist."
        ) from exc

    # 2. Resolve tag to peeled commit object
    try:
        peeled_commit = run_git(["git", "rev-parse", f"{exact_tag_ref}^{{commit}}"], cwd=repo_root)
    except subprocess.CalledProcessError as exc:
        raise SourceLockError(
            f"IMMUTABLE SOURCE LOCK FAILED: Tag '{reviewed_source_tag}' cannot be resolved to a commit: {exc}"
        ) from exc

    # 3. Inspect current HEAD commit
    try:
        current_head = run_git(["git", "rev-parse", "HEAD"], cwd=repo_root)
    except subprocess.CalledProcessError as exc:
        raise SourceLockError(f"Failed to inspect HEAD commit: {exc}") from exc

    # 4. If HEAD != peeled_commit, perform controlled detached checkout
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

    # 5. Verify HEAD is DETACHED (symbolic-ref must fail / return non-zero)
    sym_ref_res = subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    if sym_ref_res.returncode == 0 and sym_ref_res.stdout.strip():
        branch_name = sym_ref_res.stdout.strip()
        raise SourceLockError(
            f"IMMUTABLE SOURCE LOCK FAILED: HEAD is attached to mutable branch '{branch_name}'. "
            "A detached HEAD at the peeled tag commit is strictly required."
        )

    # 6. Verify clean worktree
    status_out = run_git(["git", "status", "--porcelain"], cwd=repo_root)
    if status_out.strip():
        raise SourceLockError(
            f"IMMUTABLE SOURCE LOCK FAILED: Working tree is dirty:\n{status_out}"
        )

    return {
        "status": "SOURCE_LOCKED",
        "reviewed_source_tag": reviewed_source_tag,
        "exact_tag_ref": exact_tag_ref,
        "peeled_commit": peeled_commit,
        "head_commit": current_head,
        "is_detached": True,
        "clean_worktree": True,
    }
