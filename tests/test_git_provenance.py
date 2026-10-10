"""Tests for Git Provenance & Source Snapshot Integration (Phase 14)."""

import tempfile

from researchlens.execution.environment import (
    capture_source_snapshot,
    get_git_commit_sha,
    get_git_provenance,
)
from researchlens.workspace.schema import SourceSnapshot


def test_git_provenance_in_active_repo():
    """Verify git provenance captures commit and branch without fabricating metadata."""
    prov = get_git_provenance(".")
    assert "commit_sha" in prov
    assert "branch" in prov
    assert "dirty_worktree" in prov

    # In active git repo, commit_sha is 40 chars hex or NOT_AVAILABLE
    if prov["commit_sha"] != "NOT_AVAILABLE":
        assert len(prov["commit_sha"]) == 40
        assert isinstance(prov["dirty_worktree"], bool)


def test_git_provenance_in_empty_non_git_dir():
    """Verify non-git directory returns NOT_AVAILABLE safely."""
    with tempfile.TemporaryDirectory() as empty_dir:
        prov = get_git_provenance(empty_dir)
        assert prov["commit_sha"] == "NOT_AVAILABLE"
        assert prov["branch"] == "NOT_AVAILABLE"
        assert prov["dirty_worktree"] is False

        sha = get_git_commit_sha(empty_dir)
        assert sha == "NOT_AVAILABLE"


def test_source_snapshot_creation():
    """Verify SourceSnapshot schema fields."""
    snapshot = capture_source_snapshot(".")
    assert isinstance(snapshot, SourceSnapshot)
    d = snapshot.to_dict()
    assert "repository" in d
    assert "branch" in d
    assert "commit" in d
    assert "dirty" in d
    assert "snapshot_hash" in d
