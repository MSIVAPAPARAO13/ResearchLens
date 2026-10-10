"""Unit tests for ResearchReview peer-review workflows."""

import shutil
import tempfile

import pytest

from researchlens.workspace.schema import ResearchReview
from researchlens.workspace.store import WorkspaceStore


@pytest.fixture
def temp_ws_store():
    tmp_dir = tempfile.mkdtemp(prefix="rl_test_rev_")
    store = WorkspaceStore(base_dir=tmp_dir)
    yield store
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_research_review_lifecycle(temp_ws_store):
    store = temp_ws_store
    ws = store.create_workspace("Peer Review Lab")
    ws_id = ws.workspace_id

    rev = ResearchReview(
        review_id="rev_101",
        target_type="FINDING",
        target_id="fnd_01",
        reviewer_id="mem_expert_reviewer",
        decision="REQUIRES_REVISION",
        comments="Please test with 2 additional random seeds to demonstrate robustness.",
    )
    store.save_review(rev, ws_id)

    reviews = store.list_reviews(ws_id)
    assert len(reviews) == 1
    r = reviews[0]
    assert r.target_type == "FINDING"
    assert r.decision == "REQUIRES_REVISION"
    assert "additional random seeds" in r.comments

    # Update review after revisions submitted
    r.decision = "APPROVED"
    r.comments = "Seeds verified; finding confirmed."
    store.save_review(r, ws_id)
    assert store.list_reviews(ws_id)[0].decision == "APPROVED"
