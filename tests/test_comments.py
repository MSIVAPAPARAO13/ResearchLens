"""Unit tests for ResearchComment creation, retrieval, and status changes."""

import shutil
import tempfile

import pytest

from researchlens.workspace.schema import ResearchComment
from researchlens.workspace.store import WorkspaceStore


@pytest.fixture
def temp_ws_store():
    tmp_dir = tempfile.mkdtemp(prefix="rl_test_cmt_")
    store = WorkspaceStore(base_dir=tmp_dir)
    yield store
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_research_comment_lifecycle(temp_ws_store):
    store = temp_ws_store
    ws = store.create_workspace("NLP Alignment")
    ws_id = ws.workspace_id

    # Create comments on different targets
    c1 = ResearchComment(
        comment_id="cmt_01",
        workspace_id=ws_id,
        author_id="mem_alice",
        target_type="PAPER",
        target_id="doi:10.1001/paper_bert",
        content="Check baseline learning rate schedule on page 4.",
    )
    c2 = ResearchComment(
        comment_id="cmt_02",
        workspace_id=ws_id,
        author_id="mem_bob",
        target_type="PAPER",
        target_id="doi:10.1001/paper_bert",
        content="Confirmed, they used warmup for 10% of total steps.",
    )
    c3 = ResearchComment(
        comment_id="cmt_03",
        workspace_id=ws_id,
        author_id="mem_carol",
        target_type="HYPOTHESIS",
        target_id="hyp_01",
        content="Should we add latency as a secondary metric?",
    )

    store.save_comment(c1)
    store.save_comment(c2)
    store.save_comment(c3)

    assert len(store.list_comments(ws_id)) == 3

    # Filter by target
    paper_comments = store.get_comments_for_target(ws_id, "PAPER", "doi:10.1001/paper_bert")
    assert len(paper_comments) == 2

    hyp_comments = store.get_comments_for_target(ws_id, "HYPOTHESIS", "hyp_01")
    assert len(hyp_comments) == 1

    # Resolve comment
    c1.status = "RESOLVED"
    store.save_comment(c1)
    updated = store.get_comments_for_target(ws_id, "PAPER", "doi:10.1001/paper_bert")
    assert any(c.status == "RESOLVED" for c in updated)

    # Delete comment
    assert store.delete_comment(ws_id, "cmt_03") is True
    assert len(store.list_comments(ws_id)) == 2
