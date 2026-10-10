"""Unit tests for ResearchMilestone timeline tracking."""

import shutil
import tempfile

import pytest

from researchlens.workspace.schema import ResearchMilestone
from researchlens.workspace.store import WorkspaceStore


@pytest.fixture
def temp_ws_store():
    tmp_dir = tempfile.mkdtemp(prefix="rl_test_ms_")
    store = WorkspaceStore(base_dir=tmp_dir)
    yield store
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_research_milestones_lifecycle(temp_ws_store):
    store = temp_ws_store
    ws = store.create_workspace("Quantum NLP")
    ws_id = ws.workspace_id

    m1 = ResearchMilestone(
        milestone_id="ms_01",
        workspace_id=ws_id,
        title="Literature Synthesis & Baseline Identification",
        description="Review 20 key papers and identify primary baselines",
        target_date="2026-11-01",
        status="IN_PROGRESS",
    )
    m2 = ResearchMilestone(
        milestone_id="ms_02",
        workspace_id=ws_id,
        title="Prototype Implementation",
        description="Implement initial quantum circuit simulation model",
        target_date="2026-12-15",
        status="NOT_STARTED",
    )

    store.save_milestone(m1)
    store.save_milestone(m2)

    milestones = store.list_milestones(ws_id)
    assert len(milestones) == 2

    # Update status
    m1.status = "COMPLETED"
    store.save_milestone(m1)
    assert store.list_milestones(ws_id)[0].status == "COMPLETED"

    assert store.delete_milestone(ws_id, "ms_01") is True
    assert len(store.list_milestones(ws_id)) == 1
