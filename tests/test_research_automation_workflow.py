"""Unit tests for individual stages of ResearchAutomationWorkflow (Phase 15)."""

import os
import shutil
import tempfile

import pytest

from researchlens.automation.workflow import ResearchWorkflowOrchestrator
from researchlens.workspace.schema import (
    ResearchGap,
    SavedPaper,
    WorkspaceMember,
)
from researchlens.workspace.store import WorkspaceStore


@pytest.fixture
def workflow_workspace():
    temp_dir = tempfile.mkdtemp(prefix="rl_test_stages_")
    store = WorkspaceStore(base_dir=os.path.join(temp_dir, "workspaces"))
    ws = store.create_workspace("Workflow Stages Test Workspace")
    ws_id = ws.workspace_id

    store.save_member(WorkspaceMember("m_owner", ws_id, "Alice", "alice@lab.org", "OWNER"))
    store.save_member(WorkspaceMember("m_researcher", ws_id, "Bob", "bob@lab.org", "RESEARCHER"))

    yield store, ws_id
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_workflow_initialization_and_store_lifecycle(workflow_workspace):
    store, ws_id = workflow_workspace
    orchestrator = ResearchWorkflowOrchestrator(store)

    wf = orchestrator.initialize_workflow(
        workspace_id=ws_id,
        title="Stage Test Workflow",
        topic="Cross-Domain QA",
        actor_id="m_researcher",
    )

    assert wf.workflow_id.startswith("wf_")
    assert wf.current_stage == "LITERATURE_DISCOVERY"
    assert wf.status == "IN_PROGRESS"

    # Store persistence check
    loaded = store.get_workflow(ws_id, wf.workflow_id)
    assert loaded is not None
    assert loaded.title == "Stage Test Workflow"

    # Stage 1: Literature Discovery
    paper = SavedPaper(paper_id="p1", workspace_id=ws_id, title="Test Paper", authors=["A"], year=2024)
    store.save_paper(paper)

    wf, err = orchestrator.run_stage_literature_discovery(wf.workflow_id, ws_id)
    assert err is None
    assert wf.current_stage == "EVIDENCE_COLLECTION"
    assert wf.stage_data["literature"]["paper_count"] == 1

    # Stage 2: Evidence Collection (empty gracefully handled)
    wf, err = orchestrator.run_stage_evidence_collection(wf.workflow_id, ws_id)
    assert err is None
    assert wf.current_stage == "GAP_VALIDATION"
    assert wf.stage_data["evidence"]["evidence_count"] == 0

    # Stage 3: Gap Validation
    gap = ResearchGap(gap_id="g1", workspace_id=ws_id, category="DATA", description="Limited dataset size")
    store.save_gap(gap)
    wf, err = orchestrator.run_stage_gap_validation(wf.workflow_id, ws_id)
    assert err is None
    assert wf.current_stage == "RESEARCH_QUESTION"
    assert wf.stage_data["gaps"]["gap_count"] == 1

    # Stage 4: Question Formulation
    wf, err = orchestrator.run_stage_research_question(wf.workflow_id, ws_id)
    assert err is None
    assert wf.current_stage == "HYPOTHESIS_PROPOSAL"
    assert wf.stage_data["questions"]["question_count"] >= 1

    # Stage 5: Hypothesis Proposal
    wf, err = orchestrator.run_stage_hypothesis_proposal(wf.workflow_id, ws_id)
    assert err is None
    assert wf.current_stage == "EXPERIMENT_PLAN"
    assert "proposal" in wf.stage_data["hypothesis"]

    # Stage 6: Experiment Plan
    wf, err = orchestrator.run_stage_experiment_plan(wf.workflow_id, ws_id)
    assert err is None
    assert wf.current_stage == "RESEARCHER_APPROVAL"
    assert wf.status == "AWAITING_APPROVAL"


def test_workflow_nonexistent_ids(workflow_workspace):
    store, ws_id = workflow_workspace
    orchestrator = ResearchWorkflowOrchestrator(store)

    # Calling stage on nonexistent workflow returns clean error
    wf, err = orchestrator.run_stage_literature_discovery("nonexistent_id", ws_id)
    assert wf is None
    assert "not found" in err
