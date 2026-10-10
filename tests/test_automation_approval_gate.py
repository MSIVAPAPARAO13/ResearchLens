"""Unit tests for researcher approval gate and execution authorization (Phase 15)."""

import os
import shutil
import tempfile

import pytest

from researchlens.automation.workflow import ResearchWorkflowOrchestrator
from researchlens.workspace.schema import WorkspaceMember
from researchlens.workspace.store import WorkspaceStore


@pytest.fixture
def auth_workspace():
    temp_dir = tempfile.mkdtemp(prefix="rl_test_gate_")
    store = WorkspaceStore(base_dir=os.path.join(temp_dir, "workspaces"))
    ws = store.create_workspace("Approval Gate Test Workspace")
    ws_id = ws.workspace_id

    # Create members
    store.save_member(WorkspaceMember("m_owner", ws_id, "Alice", "alice@lab.org", "OWNER"))
    store.save_member(WorkspaceMember("m_researcher", ws_id, "Bob", "bob@lab.org", "RESEARCHER"))
    store.save_member(WorkspaceMember("m_reviewer", ws_id, "Carol", "carol@lab.org", "REVIEWER"))
    store.save_member(WorkspaceMember("m_viewer", ws_id, "David", "david@lab.org", "VIEWER"))

    yield store, ws_id
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_approval_gate_enforces_permissions(auth_workspace):
    store, ws_id = auth_workspace
    orchestrator = ResearchWorkflowOrchestrator(store)

    wf = orchestrator.initialize_workflow(ws_id, "Auth Gate Workflow", topic="Biomedical NLP")
    wf, err = orchestrator.run_stage_experiment_plan(wf.workflow_id, ws_id)
    assert err is None
    assert wf.approval_status == "PENDING"
    assert wf.status == "AWAITING_APPROVAL"

    # 1. VIEWER cannot approve
    _, err_v = orchestrator.approve_workflow_plan(wf.workflow_id, ws_id, approver_id="m_viewer")
    assert "lacks permission" in err_v

    # 2. REVIEWER cannot approve execution plan (only review findings/models)
    _, err_rev = orchestrator.approve_workflow_plan(wf.workflow_id, ws_id, approver_id="m_reviewer")
    assert "lacks permission" in err_rev

    # 3. Matrix generation blocked while pending approval
    _, err_mat = orchestrator.run_stage_experiment_matrix(wf.workflow_id, ws_id)
    assert "requires prior researcher approval" in err_mat

    # 4. Controlled execution blocked while pending approval
    _, err_exec = orchestrator.run_stage_controlled_execution(wf.workflow_id, ws_id, actor_id="m_researcher")
    assert "has not received researcher approval" in err_exec

    # 5. RESEARCHER successfully approves
    approved_wf, err_app = orchestrator.approve_workflow_plan(
        wf.workflow_id,
        ws_id,
        approver_id="m_researcher",
        approval_notes="Hypothesis and metrics validated against protocol.",
    )
    assert err_app is None
    assert approved_wf.approval_status == "APPROVED"
    assert approved_wf.approved_by == "m_researcher"
    assert approved_wf.status == "APPROVED"


def test_plan_rejection_and_cancellation(auth_workspace):
    store, ws_id = auth_workspace
    orchestrator = ResearchWorkflowOrchestrator(store)

    wf = orchestrator.initialize_workflow(ws_id, "Rejection Test Workflow")
    wf, _ = orchestrator.run_stage_experiment_plan(wf.workflow_id, ws_id)

    # Rejection by reviewer
    rejected_wf, err_rej = orchestrator.reject_workflow_plan(
        wf.workflow_id,
        ws_id,
        reviewer_id="m_reviewer",
        notes="Dataset split needs clarification before compute approval.",
    )
    assert err_rej is None
    assert rejected_wf.approval_status == "REJECTED"
    assert rejected_wf.status == "REJECTED"

    # Safe cancellation without destroying data
    cancelled_wf, err_canc = orchestrator.cancel_workflow(
        wf.workflow_id,
        ws_id,
        actor_id="m_owner",
        reason="Project direction redirected",
    )
    assert err_canc is None
    assert cancelled_wf.status == "CANCELLED"
    assert cancelled_wf.stage_data["cancellation"]["cancelled_by"] == "m_owner"
