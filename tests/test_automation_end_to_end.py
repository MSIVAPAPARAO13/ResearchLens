"""End-to-End integration test for Phase 15 research automation workflow (Stages 1 to 13)."""

import os
import shutil
import tempfile

import pytest

from researchlens.automation.workflow import ResearchWorkflowOrchestrator
from researchlens.execution.registry import ExperimentRegistryService
from researchlens.workspace.schema import (
    EvidenceBookmark,
    ResearchGap,
    SavedPaper,
    WorkspaceMember,
)
from researchlens.workspace.store import WorkspaceStore


@pytest.fixture
def populated_e2e_workspace():
    temp_dir = tempfile.mkdtemp(prefix="rl_test_e2e_auto_")
    store = WorkspaceStore(base_dir=os.path.join(temp_dir, "workspaces"))
    ws = store.create_workspace("E2E Automation Workspace")
    ws_id = ws.workspace_id

    # Add workspace members
    store.save_member(WorkspaceMember("m_owner", ws_id, "Dr. Alice", "alice@lab.org", "OWNER"))
    store.save_member(WorkspaceMember("m_researcher", ws_id, "Bob", "bob@lab.org", "RESEARCHER"))
    store.save_member(WorkspaceMember("m_reviewer", ws_id, "Carol", "carol@lab.org", "REVIEWER"))

    # Register deterministic experiment script
    registry = ExperimentRegistryService(store)
    entrypoint = "scripts/experiments/deterministic_demo.py"
    registry.register_experiment(
        workspace_id=ws_id,
        name="Deterministic Benchmark Run",
        entrypoint=entrypoint,
        owner="m_owner",
    )

    # Ingest a saved paper
    paper = SavedPaper(
        paper_id="paper_e2e_01",
        workspace_id=ws_id,
        title="Deep Retrieval Models on Scientific Question Answering",
        authors=["Vaswani et al."],
        year=2023,
    )
    store.save_paper(paper)

    # Bookmark evidence
    bm = EvidenceBookmark(
        bookmark_id="bm_e2e_01",
        workspace_id=ws_id,
        paper_id=paper.paper_id,
        label="Biomedical Retrieval Gap",
        source_text="Dense embeddings alone struggle with precise numerical metrics and acronyms in PubMedQA.",
        page_number=5,
        section_name="Discussion",
    )
    store.save_evidence(bm)

    # Verified gap
    gap = ResearchGap(
        gap_id="gap_e2e_01",
        workspace_id=ws_id,
        category="METHODOLOGY",
        description="Dense embeddings struggle with exact lexical tokens in scientific Q&A.",
        paper_ids=[paper.paper_id],
        verified=True,
    )
    store.save_gap(gap)

    yield store, ws_id
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_full_13_stage_guided_research_workflow(populated_e2e_workspace):
    store, ws_id = populated_e2e_workspace
    orchestrator = ResearchWorkflowOrchestrator(store)

    # 1. Initialize and run Stages 1-6 up to approval gate
    wf, err = orchestrator.execute_guided_workflow_to_approval(
        workspace_id=ws_id,
        title="PubMedQA Hybrid Evaluation Workflow",
        topic="Biomedical Literature QA",
        actor_id="m_researcher",
    )
    assert err is None
    assert wf is not None
    assert wf.current_stage == "RESEARCHER_APPROVAL"
    assert wf.status == "AWAITING_APPROVAL"
    assert wf.approval_status == "PENDING"

    # Verify Stage 1: Literature Discovery
    assert wf.stage_data["literature"]["paper_count"] >= 1

    # Verify Stage 2: Evidence Collection
    assert wf.stage_data["evidence"]["evidence_count"] >= 1

    # Verify Stage 3: Gap Validation
    assert wf.stage_data["gaps"]["gap_count"] >= 1
    assert wf.stage_data["gaps"]["validated_gaps"][0]["status"] == "CONFIRMED_OPEN_GAP"

    # Verify Stage 4: Question Formulation
    assert wf.stage_data["questions"]["question_count"] >= 1

    # Verify Stage 5: Hypothesis Proposal
    assert "proposal" in wf.stage_data["hypothesis"]
    assert wf.stage_data["hypothesis"]["proposal"]["testable"] is True

    # Verify Stage 6: Experiment Plan
    plan_info = wf.stage_data["experiment_plan"]
    assert plan_info["status"] == "PLANNED"
    assert plan_info["dataset"] == "PubMedQA"

    # 2. Stage 7: Researcher Approval Gate
    approved_wf, app_err = orchestrator.approve_workflow_plan(
        workflow_id=wf.workflow_id,
        workspace_id=ws_id,
        approver_id="m_researcher",
        approval_notes="Grounded in Vaswani et al. 2023 evidence bookmark.",
    )
    assert app_err is None
    assert approved_wf.approval_status == "APPROVED"
    assert approved_wf.status == "APPROVED"

    # 3. Stages 8–12: Post-Approval Execution Pipeline
    exec_wf, post_err = orchestrator.execute_post_approval_pipeline(
        workflow_id=wf.workflow_id,
        workspace_id=ws_id,
        actor_id="m_researcher",
    )
    assert post_err is None
    assert exec_wf is not None
    assert exec_wf.current_stage == "HUMAN_REVIEW"

    # Verify Stage 8: Matrix
    assert exec_wf.stage_data["matrix"]["planned_run_count"] <= 20

    # Verify Stage 9: Controlled Execution
    exec_info = exec_wf.stage_data["execution"]
    assert exec_info["status"] == "COMPLETED"
    assert exec_info["run_id"] != ""

    # Verify Stage 10: Evaluation
    eval_info = exec_wf.stage_data["evaluation"]
    assert eval_info["status"] == "COMPLETED"
    assert eval_info["scientific_status"] == "EVALUATED_NOT_CONCLUDED"

    # Verify Stage 11: Scientific Comparability
    comp_info = exec_wf.stage_data["comparability"]
    assert comp_info["status"] == "COMPLETED"
    assert comp_info["statistical_significance_tested"] is False

    # Verify Stage 12: Finding Candidate
    fc_info = exec_wf.stage_data["finding_candidate"]
    assert fc_info["status"] == "PENDING_RESEARCHER_REVIEW"

    # 4. Stage 13: Human Review
    final_wf, rev_err = orchestrator.review_workflow_finding(
        workflow_id=wf.workflow_id,
        workspace_id=ws_id,
        reviewer_id="m_reviewer",
        decision="CONFIRMED",
        review_notes="Empirical metrics verified; findings confirmed for publication.",
    )
    assert rev_err is None
    assert final_wf.status == "COMPLETED"
    assert final_wf.stage_data["human_review"]["decision"] == "CONFIRMED"

    # Check Audit events logged
    audits = store.list_audit(ws_id)
    actions = [a.action for a in audits]
    assert "WORKFLOW_INITIALIZED" in actions
    assert "WORKFLOW_APPROVED" in actions
    assert "WORKFLOW_REVIEW_COMPLETED" in actions
