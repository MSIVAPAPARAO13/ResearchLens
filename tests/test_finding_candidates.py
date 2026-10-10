"""Unit tests for finding candidate generation and human review (Phase 15)."""

import os
import shutil
import tempfile

import pytest

from researchlens.automation.finding_candidate import FindingCandidateGenerator
from researchlens.automation.workflow import ResearchWorkflowOrchestrator
from researchlens.workspace.schema import (
    ExperimentResult,
    ExperimentRun,
    ResearchFinding,
    WorkspaceMember,
)
from researchlens.workspace.store import WorkspaceStore


@pytest.fixture
def finding_workspace():
    temp_dir = tempfile.mkdtemp(prefix="rl_test_fnd_")
    store = WorkspaceStore(base_dir=os.path.join(temp_dir, "workspaces"))
    ws = store.create_workspace("Finding Candidates Workspace")
    ws_id = ws.workspace_id

    store.save_member(WorkspaceMember("m_owner", ws_id, "Alice", "alice@lab.org", "OWNER"))
    store.save_member(WorkspaceMember("m_reviewer", ws_id, "Carol", "carol@lab.org", "REVIEWER"))
    store.save_member(WorkspaceMember("m_viewer", ws_id, "David", "david@lab.org", "VIEWER"))

    yield store, ws_id
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_finding_candidate_generation_and_provenance(finding_workspace):
    store, ws_id = finding_workspace
    generator = FindingCandidateGenerator(ws_id)

    run = ExperimentRun(
        run_id="run_fnd_01",
        experiment_id="exp_01",
        status="COMPLETED",
        git_commit_sha="c123456",
        mlflow_run_id="mlflow_run_01",
    )
    store.save_run(run, ws_id)

    results = [
        ExperimentResult(
            result_id="res_01",
            run_id=run.run_id,
            model="Hybrid-MiniLM",
            dataset="PubMedQA",
            metric="MRR",
            value=0.785,
        ),
        ExperimentResult(
            result_id="res_02",
            run_id=run.run_id,
            model="Hybrid-MiniLM",
            dataset="PubMedQA",
            metric="precision",
            value=0.820,
        ),
    ]

    baseline_results = [
        ExperimentResult(
            result_id="res_base_01",
            run_id="run_base",
            model="BM25-Baseline",
            dataset="PubMedQA",
            metric="MRR",
            value=0.690,
        ),
        ExperimentResult(
            result_id="res_base_02",
            run_id="run_base",
            model="BM25-Baseline",
            dataset="PubMedQA",
            metric="precision",
            value=0.740,
        ),
    ]

    finding = generator.generate_candidate(
        experiment_id="exp_01",
        run=run,
        results=results,
        baseline_results=baseline_results,
        reproducibility_status="REPRODUCED_MATCH",
    )

    assert isinstance(finding, ResearchFinding)
    assert finding.status == "PENDING_RESEARCHER_REVIEW"
    assert finding.reproducibility_status == "REPRODUCED_MATCH"
    assert finding.mlflow_run_id == "mlflow_run_01"

    # Baseline comparison verification
    assert "MRR" in finding.baseline_comparison
    mrr_comp = finding.baseline_comparison["MRR"]
    assert mrr_comp["proposed_value"] == 0.785
    assert mrr_comp["baseline_value"] == 0.690
    assert mrr_comp["delta"] == 0.095
    assert mrr_comp["status"] == "IMPROVED"

    # Integrity & limitations
    assert len(finding.limitations) >= 2
    assert "SCIENTIFIC_INTEGRITY_DISCLAIMER" in finding.uncertainty_notes


def test_human_review_workflow_and_authorization(finding_workspace):
    store, ws_id = finding_workspace
    orchestrator = ResearchWorkflowOrchestrator(store)

    wf = orchestrator.initialize_workflow(ws_id, "Review Test Workflow")
    # Simulate finding candidate ready in workflow
    finding = ResearchFinding(
        finding_id="fnd_test_01",
        workspace_id=ws_id,
        title="Candidate Finding",
        statement="Proposed model achieves 0.785 MRR on PubMedQA.",
        status="PENDING_RESEARCHER_REVIEW",
    )
    store.save_finding(finding)
    wf.stage_data["finding_candidate"] = {
        "finding_id": finding.finding_id,
        "title": finding.title,
    }
    store.save_workflow(wf, ws_id)

    # 1. VIEWER cannot review
    _, err_v = orchestrator.review_workflow_finding(
        wf.workflow_id,
        ws_id,
        reviewer_id="m_viewer",
        decision="CONFIRMED",
    )
    assert "lacks permission" in err_v

    # 2. REVIEWER successfully reviews and confirms
    reviewed_wf, err_rev = orchestrator.review_workflow_finding(
        wf.workflow_id,
        ws_id,
        reviewer_id="m_reviewer",
        decision="CONFIRMED",
        review_notes="Results verified against holdout protocol.",
    )
    assert err_rev is None
    assert reviewed_wf.status == "COMPLETED"
    assert reviewed_wf.stage_data["human_review"]["decision"] == "CONFIRMED"
    assert reviewed_wf.stage_data["human_review"]["reviewer_id"] == "m_reviewer"

    # Check finding status updated
    updated_finding = store.list_findings(ws_id)[0]
    assert updated_finding.status == "CONFIRMED"
