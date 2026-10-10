"""Unit tests for hypothesis update workflows and human-in-the-loop validation."""

import shutil
import tempfile

import pytest

from researchlens.workspace.schema import (
    ExperimentResult,
    ResearchHypothesis,
    ResearchReview,
)
from researchlens.workspace.store import WorkspaceStore


@pytest.fixture
def temp_ws_store():
    tmp_dir = tempfile.mkdtemp(prefix="rl_test_hyp_up_")
    store = WorkspaceStore(base_dir=tmp_dir)
    yield store
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_hypothesis_update_workflow_requires_researcher_confirmation(temp_ws_store):
    store = temp_ws_store
    ws = store.create_workspace("Hypothesis Workflow")
    ws_id = ws.workspace_id

    hyp = ResearchHypothesis(
        hypothesis_id="hyp_eval_01",
        workspace_id=ws_id,
        statement="Sparse attention reduces memory overhead by 50% without loss of accuracy.",
        status="PROPOSED",
    )
    store.save_hypothesis(hyp)

    # Ingesting an experiment result does NOT automatically flip hypothesis to SUPPORTED
    res = ExperimentResult("res_mem_01", "run_01", "SparseTransformer", "LongBench", "memory_reduction", 0.52)
    store.save_result(res, ws_id)

    # The hypothesis MUST remain PROPOSED until researcher formally reviews
    stored_hyp = store.list_hypotheses(ws_id)[0]
    assert stored_hyp.status == "PROPOSED"
    assert stored_hyp.status != "SUPPORTED"

    # Researcher conducts review and explicitly confirms hypothesis
    review = ResearchReview(
        review_id="rev_01",
        target_type="HYPOTHESIS",
        target_id="hyp_eval_01",
        reviewer_id="mem_lead_scientist",
        decision="APPROVED",
        comments="Confirmed by 3 benchmark runs across multiple sequence lengths.",
    )
    store.save_review(review, ws_id)

    # Researcher manual update
    stored_hyp.status = "SUPPORTED"
    stored_hyp.supporting_evidence.append(res.result_id)
    store.save_hypothesis(stored_hyp)

    final_hyp = store.list_hypotheses(ws_id)[0]
    assert final_hyp.status == "SUPPORTED"
    assert "res_mem_01" in final_hyp.supporting_evidence
