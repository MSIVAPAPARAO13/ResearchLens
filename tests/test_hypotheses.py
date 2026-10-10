"""Unit tests for ResearchHypothesis formulation and scientific integrity constraints."""

import shutil
import tempfile

import pytest

from researchlens.workspace.schema import ResearchHypothesis
from researchlens.workspace.store import WorkspaceStore


@pytest.fixture
def temp_ws_store():
    tmp_dir = tempfile.mkdtemp(prefix="rl_test_hyp_")
    store = WorkspaceStore(base_dir=tmp_dir)
    yield store
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_research_hypothesis_lifecycle_and_integrity(temp_ws_store):
    store = temp_ws_store
    ws = store.create_workspace("Reasoning Models")
    ws_id = ws.workspace_id

    hyp = ResearchHypothesis(
        hypothesis_id="hyp_001",
        workspace_id=ws_id,
        question_id="rq_001",
        statement="Incorporating tree-based search improves multi-step deduction accuracy over linear chain-of-thought.",
        independent_variables=["Search topology (tree vs chain)"],
        dependent_variables=["Multi-step deduction accuracy"],
        expected_relationship="Positive improvement in solve rate",
        supporting_evidence=["bm_tot_paper_01"],
        status="PROPOSED",
    )
    store.save_hypothesis(hyp)

    saved = store.list_hypotheses(ws_id)
    assert len(saved) == 1
    h = saved[0]
    # Scientific Integrity Check: Newly formulated hypothesis must NOT default to SUPPORTED
    assert h.status != "SUPPORTED"
    assert h.status in ["DRAFT", "PROPOSED"]
    assert h.question_id == "rq_001"
    assert len(h.independent_variables) == 1
    assert len(h.dependent_variables) == 1

    # User explicitly sets TESTED / SUPPORTED after experiment
    h.status = "SUPPORTED"
    store.save_hypothesis(h)
    assert store.list_hypotheses(ws_id)[0].status == "SUPPORTED"

    assert store.delete_hypothesis(ws_id, "hyp_001") is True
    assert len(store.list_hypotheses(ws_id)) == 0
