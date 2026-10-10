"""Unit tests for ResearchQuestion formulation and gap/evidence linking."""

import shutil
import tempfile

import pytest

from researchlens.workspace.schema import ResearchQuestion
from researchlens.workspace.store import WorkspaceStore


@pytest.fixture
def temp_ws_store():
    tmp_dir = tempfile.mkdtemp(prefix="rl_test_rq_")
    store = WorkspaceStore(base_dir=tmp_dir)
    yield store
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_research_question_linking(temp_ws_store):
    store = temp_ws_store
    ws = store.create_workspace("RAG Hallucination")
    ws_id = ws.workspace_id

    rq = ResearchQuestion(
        question_id="rq_001",
        workspace_id=ws_id,
        text="Does hybrid retrieval reduce factual hallucination rate compared to dense-only retrieval?",
        type="PRIMARY",
        priority="HIGH",
        supporting_gap_ids=["gap_rag_eval_01"],
        supporting_evidence_ids=["bm_dense_failure_02"],
    )
    store.save_question(rq)

    questions = store.list_questions(ws_id)
    assert len(questions) == 1
    q = questions[0]
    assert q.type == "PRIMARY"
    assert q.priority == "HIGH"
    assert q.supporting_gap_ids == ["gap_rag_eval_01"]
    assert q.supporting_evidence_ids == ["bm_dense_failure_02"]

    # Update question
    q.status = "ANSWERED"
    store.save_question(q)
    assert store.list_questions(ws_id)[0].status == "ANSWERED"

    assert store.delete_question(ws_id, "rq_001") is True
    assert len(store.list_questions(ws_id)) == 0
