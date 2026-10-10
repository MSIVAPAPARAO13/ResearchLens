"""Unit tests for evidence-grounded research proposals (Phase 15)."""

import pytest

from researchlens.automation.proposer import (
    SCIENTIFIC_INTEGRITY_NOTICE,
    EvidenceGroundedProposer,
    GroundedHypothesisProposal,
)
from researchlens.workspace.schema import (
    EvidenceBookmark,
    ResearchGap,
    ResearchQuestion,
)


@pytest.fixture
def sample_evidence():
    return [
        EvidenceBookmark(
            bookmark_id="bm_01",
            workspace_id="ws_prop_test",
            paper_id="paper_retrieval_01",
            label="Hybrid Search Study",
            source_text="Dense retrieval exhibits sharp performance degradation on out-of-domain biomedical queries.",
            page_number=6,
            section_name="Discussion",
            chunk_id="chunk_01",
        ),
        EvidenceBookmark(
            bookmark_id="bm_02",
            workspace_id="ws_prop_test",
            paper_id="paper_retrieval_02",
            label="Reranking Benchmark",
            source_text="Cross-encoder reranking mitigates false positives but introduces latency bottlenecks.",
            page_number=9,
            section_name="Limitations",
            chunk_id="chunk_02",
        ),
    ]


def test_formulate_questions_from_gaps(sample_evidence):
    proposer = EvidenceGroundedProposer("ws_prop_test")
    gaps = [
        ResearchGap(
            gap_id="gap_01",
            workspace_id="ws_prop_test",
            category="GENERALIZATION",
            description="Out-of-domain retrieval degradation in dense models.",
            paper_ids=["paper_retrieval_01"],
            verified=True,
        )
    ]

    questions = proposer.formulate_questions_from_gaps(gaps=gaps, bookmarks=sample_evidence)
    assert len(questions) == 1
    q = questions[0]
    assert isinstance(q, ResearchQuestion)
    assert "Out-of-domain retrieval degradation" in q.text
    assert q.supporting_gap_ids == ["gap_01"]
    assert "bm_01" in q.supporting_evidence_ids


def test_propose_hypothesis_structure_and_integrity(sample_evidence):
    proposer = EvidenceGroundedProposer("ws_prop_test")
    q = ResearchQuestion(
        question_id="q_test_01",
        workspace_id="ws_prop_test",
        text="How can we address out-of-domain degradation?",
    )

    prop = proposer.propose_hypothesis(
        question=q,
        bookmarks=sample_evidence,
        target_dataset="PubMedQA",
        target_metric="MRR",
        target_baseline="BM25-Baseline",
    )

    assert isinstance(prop, GroundedHypothesisProposal)
    assert prop.testable is True
    assert "Hybrid Dense-BM25" in prop.hypothesis_statement
    assert "PubMedQA" in prop.hypothesis_statement

    # Evidence quotations check
    assert len(prop.supporting_evidence) >= 2
    eq1 = prop.supporting_evidence[0]
    assert eq1["page_number"] == 6
    assert eq1["section_name"] == "Discussion"
    assert "Dense retrieval exhibits" in eq1["quote_text"]

    # Explicit assumptions and missing evidence check
    assert len(prop.explicit_assumptions) >= 2
    assert len(prop.missing_evidence) >= 1
    assert "MRR" in prop.measurable_criteria

    # Scientific integrity notice check
    assert SCIENTIFIC_INTEGRITY_NOTICE == prop.scientific_integrity_notice
    assert "NOT established scientific facts" in prop.scientific_integrity_notice


def test_convert_to_persistent_hypothesis(sample_evidence):
    proposer = EvidenceGroundedProposer("ws_prop_test")
    q = ResearchQuestion(question_id="q_01", workspace_id="ws_prop_test", text="Test Question")
    prop = proposer.propose_hypothesis(question=q, bookmarks=sample_evidence)

    hyp = proposer.convert_to_hypothesis(prop, question_id="q_01")
    assert hyp.status == "DRAFT"
    assert hyp.testable is True
    assert hyp.question_id == "q_01"
    assert any("p.6" in ev for ev in hyp.supporting_evidence)
