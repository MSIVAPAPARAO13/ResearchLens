"""Traceability and scientific integrity unit tests for research plans and hypotheses."""

from researchlens.workspace.planner import ResearchPlanner
from researchlens.workspace.schema import (
    EvidenceBookmark,
    ResearchHypothesis,
    ResearchPlan,
    ResearchQuestion,
)


def test_literature_to_plan_traceability_chain():
    # Construct complete traceability chain: Plan -> Question -> Gap/Evidence -> Paper -> Excerpt
    bm = EvidenceBookmark(
        bookmark_id="bm_provenance_01",
        workspace_id="ws_trace_01",
        paper_id="doi:10.1001/transformer_bert",
        page_number=7,
        section_name="Experiments",
        chunk_id="chk_exp_4",
        source_text="BERT achieves 90.9% F1 score on SQuAD v1.1.",
        label="SQuAD Benchmark Result",
    )

    rq = ResearchQuestion(
        question_id="rq_trace_01",
        workspace_id="ws_trace_01",
        text="Can bidirectional self-attention generalize to technical domain question answering?",
        type="PRIMARY",
        supporting_evidence_ids=["bm_provenance_01"],
        supporting_gap_ids=["gap_domain_shift_01"],
    )

    hyp = ResearchHypothesis(
        hypothesis_id="hyp_trace_01",
        workspace_id="ws_trace_01",
        question_id="rq_trace_01",
        statement="Bidirectional contextual representations will achieve higher F1 than unidirectional baselines on technical QA.",
        independent_variables=["Contextual attention directionality"],
        dependent_variables=["Technical QA F1 score"],
        supporting_evidence=["bm_provenance_01"],
        status="PROPOSED",
    )

    datasets = [{"name": "SQuAD v1.1", "source_paper": "Rajpurkar et al. (2016)", "provenance": "Page 3, Section: Data"}]
    baselines = [{"name": "BiDAF", "source_paper": "Seo et al. (2017)", "provenance": "Page 5, Section: Baseline"}]
    metrics = ["F1", "Exact Match"]

    plan = ResearchPlanner.generate_draft_plan(
        workspace_id="ws_trace_01",
        title="Contextual Representation for Technical QA",
        problem_statement="Domain shift degradation in open-domain question answering.",
        questions=[rq],
        hypotheses=[hyp],
        directions=[],
        bookmarks=[bm],
        candidate_datasets=datasets,
        candidate_baselines=baselines,
        candidate_metrics=metrics,
    )

    # Verify each component in traceability array
    trace_types = {t["type"] for t in plan.traceability}
    assert "QUESTION" in trace_types
    assert "HYPOTHESIS" in trace_types
    assert "DATASET" in trace_types
    assert "BASELINE" in trace_types
    assert "METRIC" in trace_types
    assert "EVIDENCE" in trace_types

    # Find the evidence trace
    ev_trace = next(t for t in plan.traceability if t["type"] == "EVIDENCE")
    assert ev_trace["paper_id"] == "doi:10.1001/transformer_bert"
    assert ev_trace["page"] == 7
    assert ev_trace["section"] == "Experiments"
    assert "90.9% F1" in ev_trace["source_text"]


def test_scientific_integrity_guarantees():
    """Verify that ResearchLens strictly enforces scientific integrity constraints."""
    # 1. AI cannot set hypothesis status to SUPPORTED on creation
    hyp = ResearchHypothesis(
        hypothesis_id="hyp_int_01",
        workspace_id="ws_int",
        statement="Unvalidated hypothesis",
    )
    assert hyp.status != "SUPPORTED"
    assert hyp.status in ["DRAFT", "PROPOSED"]

    # 2. Plan export contains mandatory scientific integrity notice
    empty_plan = ResearchPlan(
        plan_id="rp_int",
        workspace_id="ws_int",
        title="Integrity Test Plan",
    )
    md = ResearchPlanner.export_plan_to_markdown(empty_plan)
    assert "Scientific Integrity Notice" in md
    assert "do not constitute a peer-reviewed experimental protocol" in md

    # 3. Plan validation flags absence of evidence/traceability
    val = ResearchPlanner.validate_plan(empty_plan)
    assert "No evidence traceability citations attached." in val["warnings"]
