"""Unit tests for ResearchPlan generation, validation, and multi-format exports."""

import json
import shutil
import tempfile

import pytest

from researchlens.workspace.planner import ResearchPlanner
from researchlens.workspace.schema import (
    EvidenceBookmark,
    ResearchDirection,
    ResearchHypothesis,
    ResearchPlan,
    ResearchQuestion,
)
from researchlens.workspace.store import WorkspaceStore


@pytest.fixture
def temp_ws_store():
    tmp_dir = tempfile.mkdtemp(prefix="rl_test_plan_")
    store = WorkspaceStore(base_dir=tmp_dir)
    yield store
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_generate_draft_plan_and_validation(temp_ws_store):
    store = temp_ws_store
    ws = store.create_workspace("Speech Synthesis")
    ws_id = ws.workspace_id

    rq = ResearchQuestion(
        question_id="rq_1",
        workspace_id=ws_id,
        text="Can discrete neural audio codecs improve low-resource voice cloning?",
        supporting_evidence_ids=["bm_1"],
    )
    hyp = ResearchHypothesis(
        hypothesis_id="hyp_1",
        workspace_id=ws_id,
        question_id="rq_1",
        statement="Residual vector quantization will lower phoneme error rate.",
        supporting_evidence=["bm_1"],
    )
    rd = ResearchDirection(
        direction_id="rd_1",
        workspace_id=ws_id,
        title="Codec-based Speech Infilling",
    )
    bm = EvidenceBookmark(
        bookmark_id="bm_1",
        workspace_id=ws_id,
        paper_id="doi:10.1001/codec2023",
        page_number=4,
        section_name="Results",
        source_text="Discrete codecs achieve 95% intelligibility at 3kbps.",
    )

    plan = ResearchPlanner.generate_draft_plan(
        workspace_id=ws_id,
        title="Neural Speech Synthesis Plan",
        problem_statement="Overcoming artifacts in low-bitrate neural audio synthesis.",
        questions=[rq],
        hypotheses=[hyp],
        directions=[rd],
        bookmarks=[bm],
        candidate_datasets=[{"name": "LibriSpeech", "source_paper": "Panayotov et al."}],
        candidate_baselines=[{"name": "SoundStream", "source_paper": "Zeghidour et al."}],
        candidate_metrics=["PESQ", "STOI", "WER"],
    )
    store.save_plan(plan)

    # 1. Validation test
    val = ResearchPlanner.validate_plan(plan)
    assert val["status"] == "COMPLETE"
    assert len(val["missing_components"]) == 0
    assert val["traceability_count"] >= 3

    # 2. Markdown export test
    md = ResearchPlanner.export_plan_to_markdown(plan)
    assert "# Research Plan: Neural Speech Synthesis Plan" in md
    assert "LibriSpeech" in md
    assert "PESQ, STOI, WER" in md
    assert "SoundStream" in md
    assert "Scientific Integrity Notice" in md

    # 3. JSON export test
    js = ResearchPlanner.export_plan_to_json(plan)
    data = json.loads(js)
    assert data["title"] == "Neural Speech Synthesis Plan"
    assert len(data["datasets"]) == 1


def test_incomplete_plan_validation():
    # Empty plan should flag missing components
    empty_plan = ResearchPlan(
        plan_id="rp_empty",
        workspace_id="ws_0",
        title="Empty Plan",
        problem_statement="",
        research_questions=[],
        datasets=[],
        metrics=[],
    )
    val = ResearchPlanner.validate_plan(empty_plan)
    assert val["status"] == "INCOMPLETE"
    assert "Problem statement is missing." in val["missing_components"]
    assert "At least one research question is required." in val["missing_components"]
    assert "No evaluation datasets identified." in val["missing_components"]
