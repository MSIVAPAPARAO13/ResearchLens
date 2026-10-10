"""Unit tests for ResearchDirection and transparent heuristic prioritization."""

import shutil
import tempfile

import pytest

from researchlens.workspace.planner import calculate_direction_priority_score
from researchlens.workspace.schema import ResearchDirection
from researchlens.workspace.store import WorkspaceStore


@pytest.fixture
def temp_ws_store():
    tmp_dir = tempfile.mkdtemp(prefix="rl_test_rd_")
    store = WorkspaceStore(base_dir=tmp_dir)
    yield store
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_calculate_direction_priority_score():
    score, breakdown = calculate_direction_priority_score(
        evidence_strength=1.0,
        research_gap_strength=1.0,
        feasibility=0.8,
        impact=0.9,
        data_availability=0.7,
        evaluation_feasibility=0.8,
    )
    # Expected: 0.25*1.0 + 0.20*1.0 + 0.15*0.8 + 0.15*0.9 + 0.15*0.7 + 0.10*0.8 = 0.25 + 0.20 + 0.12 + 0.135 + 0.105 + 0.08 = 0.89
    assert 0.88 <= score <= 0.90
    assert "evidence_strength" in breakdown
    assert "total_score" in breakdown
    assert breakdown["total_score"] == score


def test_research_direction_crud(temp_ws_store):
    store = temp_ws_store
    ws = store.create_workspace("Clinical NLP")
    ws_id = ws.workspace_id

    score, breakdown = calculate_direction_priority_score(0.9, 0.8, 0.7, 0.85, 0.8, 0.75)
    rd = ResearchDirection(
        direction_id="rd_001",
        workspace_id=ws_id,
        title="Cross-Institutional EHR Phenotyping",
        description="Developing domain-invariant patient representations",
        source_gap_ids=["gap_ehr_domain_shift"],
        priority_score=score,
        score_breakdown=breakdown,
        status="SHORTLISTED",
    )
    store.save_direction(rd)

    dirs = store.list_directions(ws_id)
    assert len(dirs) == 1
    assert dirs[0].status == "SHORTLISTED"
    assert dirs[0].priority_score == score

    assert store.delete_direction(ws_id, "rd_001") is True
    assert len(store.list_directions(ws_id)) == 0
