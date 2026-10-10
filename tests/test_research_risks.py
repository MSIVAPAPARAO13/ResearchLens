"""Unit tests for ResearchRisk register and literature-grounded risk inference."""

import shutil
import tempfile

import pytest

from researchlens.workspace.planner import infer_risks_from_limitations
from researchlens.workspace.schema import ResearchRisk
from researchlens.workspace.store import WorkspaceStore


@pytest.fixture
def temp_ws_store():
    tmp_dir = tempfile.mkdtemp(prefix="rl_test_risk_")
    store = WorkspaceStore(base_dir=tmp_dir)
    yield store
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_infer_risks_from_literature_limitations(temp_ws_store):
    store = temp_ws_store
    ws = store.create_workspace("LLM Alignment")
    ws_id = ws.workspace_id

    mock_limitations = [
        {"text": "Evaluations are restricted to a single dataset (AlpacaEval) with limited domain diversity.", "paper_title": "Paper A"},
        {"text": "Training requires substantial GPU compute overhead and memory footprint.", "paper_title": "Paper B"},
        {"text": "Data leakage and benchmark contamination remains a severe risk in web-scraped pretraining.", "paper_title": "Paper C"},
    ]

    risks = infer_risks_from_limitations(ws_id, mock_limitations)
    assert len(risks) == 3

    categories = {r.category for r in risks}
    assert "GENERALIZATION" in categories
    assert "COMPUTE" in categories
    assert "DATA_LEAKAGE" in categories

    for r in risks:
        store.save_risk(r)
        assert len(r.evidence) > 0
        assert r.severity in ["LOW", "MEDIUM", "HIGH"]
        assert len(r.mitigation) > 0

    saved_risks = store.list_risks(ws_id)
    assert len(saved_risks) == 3


def test_custom_research_risk_crud(temp_ws_store):
    store = temp_ws_store
    ws = store.create_workspace("Medical AI")
    ws_id = ws.workspace_id

    risk = ResearchRisk(
        risk_id="risk_bias_01",
        workspace_id=ws_id,
        category="BIAS",
        description="Dataset lacks demographic stratification across age groups.",
        evidence=["Reported in cohort metadata"],
        severity="HIGH",
        mitigation="Collect stratified external validation cohort.",
    )
    store.save_risk(risk)
    assert len(store.list_risks(ws_id)) == 1

    assert store.delete_risk(ws_id, "risk_bias_01") is True
    assert len(store.list_risks(ws_id)) == 0
