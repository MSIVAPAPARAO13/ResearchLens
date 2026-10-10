"""Unit tests for ResearchFinding creation, negative results, and provenance links."""

from researchlens.workspace.experiment_service import ExperimentService
from researchlens.workspace.schema import ExperimentResult


def test_research_finding_creation_and_negative_results():
    # 1. Grounded positive finding
    res = [ExperimentResult("res_01", "run_01", "ModelX", "DatasetY", "f1", 0.92)]
    f_pos = ExperimentService.generate_research_finding(
        workspace_id="ws_01",
        title="ModelX Improvement on DatasetY",
        statement="ModelX achieved 0.92 F1 exceeding previous baselines.",
        finding_type="RESULT",
        related_hypothesis_id="hyp_01",
        experiment_results=res,
        evidence_ids=["bm_01"],
    )
    assert f_pos.confidence_level == "HIGH"
    assert f_pos.status == "RESEARCHER_REVIEW_REQUIRED"
    assert "res_01" in f_pos.experiment_result_ids
    assert "bm_01" in f_pos.evidence_ids

    # 2. Negative result (explicitly supported for scientific integrity)
    f_neg = ExperimentService.generate_research_finding(
        workspace_id="ws_01",
        title="ModelX Generalization Failure on Out-of-Domain Set",
        statement="Proposed architecture failed to outperform baseline on clinical notes split.",
        finding_type="NEGATIVE_RESULT",
        related_hypothesis_id="hyp_02",
        experiment_results=res,
        evidence_ids=[],
    )
    assert f_neg.finding_type == "NEGATIVE_RESULT"
    assert f_neg.status == "RESEARCHER_REVIEW_REQUIRED"

    # 3. Finding lacking evidence must be flagged as INSUFFICIENT_EVIDENCE
    f_empty = ExperimentService.generate_research_finding(
        workspace_id="ws_01",
        title="Unbacked Claim",
        statement="Claim with no data or evidence.",
        finding_type="OBSERVATION",
        experiment_results=[],
        evidence_ids=[],
    )
    assert f_empty.status == "INSUFFICIENT_EVIDENCE"
    assert f_empty.confidence_level == "LOW"
