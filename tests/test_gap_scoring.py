"""Unit tests for transparent heuristic gap scoring and confidence calibration."""

from researchlens.analysis.gap_scorer import calculate_gap_score
from researchlens.analysis.research_gap_schema import ResearchGap
from researchlens.extraction.schema import ExtractedItem


def test_calculate_gap_score_explicit_high_evidence():
    ev = ExtractedItem(value="Memory footprint is quadratic.", page_number=5, section_name="discussion")
    gap = ResearchGap(
        gap_id="g1",
        title="Quadratic Memory",
        category="EFFICIENCY",
        gap_type="EXPLICIT_LIMITATION",
        statement="Memory footprint scales quadratically.",
        supporting_evidence=[ev],
        explicit_or_inferred="EXPLICIT",
        affected_method="Transformer",
        affected_dataset="ImageNet",
        potential_direction="Implement sparse attention to achieve linear complexity.",
    )

    score, level = calculate_gap_score(gap)
    assert 0.75 <= score <= 1.0
    assert level == "High"


def test_calculate_gap_score_inferred_medium_evidence():
    ev = ExtractedItem(value="Evaluated on single dataset SQuAD.", page_number=4, section_name="experiments")
    gap = ResearchGap(
        gap_id="g2",
        title="Single Dataset",
        category="GENERALIZATION",
        gap_type="UNDEREXPLORED_DATASET",
        statement="Single dataset evaluation limits generalization.",
        supporting_evidence=[ev],
        explicit_or_inferred="INFERRED",
        potential_direction="Evaluate on multiple datasets.",
    )

    score, level = calculate_gap_score(gap)
    assert 0.55 <= score < 0.85
    assert level in ("Medium", "High")


def test_calculate_gap_score_low_evidence():
    gap = ResearchGap(
        gap_id="g3",
        title="Weak Inference Gap",
        category="OTHER",
        gap_type="INFERRED",
        statement="Short stmt.",
        supporting_evidence=[],
        explicit_or_inferred="INFERRED",
        potential_direction="",
    )

    score, level = calculate_gap_score(gap)
    assert score < 0.55
    assert level == "Low"
