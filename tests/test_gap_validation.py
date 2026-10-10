"""Unit tests for LiteratureGapValidator and external gap validation."""

from researchlens.analysis.external_provider import ExternalPaper
from researchlens.analysis.gap_validator import LiteratureGapValidator


def test_gap_validation_empty_inputs():
    validator = LiteratureGapValidator()
    res = validator.validate_gap("", [])
    assert res.status == "UNVERIFIED"
    assert res.confidence_score == 0.0


def test_gap_validation_confirmed_open_gap():
    validator = LiteratureGapValidator(similarity_threshold=0.20)
    gap = "Quadratic computational memory complexity during long-sequence attention."

    paper1 = ExternalPaper(
        "p1",
        "Challenges in Long-Context Transformers",
        abstract="Quadratic attention memory complexity remains a critical bottleneck and unresolved challenge in modern architectures.",
        year=2024,
    )
    paper2 = ExternalPaper(
        "p2",
        "Scaling Limits for Neural Attention",
        abstract="Long-sequence training is constrained by GPU memory consumption and quadratic compute scaling.",
        year=2023,
    )

    res = validator.validate_gap(gap, [paper1, paper2])
    assert res.status == "CONFIRMED_OPEN_GAP"
    assert res.confidence_score > 0.40
    assert len(res.relevant_external_papers) >= 1
    assert "Confirmed" in res.rationale


def test_gap_validation_potentially_addressed():
    validator = LiteratureGapValidator(similarity_threshold=0.20)
    gap = "Quadratic computational memory complexity during long-sequence attention."

    paper_sol = ExternalPaper(
        "p_sol",
        "FlashAttention-3: Ultra-Fast Linear Attention",
        abstract="In this work, we solve quadratic memory complexity and overcome the limitation with hardware-aware tiling algorithms.",
        year=2024,
    )

    res = validator.validate_gap(gap, [paper_sol])
    assert res.status == "POTENTIALLY_ADDRESSED"
    assert len(res.suggested_directions) >= 1
    assert "actively addressed" in res.rationale
