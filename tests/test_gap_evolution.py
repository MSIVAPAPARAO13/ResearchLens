"""Unit tests for research gap evolution tracking and lifecycle classification."""

from researchlens.analysis.external_provider import ExternalPaper
from researchlens.analysis.gap_evolution import (
    GapEvolutionEngine,
    ResearchGapEvolutionRecord,
    classify_gap_evolution,
    compare_gap_sets,
)
from researchlens.analysis.gap_validator import GapValidationResult
from researchlens.analysis.research_gap_schema import ResearchGap


def test_classify_gap_evolution_lifecycle():
    # 1. Brand new gap
    assert classify_gap_evolution(None, ["Paper A"]) == "NEW"

    # 2. Strengthening gap (supporting papers grow)
    rec = ResearchGapEvolutionRecord(
        gap_id="g1",
        gap_statement="High GPU memory footprint during attention computation",
        category="Methodological",
        first_detected="2026-10-01T00:00:00Z",
        last_detected="2026-10-01T00:00:00Z",
        status="NEW",
        supporting_papers=["Paper A"],
    )
    assert classify_gap_evolution(rec, ["Paper B", "Paper C"]) == "STRENGTHENING"

    # 3. Potentially addressed gap
    val_res = GapValidationResult(
        gap_statement="High GPU memory footprint during attention computation",
        status="POTENTIALLY_ADDRESSED",
        confidence_score=0.85,
        rationale="FlashAttention optimizes GPU memory footprint.",
        relevant_external_papers=[ExternalPaper("p_sol", "FlashAttention: Fast Memory-Efficient Attention")],
    )
    assert classify_gap_evolution(rec, ["Paper B"], validation_result=val_res) == "POTENTIALLY_ADDRESSED"

    # 4. Persistent gap (no count change)
    assert classify_gap_evolution(rec, ["Paper A"]) == "PERSISTENT"


def test_compare_gap_sets():
    prev_record = ResearchGapEvolutionRecord(
        gap_id="g1",
        gap_statement="Lack of cross-dataset generalization in biomedical imaging",
        category="Empirical",
        first_detected="2026-10-01T00:00:00Z",
        last_detected="2026-10-01T00:00:00Z",
        status="NEW",
        evidence_snippets=["Limited to single hospital dataset."],
    )

    current_gaps = [
        # Match for g1 with more evidence
        ResearchGap(
            gap_id="g1",
            title="Generalization Gap",
            category="Empirical",
            gap_type="EXPLICIT_LIMITATION",
            statement="Lack of cross-dataset generalization in biomedical imaging",
            confidence_score=0.8,
            rationale="Limited to single hospital.",
        ),
        # New gap
        ResearchGap(
            gap_id="g2",
            title="Complexity Gap",
            category="Algorithmic",
            gap_type="EXPLICIT_LIMITATION",
            statement="Quadratic computational complexity of rotary position embeddings",
            confidence_score=0.7,
            rationale="O(N^2) memory consumption observed.",
        ),
    ]

    diff = compare_gap_sets([prev_record], current_gaps)
    assert len(diff["new_gaps"]) == 1
    assert diff["new_gaps"][0].gap_id == "g2"
    assert len(diff["strengthened_gaps"]) == 0
    assert len(diff["persistent_gaps"]) == 1


def test_gap_evolution_engine_tracking():
    engine = GapEvolutionEngine()
    history = {}

    gaps_run1 = [
        ResearchGap("g1", "Trial Size", "Empirical", "EXPLICIT_LIMITATION", "Small sample size in clinical trial", confidence_score=0.7)
    ]
    papers_run1 = [ExternalPaper("p1", "Clinical Study", year=2024)]

    # Run 1
    history = engine.track_gap_history(history, gaps_run1, papers_run1, paper_title="Run 1")
    assert len(history) == 1
    assert history["g1"].status == "NEW"
    assert history["g1"].observation_count == 1

    # Run 2: Same gap observed again with additional supporting literature
    gaps_run2 = [
        ResearchGap("g1", "Trial Size", "Empirical", "EXPLICIT_LIMITATION", "Small sample size in clinical trial", confidence_score=0.85)
    ]
    papers_run2 = [ExternalPaper("p2", "Second Clinical Trial", year=2025)]

    history = engine.track_gap_history(history, gaps_run2, papers_run2, paper_title="Run 2")
    assert len(history) == 1
    assert history["g1"].status in ("STRENGTHENING", "PERSISTENT")
    assert history["g1"].observation_count == 2
    assert "Run 2" in history["g1"].supporting_papers
