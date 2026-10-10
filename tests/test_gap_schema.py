"""Unit tests for ResearchGap and ResearchHypothesis schemas."""

from researchlens.analysis.research_gap_schema import ResearchGap, ResearchHypothesis
from researchlens.extraction.schema import ExtractedItem


def test_research_gap_dataclass():
    item = ExtractedItem(value="Memory footprint is quadratic.", page_number=5, section_name="discussion", chunk_id="c_5")
    gap = ResearchGap(
        gap_id="gap_01",
        title="Quadratic Memory Constraint",
        category="EFFICIENCY",
        gap_type="EXPLICIT_LIMITATION",
        statement="Memory consumption scales quadratically with sequence length.",
        supporting_evidence=[item],
        source_pages=[5],
        source_sections=["discussion"],
        source_chunks=["c_5"],
        explicit_or_inferred="EXPLICIT",
        confidence_level="High",
        confidence_score=0.88,
        rationale="Explicitly noted by authors.",
        potential_direction="Investigate sparse attention linear complexity mechanisms.",
        affected_method="Transformer",
        severity="High",
    )

    d = gap.to_dict()
    assert d["gap_id"] == "gap_01"
    assert d["category"] == "EFFICIENCY"
    assert d["explicit_or_inferred"] == "EXPLICIT"
    assert d["confidence_score"] == 0.88
    assert d["supporting_evidence"][0]["page_number"] == 5


def test_research_hypothesis_dataclass():
    item = ExtractedItem(value="Evaluated only on English text.", page_number=7, section_name="conclusion")
    hyp = ResearchHypothesis(
        hypothesis_id="hyp_01",
        title="Multilingual Cross-Domain Transfer",
        statement="Pre-training on multilingual parallel text improves zero-shot transfer.",
        motivation="Single-language evaluation limits global applicability.",
        supporting_evidence=[item],
        variables={"independent_variable": "Multilingual pre-training", "dependent_variable": "Cross-lingual F1"},
        suggested_evaluation="Evaluate across 10 languages on XNLI dataset.",
        confidence_level="Medium",
    )

    d = hyp.to_dict()
    assert d["hypothesis_id"] == "hyp_01"
    assert d["label"] == "Potential Hypothesis"
    assert d["variables"]["independent_variable"] == "Multilingual pre-training"
