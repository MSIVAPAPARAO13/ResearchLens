"""Unit tests for ResearchGap provenance tracking and evidence attribution."""

from researchlens.analysis.gap_engine import ResearchGapEngine
from researchlens.extraction.schema import ExtractedItem, ResearchProfile


def test_gap_provenance_retention():
    ev = ExtractedItem(
        value="Drawback includes sensitivity to high feature dimensionality.",
        source_text="Drawback includes sensitivity to high feature dimensionality.",
        page_number=6,
        section_name="discussion",
        chunk_id="chunk_pm_6",
    )
    profile = ResearchProfile(
        document_id="pm_doc",
        title="Industrial Maintenance Paper",
        limitations=[ev],
    )

    engine = ResearchGapEngine(ablation_mode="full")
    gaps, _ = engine.analyze_profile(profile)

    assert len(gaps) >= 1
    g = gaps[0]
    assert g.source_pages == [6]
    assert g.source_sections == ["discussion"]
    assert g.source_chunks == ["chunk_pm_6"]
    assert len(g.supporting_evidence) == 1
    assert g.supporting_evidence[0].source_text == ev.source_text
