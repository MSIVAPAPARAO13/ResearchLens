"""Unit tests for CrossPaperResearchGapEngine and literature-level research gap detection."""

from researchlens.analysis.cross_paper_gap_engine import CrossPaperResearchGapEngine
from researchlens.analysis.literature_collection import LiteratureCollection, LiteratureDocument
from researchlens.extraction.schema import ExtractedItem, ResearchProfile


def test_cross_paper_gap_engine_analysis():
    lim1 = ExtractedItem("Limited evaluation on single benchmark dataset.", page_number=5)
    lim2 = ExtractedItem("Evaluation was restricted to a single benchmark dataset.", page_number=7)

    prof1 = ResearchProfile(document_id="d1", title="Paper A", limitations=[lim1])
    prof2 = ResearchProfile(document_id="d2", title="Paper B", limitations=[lim2])

    coll = LiteratureCollection(collection_id="c_gap", collection_name="Cross Gap Test")
    coll.add_document(LiteratureDocument("d1", "a.pdf", "Paper A", profile=prof1))
    coll.add_document(LiteratureDocument("d2", "b.pdf", "Paper B", profile=prof2))

    engine = CrossPaperResearchGapEngine(ablation_mode="full_evidence")
    gaps = engine.analyze_collection(coll)

    assert len(gaps) >= 1
    top_g = gaps[0]
    assert top_g.recurrence_count == 2
    assert "Across the analyzed collection" in top_g.statement
    assert top_g.confidence_level in ("High", "Medium")


def test_cross_paper_gap_engine_empty_collection():
    empty_coll = LiteratureCollection(collection_id="c_empty", collection_name="Empty Collection")
    engine = CrossPaperResearchGapEngine(ablation_mode="full_evidence")
    gaps = engine.analyze_collection(empty_coll)

    assert len(gaps) == 1
    assert gaps[0].gap_id == "gap_lit_refusal"
    assert gaps[0].confidence_score == 0.0
