"""Unit tests for recurring limitation theme extraction across multi-paper collections."""

from researchlens.analysis.literature_collection import LiteratureCollection, LiteratureDocument
from researchlens.analysis.semantic_clustering import cluster_limitations_and_future_work
from researchlens.extraction.schema import ExtractedItem, ResearchProfile


def test_recurring_limitation_clustering():
    lim1 = ExtractedItem("A primary limitation is high memory footprint during long-sequence training.", page_number=5)
    lim2 = ExtractedItem("Memory consumption scales quadratically with input text sequence length.", page_number=8)

    prof1 = ResearchProfile(document_id="d1", title="Paper A", limitations=[lim1])
    prof2 = ResearchProfile(document_id="d2", title="Paper B", limitations=[lim2])

    coll = LiteratureCollection(collection_id="col_lims", collection_name="Limitation Test")
    coll.add_document(LiteratureDocument("d1", "a.pdf", "Paper A", profile=prof1))
    coll.add_document(LiteratureDocument("d2", "b.pdf", "Paper B", profile=prof2))

    lim_clusters, _ = cluster_limitations_and_future_work(coll, threshold=0.55)

    assert len(lim_clusters) >= 1
    top_c = lim_clusters[0]
    assert top_c.recurrence_count == 2
    assert "Paper A" in top_c.supporting_papers
    assert "Paper B" in top_c.supporting_papers
