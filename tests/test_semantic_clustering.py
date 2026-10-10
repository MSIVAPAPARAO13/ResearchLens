"""Unit tests for semantic clustering of cross-paper statements into recurring themes."""

from researchlens.analysis.literature_collection import LiteratureCollection, LiteratureDocument
from researchlens.analysis.semantic_clustering import cluster_limitations_and_future_work
from researchlens.extraction.schema import ExtractedItem, ResearchProfile


def test_semantic_clustering_ablation_modes():
    lim1 = ExtractedItem("High GPU memory footprint during long-sequence training.", page_number=5)
    lim2 = ExtractedItem("High memory footprint constrains training long sequences.", page_number=8)

    prof1 = ResearchProfile(document_id="d1", title="Study A", limitations=[lim1])
    prof2 = ResearchProfile(document_id="d2", title="Study B", limitations=[lim2])

    coll = LiteratureCollection(collection_id="c_clust", collection_name="Cluster Test")
    coll.add_document(LiteratureDocument("d1", "a.pdf", "Study A", profile=prof1))
    coll.add_document(LiteratureDocument("d2", "b.pdf", "Study B", profile=prof2))

    # Test full_evidence mode
    c_lims, _ = cluster_limitations_and_future_work(coll, threshold=0.45, ablation_mode="full_evidence")
    assert len(c_lims) >= 1
    assert c_lims[0].recurrence_count == 2

    # Test lexical_only mode
    c_lex, _ = cluster_limitations_and_future_work(coll, ablation_mode="lexical_only")
    assert len(c_lex) >= 1
