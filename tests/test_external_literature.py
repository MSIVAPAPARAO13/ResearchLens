"""Unit tests for external literature discovery, multi-provider search, deduplication, and ranking."""

from researchlens.analysis.external_provider import (
    ExternalPaper,
    UnifiedLiteratureSearchService,
    convert_external_paper_to_document,
    deduplicate_papers,
    normalize_title,
    rank_papers_by_relevance,
)


def test_normalize_title():
    raw = "Attention Is All You Need: Transformer Models (2017)!"
    norm = normalize_title(raw)
    assert norm == "attention is all you need transformer models 2017"


def test_deduplicate_papers():
    p1 = ExternalPaper("p1", "Efficient Attention Networks", doi="10.1000/1", citation_count=10)
    p2 = ExternalPaper("p2", "Efficient Attention Networks (Extended)", doi="10.1000/1", citation_count=20)
    p3 = ExternalPaper("p3", "Different Paper on Transformers", doi="10.1000/2", citation_count=5)
    p4 = ExternalPaper("p4", "efficient attention networks", citation_count=15)  # Normalized title match

    papers = [p1, p2, p3, p4]
    deduped = deduplicate_papers(papers)

    assert len(deduped) == 2
    assert any(p.doi == "10.1000/1" for p in deduped)
    assert any(p.doi == "10.1000/2" for p in deduped)


def test_rank_papers_by_relevance():
    query = "efficient transformer memory optimization"
    p_rel = ExternalPaper(
        "p_rel",
        "Linear Transformer with Efficient Memory Optimization",
        abstract="We optimize attention memory complexity down to O(N).",
        year=2024,
        citation_count=50,
    )
    p_unrel = ExternalPaper(
        "p_unrel",
        "Economic Growth Trends in Agricultural Supply Chains",
        abstract="An econometric analysis of commodity price volatility.",
        year=2018,
        citation_count=2,
    )

    ranked = rank_papers_by_relevance(query, [p_unrel, p_rel], use_semantic=False)
    assert len(ranked) == 2
    assert ranked[0].paper_id == "p_rel"
    assert ranked[0].relevance_score > ranked[1].relevance_score


def test_unified_literature_search_service():
    service = UnifiedLiteratureSearchService()
    results = service.search_literature(
        query="graph neural networks",
        providers=["arxiv", "semanticscholar"],
        max_per_provider=2,
        use_semantic_ranking=False,
    )

    assert len(results) >= 1
    assert all(isinstance(p, ExternalPaper) for p in results)
    assert all(hasattr(p, "relevance_score") for p in results)


def test_convert_external_paper_to_document():
    ext_p = ExternalPaper(
        paper_id="ext_test_01",
        title="Multi-Scale Attention Architecture",
        authors=["Jane Smith", "Bob Jones"],
        year=2023,
        abstract="We present an empirical study of multi-scale attention architectures.",
        venue="NeurIPS 2023",
        doi="10.1145/test.01",
    )

    doc = convert_external_paper_to_document(ext_p)
    assert doc.document_id.startswith("doc_")
    assert doc.title == "Multi-Scale Attention Architecture"
    assert doc.profile is not None
    assert doc.profile.title == "Multi-Scale Attention Architecture"
    assert len(doc.document_obj.chunks) >= 1
