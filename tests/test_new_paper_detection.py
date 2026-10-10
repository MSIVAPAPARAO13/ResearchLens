"""Unit tests for new paper detection and stable key resolution."""

from researchlens.analysis.external_provider import ExternalPaper
from researchlens.analysis.monitoring_service import _resolve_paper_stable_key, detect_paper_changes
from researchlens.analysis.research_topic import PaperHistoryEntry


def test_resolve_paper_stable_key_priority():
    # DOI has highest priority
    p1 = ExternalPaper("p1", "Attention Is All You Need", doi="10.1145/123", arxiv_id="1706.03762", provider="arXiv")
    assert _resolve_paper_stable_key(p1) == "doi:10.1145/123"

    # arXiv ID second priority
    p2 = ExternalPaper("p2", "Attention Is All You Need", arxiv_id="1706.03762", provider="arXiv")
    assert _resolve_paper_stable_key(p2) == "arxiv:1706.03762"

    # Provider ID third priority
    p3 = ExternalPaper(paper_id="pmid_9999", title="Some PubMed Study", provider="PubMed")
    assert _resolve_paper_stable_key(p3) == "PubMed:pmid_9999"

    # Normalized title fallback
    p4 = ExternalPaper("p4", "A Novel Transformer for Image Super-Resolution!")
    assert _resolve_paper_stable_key(p4) == "title:a novel transformer for image superresolution"


def test_detect_new_papers_vs_existing():
    existing_history = {
        "doi:10.1000/1": PaperHistoryEntry(
            paper_id="hist_1",
            first_seen="2026-10-01T00:00:00Z",
            last_seen="2026-10-01T00:00:00Z",
            providers=["arXiv"],
            title="Old Transformer Study",
            authors=["Alice"],
            year=2024,
            doi="10.1000/1",
            arxiv_id=None,
            citation_count=10,
        )
    }

    p_old = ExternalPaper("p_old", "Old Transformer Study", doi="10.1000/1", citation_count=10)
    p_new = ExternalPaper("p_new", "Brand New Discovery in 2026", doi="10.1000/2", citation_count=0)

    new_papers, updated_papers, existing_unmod = detect_paper_changes(
        [p_old, p_new], existing_history
    )

    assert len(new_papers) == 1
    assert new_papers[0].paper_id == "p_new"

    assert len(updated_papers) == 0

    assert len(existing_unmod) == 1
    assert existing_unmod[0].paper_id == "p_old"
