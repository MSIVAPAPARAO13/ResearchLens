"""Unit tests for changed/updated paper detection across monitoring runs."""

from researchlens.analysis.external_provider import ExternalPaper
from researchlens.analysis.monitoring_service import detect_paper_changes
from researchlens.analysis.research_topic import PaperHistoryEntry


def test_detect_updated_citation_and_pdf_url():
    history = {
        "arxiv:2401.0001": PaperHistoryEntry(
            paper_id="p1",
            first_seen="2026-10-01T00:00:00Z",
            last_seen="2026-10-01T00:00:00Z",
            providers=["arXiv"],
            title="Reasoning with Chain-of-Thought",
            authors=["Jason Wei"],
            year=2024,
            doi=None,
            arxiv_id="2401.0001",
            citation_count=150,
            pdf_url=None,
            abstract="Initial abstract text.",
        )
    }

    # Same paper observed later with increased citations and new PDF URL
    p_updated = ExternalPaper(
        paper_id="p1_fresh",
        title="Reasoning with Chain-of-Thought",
        arxiv_id="2401.0001",
        citation_count=230,  # Changed from 150
        pdf_url="https://arxiv.org/pdf/2401.0001.pdf",  # Newly available PDF
        abstract="Initial abstract text.",
    )

    new_papers, updated_papers, existing_unmod = detect_paper_changes(
        [p_updated], history
    )

    assert len(new_papers) == 0
    assert len(existing_unmod) == 0
    assert len(updated_papers) == 1

    paper, diffs = updated_papers[0]
    assert paper.paper_id == "p1_fresh"
    assert "citation_count" in diffs
    assert diffs["citation_count"]["old"] == 150
    assert diffs["citation_count"]["new"] == 230
    assert "pdf_url" in diffs
