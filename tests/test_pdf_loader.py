"""Unit tests for PDF loading and text extraction."""

try:
    import pymupdf as fitz
except ImportError:
    import fitz
import pytest

from researchlens.ingestion.pdf_loader import extract_text_from_pdf, load_pdf


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Generate a clean in-memory PDF with known text."""
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((50, 72), "Abstract\nThis is a sample research abstract.")
    page2 = doc.new_page()
    page2.insert_text((50, 72), "Introduction\nThis is the introductory section.")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_extract_text_from_pdf_bytes(sample_pdf_bytes):
    text = extract_text_from_pdf(sample_pdf_bytes)
    assert "Abstract" in text
    assert "Introduction" in text
    assert "sample research abstract" in text


def test_load_pdf_structured_document(sample_pdf_bytes):
    doc = load_pdf(sample_pdf_bytes, filename="test_paper.pdf")
    assert doc.filename == "test_paper.pdf"
    assert doc.page_count == 2
    assert len(doc.pages) == 2
    assert doc.pages[0].page_number == 1
    assert "Abstract" in doc.pages[0].text
    assert doc.pages[1].page_number == 2
    assert "Introduction" in doc.pages[1].text
    assert doc.word_count > 0


def test_invalid_pdf_source():
    with pytest.raises(Exception):
        extract_text_from_pdf(b"invalid pdf data")
