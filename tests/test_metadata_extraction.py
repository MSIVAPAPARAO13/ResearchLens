"""Unit tests for paper metadata extraction (title, authors, organizations, year)."""

import pytest

from researchlens.extraction.metadata import (
    extract_authors_and_orgs,
    extract_publication_year,
    extract_title,
)
from researchlens.ingestion.document import Document, DocumentMetadata, Page


@pytest.fixture
def sample_document():
    page1 = Page(
        page_number=1,
        text="""SciTransformer: Multi-Scale Transformer for Document Intelligence
John Doe, Jane Smith
Department of Computer Science, Stanford University
Published in Proceedings of NeurIPS 2023.

Abstract:
We present a transformer model for academic papers.
""",
    )
    doc = Document(
        filename="scitransformer.pdf",
        full_text=page1.text,
        pages=[page1],
        metadata=DocumentMetadata(title="SciTransformer: Multi-Scale Transformer for Document Intelligence"),
    )
    return doc


def test_extract_title_from_metadata(sample_document):
    title_item = extract_title(sample_document)
    assert title_item.value == "SciTransformer: Multi-Scale Transformer for Document Intelligence"
    assert title_item.extraction_method == "pdf_metadata"
    assert title_item.confidence > 0.9


def test_extract_title_from_header_fallback():
    page1 = Page(page_number=1, text="Deep Learning for Biomedical Image Segmentation\nAuthors: Alice Bob\nMIT")
    doc = Document(filename="paper.pdf", full_text=page1.text, pages=[page1], metadata=DocumentMetadata(title=None))

    title_item = extract_title(doc)
    assert "Deep Learning" in title_item.value
    assert title_item.page_number == 1


def test_extract_authors_and_orgs(sample_document):
    authors, orgs = extract_authors_and_orgs(sample_document)
    assert isinstance(authors, list)
    assert isinstance(orgs, list)
    assert len(authors) > 0
    assert len(orgs) > 0


def test_extract_publication_year(sample_document):
    year = extract_publication_year(sample_document)
    assert year == "2023"


def test_empty_document_metadata():
    doc = Document(filename="empty.pdf", full_text="", pages=[])
    title = extract_title(doc)
    assert title.value == "Not detected"
    assert title.confidence == 0.0

    authors, orgs = extract_authors_and_orgs(doc)
    assert authors[0].value == "Not detected"
    assert orgs[0].value == "Not detected"
