"""Unit tests for paper limitation and future work extraction."""

import pytest

from researchlens.extraction.gaps_limitations import extract_limitations_and_future_work
from researchlens.ingestion.document import ChunkData, Document


@pytest.fixture
def limitation_document():
    chunk1 = ChunkData(
        chunk_id="c_lim1",
        text="A primary limitation of our approach is the quadratic memory footprint during long-sequence training.",
        token_count=17,
        section_name="discussion",
        page_number=8,
    )
    chunk2 = ChunkData(
        chunk_id="c_fut2",
        text="In the future, we plan to extend our evaluation to non-English multilingual corpora.",
        token_count=14,
        section_name="conclusion",
        page_number=9,
    )
    return Document(filename="limits_paper.pdf", full_text=chunk1.text + "\n" + chunk2.text, chunks=[chunk1, chunk2])


def test_extract_limitations_and_future_work(limitation_document):
    limitations, future_work = extract_limitations_and_future_work(limitation_document)

    assert len(limitations) > 0
    assert len(future_work) > 0

    assert "quadratic memory" in limitations[0].value.lower()
    assert limitations[0].page_number == 8
    assert limitations[0].chunk_id == "c_lim1"

    assert "multilingual" in future_work[0].value.lower()
    assert future_work[0].page_number == 9
    assert future_work[0].chunk_id == "c_fut2"


def test_extract_limitations_missing():
    doc = Document(filename="nolimits.pdf", full_text="The experiment finished successfully.", chunks=[])
    limitations, future_work = extract_limitations_and_future_work(doc)

    assert limitations[0].value == "Not detected"
    assert future_work[0].value == "Not detected"
