"""Unit tests for token-bounded and sentence-aware chunking."""

from researchlens.preprocessing.chunker import chunk_text, create_chunks


def test_chunk_text_empty_input():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_sentence_integrity():
    text = (
        "Sentence one is concise. "
        "Sentence two provides additional details. "
        "Sentence three concludes the paragraph."
    )
    chunks = chunk_text(text, max_tokens=100, overlap=0)
    assert len(chunks) == 1
    assert "Sentence one is concise." in chunks[0]
    assert "Sentence three concludes the paragraph." in chunks[0]


def test_chunk_text_splits_when_exceeding_budget():
    sentences = [f"This is experimental sentence number {i} for testing." for i in range(50)]
    text = " ".join(sentences)
    chunks = chunk_text(text, max_tokens=50, overlap=0)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.strip()) > 0


def test_create_chunks_metadata():
    text = "First sentence of the method. Second sentence explains parameters."
    chunks = create_chunks(text, section_name="methodology", max_tokens=50, overlap=0)
    assert len(chunks) >= 1
    first = chunks[0]
    assert first.section == "methodology"
    assert first.token_count > 0
    assert "First sentence" in first.text
    assert first.chunk_id == "doc_p01_c00"
    assert first.chunk_index == 0
    assert first.section_name == "methodology"
    d = first.to_dict()
    assert d["chunk_id"] == "doc_p01_c00"
    assert d["section_name"] == "methodology"
