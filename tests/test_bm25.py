"""Unit tests for BM25 lexical retriever and tokenizer."""

import pytest

from researchlens.ingestion.document import ChunkData
from researchlens.retrieval.bm25 import BM25Retriever, tokenize_for_bm25


@pytest.fixture
def sample_chunks():
    return [
        ChunkData(
            chunk_id="chunk_0",
            text="ResNet-50 and Vision Transformers were evaluated on the ImageNet benchmark.",
            token_count=12,
            page_number=1,
            section_name="introduction",
            document_id="doc_123",
            filename="vision.pdf",
            chunk_index=0,
        ),
        ChunkData(
            chunk_id="chunk_1",
            text="We achieved a high F1-score of 92.4% using AdamW optimizer with cosine learning rate.",
            token_count=15,
            page_number=2,
            section_name="results",
            document_id="doc_123",
            filename="vision.pdf",
            chunk_index=1,
        ),
        ChunkData(
            chunk_id="chunk_2",
            text="The primary limitation is high GPU memory requirements during training.",
            token_count=10,
            page_number=3,
            section_name="discussion",
            document_id="doc_123",
            filename="vision.pdf",
            chunk_index=2,
        ),
    ]


def test_tokenize_for_bm25():
    text = "ResNet-50 achieved 92.4% F1-score on CIFAR-100."
    tokens = tokenize_for_bm25(text)
    assert "resnet-50" in tokens
    assert "f1-score" in tokens
    assert "cifar-100" in tokens
    assert "92" in tokens or "achieved" in tokens


def test_tokenize_empty():
    assert tokenize_for_bm25("") == []
    assert tokenize_for_bm25("   ") == []


def test_bm25_indexing_and_search(sample_chunks):
    retriever = BM25Retriever()
    retriever.index_chunks(sample_chunks)

    results = retriever.retrieve("ResNet-50 benchmark", top_k=2)
    assert len(results) > 0
    assert results[0].doc_id == "chunk_0"
    assert results[0].score > 0.0


def test_bm25_metadata_preservation(sample_chunks):
    retriever = BM25Retriever()
    retriever.index_chunks(sample_chunks)

    results = retriever.retrieve("F1-score AdamW", top_k=1)
    assert len(results) == 1
    top = results[0]
    assert top.doc_id == "chunk_1"
    assert top.page_number == 2
    assert top.section_name == "results"
    assert top.filename == "vision.pdf"
    assert top.document_id == "doc_123"
    assert top.chunk_index == 1
    assert top.chunk_id == "chunk_1"


def test_bm25_empty_query_and_empty_index(sample_chunks):
    retriever = BM25Retriever()
    # Unindexed retriever
    assert retriever.retrieve("query") == []

    # Indexed but empty query
    retriever.index_chunks(sample_chunks)
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


def test_bm25_no_match(sample_chunks):
    retriever = BM25Retriever()
    retriever.index_chunks(sample_chunks)

    # Completely unrelated query
    results = retriever.retrieve("quantum astrophysics astronomy")
    # All scores should be 0; should return empty or non-positive
    assert all(r.score <= 0.0 for r in results) or len(results) == 0


def test_bm25_clear(sample_chunks):
    retriever = BM25Retriever()
    retriever.index_chunks(sample_chunks)
    assert len(retriever.retrieve("ResNet-50")) > 0

    retriever.clear()
    assert retriever.retrieve("ResNet-50") == []
