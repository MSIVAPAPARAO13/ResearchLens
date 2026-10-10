"""Unit tests for FAISS and InMemory vector storage."""

import pytest

from researchlens.ingestion.document import ChunkData
from researchlens.retrieval.vector_store import (
    FAISSVectorStore,
    InMemoryVectorStore,
    VectorDocument,
    create_vector_store,
)


@pytest.fixture
def sample_vector_docs():
    # 3-dimensional normalized unit vectors for clear geometric similarity testing
    return [
        VectorDocument(
            doc_id="doc_a",
            text="Methodology discussing transformer architecture",
            metadata={"page_number": 2, "section_name": "methodology"},
            embedding=[1.0, 0.0, 0.0],
        ),
        VectorDocument(
            doc_id="doc_b",
            text="Dataset details on ImageNet and GLUE",
            metadata={"page_number": 4, "section_name": "dataset"},
            embedding=[0.0, 1.0, 0.0],
        ),
        VectorDocument(
            doc_id="doc_c",
            text="Experimental results achieving high accuracy",
            metadata={"page_number": 6, "section_name": "results"},
            embedding=[0.0, 0.0, 1.0],
        ),
    ]


def test_in_memory_vector_store(sample_vector_docs):
    store = InMemoryVectorStore()
    store.add_documents(sample_vector_docs)

    # Query vector aligned with doc_a
    results = store.similarity_search([1.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].doc_id == "doc_a"
    assert results[0].metadata["page_number"] == 2
    assert results[0].metadata["section_name"] == "methodology"
    assert results[0].score >= 0.99

    store.clear()
    assert store.similarity_search([1.0, 0.0, 0.0]) == []


def test_faiss_vector_store(sample_vector_docs):
    store = FAISSVectorStore()
    store.add_documents(sample_vector_docs)

    # Query aligned with doc_b
    results = store.similarity_search([0.0, 1.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].doc_id == "doc_b"
    assert results[0].metadata["page_number"] == 4
    assert results[0].metadata["section_name"] == "dataset"
    assert results[0].score >= 0.99

    store.clear()
    assert store.similarity_search([0.0, 1.0, 0.0]) == []


def test_faiss_add_chunks_helper():
    chunks = [
        ChunkData(
            chunk_id="chunk_01",
            text="Testing chunk helper function.",
            token_count=8,
            page_number=3,
            section_name="introduction",
        )
    ]
    embeddings = [[0.7071, 0.7071]]

    store = FAISSVectorStore()
    store.add_chunks(chunks, embeddings)
    res = store.similarity_search([0.7071, 0.7071], top_k=1)
    assert len(res) == 1
    assert res[0].doc_id == "chunk_01"
    assert res[0].metadata["page_number"] == 3


def test_create_vector_store_factory():
    store = create_vector_store("faiss")
    assert isinstance(store, (FAISSVectorStore, InMemoryVectorStore))
    in_mem = create_vector_store("in_memory")
    assert isinstance(in_mem, InMemoryVectorStore)
