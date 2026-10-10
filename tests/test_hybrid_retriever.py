"""Unit tests for HybridRetriever, score normalization, candidate fusion, and pipeline."""

import pytest

from researchlens.ingestion.document import ChunkData
from researchlens.retrieval.bm25 import BM25Retriever
from researchlens.retrieval.embeddings import MockEmbeddingModel
from researchlens.retrieval.hybrid_retriever import (
    HybridRetriever,
    ScientificRetrievalPipeline,
    min_max_normalize,
)
from researchlens.retrieval.reranker import MockReranker
from researchlens.retrieval.retriever import DenseRetriever, SearchResult
from researchlens.retrieval.vector_store import InMemoryVectorStore


@pytest.fixture
def mock_retrieval_components():
    chunks = [
        ChunkData(
            chunk_id="chunk_1",
            text="Convolutional neural networks for image classification on ImageNet.",
            token_count=10,
            page_number=1,
            section_name="introduction",
            document_id="doc_1",
            filename="test.pdf",
            chunk_index=0,
        ),
        ChunkData(
            chunk_id="chunk_2",
            text="Transformers achieve state of the art results with attention mechanisms.",
            token_count=10,
            page_number=2,
            section_name="methods",
            document_id="doc_1",
            filename="test.pdf",
            chunk_index=1,
        ),
        ChunkData(
            chunk_id="chunk_3",
            text="Evaluation metrics reported include accuracy, F1-score, and top-5 error.",
            token_count=10,
            page_number=3,
            section_name="results",
            document_id="doc_1",
            filename="test.pdf",
            chunk_index=2,
        ),
    ]

    # BM25
    bm25 = BM25Retriever()
    bm25.index_chunks(chunks)

    # Dense with mock embeddings
    emb_model = MockEmbeddingModel(dimension=32)
    vectors = emb_model.embed_documents(chunks)
    vstore = InMemoryVectorStore()
    vstore.add_chunks(chunks, vectors)
    dense = DenseRetriever(embedding_model=emb_model, vector_store=vstore, similarity_threshold=0.0)

    return dense, bm25, chunks


def test_min_max_normalize():
    scores = {"doc_a": 10.0, "doc_b": 20.0, "doc_c": 30.0}
    normed = min_max_normalize(scores)
    assert normed["doc_a"] == 0.0
    assert normed["doc_c"] == 1.0
    assert pytest.approx(normed["doc_b"]) == 0.5


def test_min_max_normalize_all_equal():
    scores = {"doc_a": 5.0, "doc_b": 5.0}
    normed = min_max_normalize(scores)
    assert normed["doc_a"] == 1.0
    assert normed["doc_b"] == 1.0

    empty_normed = min_max_normalize({})
    assert empty_normed == {}


def test_hybrid_retriever_fusion(mock_retrieval_components):
    dense, bm25, _ = mock_retrieval_components
    hybrid = HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
        dense_weight=0.6,
        bm25_weight=0.4,
    )

    results = hybrid.retrieve("neural networks accuracy", top_k=2)
    assert len(results) <= 2
    assert len(results) > 0
    top = results[0]
    assert isinstance(top, SearchResult)
    assert "dense_raw_score" in top.metadata
    assert "bm25_raw_score" in top.metadata
    assert "hybrid_score" in top.metadata


def test_hybrid_weight_normalization(mock_retrieval_components):
    dense, bm25, _ = mock_retrieval_components
    hybrid = HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
        dense_weight=3.0,
        bm25_weight=1.0,
    )
    # Weights should normalize to 0.75 and 0.25
    assert pytest.approx(hybrid.dense_weight) == 0.75
    assert pytest.approx(hybrid.bm25_weight) == 0.25


def test_duplicate_candidate_handling(mock_retrieval_components):
    dense, bm25, _ = mock_retrieval_components
    hybrid = HybridRetriever(dense, bm25)

    results = hybrid.retrieve("classification", top_k=5)
    # Ensure no duplicates in final results
    doc_ids = [r.doc_id for r in results]
    assert len(doc_ids) == len(set(doc_ids))


def test_scientific_retrieval_pipeline_strategies(mock_retrieval_components):
    dense, bm25, _ = mock_retrieval_components
    reranker = MockReranker()
    pipeline = ScientificRetrievalPipeline(
        dense_retriever=dense,
        bm25_retriever=bm25,
        reranker=reranker,
    )

    dense_res = pipeline.retrieve("attention mechanisms", strategy="dense", top_k=2)
    assert len(dense_res) > 0

    bm25_res = pipeline.retrieve("attention mechanisms", strategy="bm25", top_k=2)
    assert len(bm25_res) > 0

    hybrid_res = pipeline.retrieve("attention mechanisms", strategy="hybrid", top_k=2)
    assert len(hybrid_res) > 0

    reranked_res = pipeline.retrieve("attention mechanisms", strategy="hybrid_reranked", top_k=2)
    assert len(reranked_res) > 0
    assert "reranker_score" in reranked_res[0].metadata


def test_invalid_strategy_raises_error(mock_retrieval_components):
    dense, bm25, _ = mock_retrieval_components
    pipeline = ScientificRetrievalPipeline(dense, bm25)
    with pytest.raises(ValueError, match="Unknown retrieval strategy"):
        pipeline.retrieve("test query", strategy="unsupported_strategy")
