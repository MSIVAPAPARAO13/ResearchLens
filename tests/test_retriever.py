"""Unit tests for DenseRetriever orchestration and provenance preservation."""

from researchlens.ingestion.document import ChunkData
from researchlens.retrieval.embeddings import MockEmbeddingModel
from researchlens.retrieval.retriever import DenseRetriever
from researchlens.retrieval.vector_store import InMemoryVectorStore


def test_dense_retriever_query_and_provenance():
    emb_model = MockEmbeddingModel(dimension=16)
    chunks = [
        ChunkData(
            chunk_id="paper1_p03_c01",
            text="The Transformer architecture uses multi-head attention.",
            token_count=12,
            page_number=3,
            section_name="methodology",
        ),
        ChunkData(
            chunk_id="paper1_p05_c02",
            text="We evaluate on the ImageNet validation dataset.",
            token_count=10,
            page_number=5,
            section_name="dataset",
        ),
    ]

    store = InMemoryVectorStore()
    embeddings = emb_model.embed_documents(chunks)
    store.add_chunks(chunks, embeddings)

    retriever = DenseRetriever(
        embedding_model=emb_model,
        vector_store=store,
        similarity_threshold=0.0,
    )

    results = retriever.retrieve("What architecture was used?", top_k=2)
    assert len(results) == 2

    first = results[0]
    assert first.doc_id in ["paper1_p03_c01", "paper1_p05_c02"]
    assert first.page_number in [3, 5]
    assert first.section_name in ["methodology", "dataset"]
    assert first.score >= 0.0
    assert first.text


def test_dense_retriever_empty_query():
    emb_model = MockEmbeddingModel(dimension=16)
    store = InMemoryVectorStore()
    retriever = DenseRetriever(emb_model, store)
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


def test_dense_retriever_similarity_threshold_filtering():
    emb_model = MockEmbeddingModel(dimension=4)
    chunks = [
        ChunkData(
            chunk_id="c1",
            text="Only chunk in index.",
            token_count=5,
            page_number=1,
            section_name="intro",
        )
    ]
    store = InMemoryVectorStore()
    embeddings = [[0.1, 0.0, 0.0, 0.0]]
    store.add_chunks(chunks, embeddings)

    # Set an impossible similarity threshold (e.g. 1.5)
    retriever = DenseRetriever(
        embedding_model=emb_model,
        vector_store=store,
        similarity_threshold=1.5,
    )
    results = retriever.retrieve("test query")
    assert results == []
