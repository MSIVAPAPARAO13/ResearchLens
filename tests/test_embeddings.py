"""Unit tests for embedding model interfaces and SentenceTransformer abstraction."""

from unittest.mock import MagicMock, patch

import numpy as np

from researchlens.ingestion.document import ChunkData
from researchlens.retrieval.embeddings import (
    MockEmbeddingModel,
    SentenceTransformerEmbeddings,
    get_embedding_model,
)


def test_mock_embedding_model_interface():
    model = MockEmbeddingModel(dimension=64)
    query_vec = model.embed_query("What dataset was used?")
    assert len(query_vec) == 64
    assert isinstance(query_vec[0], float)

    texts = ["Sample text one", "Sample text two"]
    doc_vecs = model.embed_documents(texts)
    assert len(doc_vecs) == 2
    assert len(doc_vecs[0]) == 64

    # Test embed_text alias
    alias_vec = model.embed_text("What dataset was used?")
    assert alias_vec == query_vec


def test_mock_embedding_with_chunk_objects():
    model = MockEmbeddingModel(dimension=32)
    chunk = ChunkData(
        chunk_id="c1",
        text="Sample method description.",
        token_count=10,
        section_name="methodology",
        page_number=2,
    )
    vecs = model.embed_documents([chunk])
    assert len(vecs) == 1
    assert len(vecs[0]) == 32


def test_sentence_transformer_mocked_inference():
    """Verify SentenceTransformerEmbeddings delegates properly without downloading model."""
    mock_st_instance = MagicMock()
    mock_st_instance.get_sentence_embedding_dimension.return_value = 384
    mock_st_instance.encode.return_value = np.zeros((1, 384), dtype=np.float32)

    with patch("sentence_transformers.SentenceTransformer", return_value=mock_st_instance):
        emb_wrapper = SentenceTransformerEmbeddings(model_name="dummy-model", device="cpu")
        query_emb = emb_wrapper.embed_query("Test research question")
        assert len(query_emb) == 384

        doc_embs = emb_wrapper.embed_documents(["Document content excerpt"])
        assert len(doc_embs) == 1
        assert len(doc_embs[0]) == 384


def test_get_embedding_model_singleton():
    m1 = get_embedding_model(use_mock=True)
    assert isinstance(m1, MockEmbeddingModel)
