"""Unit tests for Cross-Encoder reranker, candidate scoring, and metadata preservation."""

from unittest.mock import MagicMock, patch

import pytest

from researchlens.retrieval.reranker import (
    CrossEncoderReranker,
    MockReranker,
    get_reranker,
)
from researchlens.retrieval.retriever import SearchResult


@pytest.fixture
def sample_candidates():
    return [
        SearchResult(
            doc_id="c_1",
            text="General overview of machine learning applications in astronomy.",
            score=0.85,
            page_number=1,
            section_name="introduction",
            metadata={"filename": "paper.pdf", "chunk_index": 0},
        ),
        SearchResult(
            doc_id="c_2",
            text="The proposed ResNet model achieved an accuracy of 94.2% on benchmark datasets.",
            score=0.72,
            page_number=4,
            section_name="results",
            metadata={"filename": "paper.pdf", "chunk_index": 3},
        ),
        SearchResult(
            doc_id="c_3",
            text="Hyperparameter tuning was conducted using grid search and 5-fold cross validation.",
            score=0.68,
            page_number=3,
            section_name="methodology",
            metadata={"filename": "paper.pdf", "chunk_index": 2},
        ),
    ]


def test_mock_reranker_reordering(sample_candidates):
    reranker = MockReranker()
    # Query matching c_2 heavily
    query = "ResNet model accuracy benchmark"
    reranked = reranker.rerank(query, sample_candidates, top_k=2)

    assert len(reranked) == 2
    # c_2 should be ranked top due to highest lexical term match with query
    assert reranked[0].doc_id == "c_2"
    assert reranked[0].reranker_score is not None
    assert reranked[0].metadata["retrieval_score"] == 0.72
    assert reranked[0].page_number == 4
    assert reranked[0].section_name == "results"


def test_reranker_empty_inputs(sample_candidates):
    reranker = MockReranker()
    assert reranker.rerank("", sample_candidates) == []
    assert reranker.rerank("   ", sample_candidates) == []
    assert reranker.rerank("valid query", []) == []


def test_cross_encoder_lazy_loading():
    reranker = CrossEncoderReranker(model_name="dummy/test-model")
    # Model should NOT be loaded upon instantiation
    assert reranker._model is None


@patch("sentence_transformers.CrossEncoder")
def test_cross_encoder_rerank_with_mock(mock_ce_cls, sample_candidates):
    # Mock the underlying CrossEncoder instance
    mock_instance = MagicMock()
    # Return scores where candidate index 1 has highest score, 0 has lowest
    mock_instance.predict.return_value = [-1.5, 3.8, 1.2]
    mock_ce_cls.return_value = mock_instance

    reranker = CrossEncoderReranker(model_name="dummy/test-model")
    reranked = reranker.rerank("accuracy benchmark", sample_candidates, top_k=3)

    assert len(reranked) == 3
    # Candidate 1 (c_2) should be first with score 3.8
    assert reranked[0].doc_id == "c_2"
    assert reranked[0].score == 3.8
    # Candidate 2 (c_3) should be second with score 1.2
    assert reranked[1].doc_id == "c_3"
    assert reranked[1].score == 1.2
    # Candidate 0 (c_1) should be third with score -1.5
    assert reranked[2].doc_id == "c_1"
    assert reranked[2].score == -1.5

    # Verify (query, passage) pairs were passed to predict
    mock_instance.predict.assert_called_once()
    pairs = mock_instance.predict.call_args[0][0]
    assert len(pairs) == 3
    assert pairs[0] == ("accuracy benchmark", sample_candidates[0].text)


def test_get_reranker_factory():
    mock_r = get_reranker(use_mock=True)
    assert isinstance(mock_r, MockReranker)
