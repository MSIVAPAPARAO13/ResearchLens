"""Unit tests for NLP and RAG-specific MLflow experiment tracking (Phase 13)."""

from unittest.mock import MagicMock

from researchlens.mlops.tracking import MLflowTracker


def test_nlp_rag_parameter_tracking_and_exact_model_identifiers():
    mock_client = MagicMock()
    tracker = MLflowTracker(client=mock_client)

    rag_params = {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "summarizer_model": "facebook/bart-large-cnn",
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "chunk_size": 900,
        "chunk_overlap": 100,
        "top_k": 5,
        "similarity_threshold": 0.25,
        "openai_api_key": "sk-secret-do-not-log",
    }

    tracker.log_parameters("run_nlp_01", rag_params)

    calls = mock_client.log_param.call_args_list
    logged = {call[0][1]: call[0][2] for call in calls}

    assert logged["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert logged["summarizer_model"] == "facebook/bart-large-cnn"
    assert logged["reranker_model"] == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert logged["top_k"] == "5"
    assert logged["openai_api_key"] == "[MASKED]"


def test_rag_evaluation_metric_tracking():
    mock_client = MagicMock()
    tracker = MLflowTracker(client=mock_client)

    rag_eval_metrics = {
        "retrieval_precision": 0.88,
        "retrieval_recall": 1.0,
        "mrr": 0.92,
        "answerability": 0.95,
        "citation_accuracy": 0.96,
        "rouge1": 0.42,
        "f1": 0.91,
    }

    tracker.log_metrics("run_nlp_01", rag_eval_metrics)

    calls = mock_client.log_metric.call_args_list
    logged_keys = {call[1]["key"]: call[1]["value"] for call in calls}

    assert logged_keys["retrieval_precision"] == 0.88
    assert logged_keys["retrieval_recall"] == 1.0
    assert logged_keys["mrr"] == 0.92
    assert logged_keys["answerability"] == 0.95
    assert logged_keys["citation_accuracy"] == 0.96
    assert logged_keys["rouge1"] == 0.42
    assert logged_keys["f1"] == 0.91
