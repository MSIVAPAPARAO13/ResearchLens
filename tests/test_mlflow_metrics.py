"""Unit tests for MLflow metric logging and range normalization (Phase 13)."""

from unittest.mock import MagicMock

from researchlens.mlops.tracking import MLflowTracker


def test_log_metrics_normalizes_percentages_and_logs():
    mock_client = MagicMock()
    tracker = MLflowTracker(client=mock_client)

    metrics = {
        "accuracy": 93.4,  # should be normalized to 0.934
        "f1": 0.912,       # already fraction
        "latency": 15.6,   # unbounded non-negative
        "invalid_metric": "non_numeric_score",  # should be dropped
    }

    tracker.log_metrics("run_123", metrics, step=1)

    calls = mock_client.log_metric.call_args_list
    logged_keys = {call[1]["key"]: call[1]["value"] for call in calls}

    assert "accuracy" in logged_keys
    assert abs(logged_keys["accuracy"] - 0.934) < 1e-4
    assert "f1" in logged_keys
    assert abs(logged_keys["f1"] - 0.912) < 1e-4
    assert "latency" in logged_keys
    assert abs(logged_keys["latency"] - 15.6) < 1e-4
    assert "invalid_metric" not in logged_keys


def test_log_metrics_step_support():
    mock_client = MagicMock()
    tracker = MLflowTracker(client=mock_client)

    tracker.log_metrics("run_123", {"loss": 0.45}, step=1)
    tracker.log_metrics("run_123", {"loss": 0.32}, step=2)

    calls = mock_client.log_metric.call_args_list
    assert len(calls) == 2
    assert calls[0][1]["step"] == 1
    assert calls[0][1]["value"] == 0.45
    assert calls[1][1]["step"] == 2
    assert calls[1][1]["value"] == 0.32
