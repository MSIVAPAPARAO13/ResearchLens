"""Unit tests for MLflow client initialization and health checks (Phase 13)."""

from unittest.mock import MagicMock, patch

from researchlens.mlops.client import check_mlflow_health
from researchlens.mlops.configuration import MLflowConfig


def test_mlflow_client_disabled():
    config = MLflowConfig(enabled=False, tracking_uri="./mlruns")
    health = check_mlflow_health(config)
    assert health["status"] == "DISABLED"
    assert health["experiments_count"] == 0
    assert health["error"] is None


def test_mlflow_client_connected_mock():
    config = MLflowConfig(enabled=True, tracking_uri="./mlruns")
    mock_client = MagicMock()
    mock_client.search_experiments.return_value = [MagicMock(), MagicMock()]

    with patch("mlflow.tracking.MlflowClient", return_value=mock_client), patch("mlflow.set_tracking_uri"):
        health = check_mlflow_health(config)
        assert health["status"] == "CONNECTED"
        assert health["experiments_count"] == 2
        assert health["error"] is None


def test_mlflow_client_unavailable_handles_gracefully():
    config = MLflowConfig(enabled=True, tracking_uri="http://bad-uri-that-fails:9999")
    with patch("mlflow.tracking.MlflowClient", side_effect=ConnectionError("Cannot connect to remote server")):
        health = check_mlflow_health(config)
        assert health["status"] == "UNAVAILABLE"
        assert "Cannot connect to remote server" in health["error"]
        assert health["experiments_count"] == 0


def test_mlflow_client_missing_import_handles_gracefully():
    config = MLflowConfig(enabled=True, tracking_uri="./mlruns")
    with patch.dict("sys.modules", {"mlflow": None, "mlflow.tracking": None}):
        with patch("builtins.__import__", side_effect=ImportError("No module named 'mlflow'")):
            health = check_mlflow_health(config)
            assert health["status"] == "UNAVAILABLE"
            assert "not installed" in health["error"]
