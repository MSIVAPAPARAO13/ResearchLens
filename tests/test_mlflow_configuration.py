"""Unit tests for MLflow configuration and environment variable loading (Phase 13)."""

import os
from unittest.mock import patch

from researchlens.mlops.configuration import get_mlflow_config


def test_mlflow_configuration_defaults():
    with patch.dict(os.environ, {}, clear=True):
        config = get_mlflow_config()
        assert config.enabled is False
        assert config.tracking_uri == "./mlruns"
        assert config.experiment_prefix == "ResearchLens::"
        assert config.auto_logging is True
        assert config.max_artifact_size_mb == 50.0
        assert config.retry_attempts == 3


def test_mlflow_configuration_environment_overrides():
    env_vars = {
        "RESEARCHLENS_MLFLOW_ENABLED": "true",
        "MLFLOW_TRACKING_URI": "http://localhost:5000",
        "RESEARCHLENS_MLFLOW_EXPERIMENT_PREFIX": "CustomLab::",
        "MAX_MLFLOW_ARTIFACT_SIZE_MB": "100.5",
        "MLFLOW_RETRY_ATTEMPTS": "5",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        config = get_mlflow_config()
        assert config.enabled is True
        assert config.tracking_uri == "http://localhost:5000"
        assert config.experiment_prefix == "CustomLab::"
        assert config.max_artifact_size_mb == 100.5
        assert config.retry_attempts == 5


def test_mlflow_configuration_invalid_env_fallbacks():
    env_vars = {
        "MAX_MLFLOW_ARTIFACT_SIZE_MB": "not_a_number",
        "MLFLOW_RETRY_ATTEMPTS": "invalid_int",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        config = get_mlflow_config()
        assert config.max_artifact_size_mb == 50.0
        assert config.retry_attempts == 3
