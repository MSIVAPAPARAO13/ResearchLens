"""Unit tests for MLflow parameter logging and secret masking (Phase 13)."""

from unittest.mock import MagicMock

from researchlens.mlops.models import sanitize_parameters
from researchlens.mlops.tracking import MLflowTracker


def test_sanitize_parameters_masks_credentials():
    dirty_params = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "optimizer": "adamw",
        "api_key": "sk-proj-supersecret12345",
        "aws_secret_key": "AKIA123SECRET456",
        "db_password": "p@ssword99!",
        "auth_token": "bearer eyJhbGciOi...",
    }

    clean = sanitize_parameters(dirty_params)
    assert clean["learning_rate"] == 0.001
    assert clean["batch_size"] == 32
    assert clean["optimizer"] == "adamw"
    assert clean["api_key"] == "[MASKED]"
    assert clean["aws_secret_key"] == "[MASKED]"
    assert clean["db_password"] == "[MASKED]"
    assert clean["auth_token"] == "[MASKED]"


def test_tracker_log_parameters_invokes_client_with_clean_data():
    mock_client = MagicMock()
    tracker = MLflowTracker(client=mock_client)

    params = {
        "epochs": 10,
        "user_token": "token_xyz",
        "seed": 42,
    }

    tracker.log_parameters("mlflow_run_99", params)

    calls = mock_client.log_param.call_args_list
    logged = {call[0][1]: call[0][2] for call in calls}

    assert logged["epochs"] == "10"
    assert logged["seed"] == "42"
    assert logged["user_token"] == "[MASKED]"
