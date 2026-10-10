"""Unit tests for MLflow experiment and run tracking lifecycle (Phase 13)."""

from unittest.mock import MagicMock

from researchlens.mlops.configuration import MLflowConfig
from researchlens.mlops.tracking import MLflowTracker


def test_get_or_create_experiment_existing():
    mock_client = MagicMock()
    mock_exp = MagicMock()
    mock_exp.experiment_id = "exp_101"
    mock_client.get_experiment_by_name.return_value = mock_exp

    config = MLflowConfig(enabled=True, experiment_prefix="TestLab::")
    tracker = MLflowTracker(config=config, client=mock_client)

    exp_id = tracker.get_or_create_experiment("Bioinformatics Workspace", "ws_01")
    assert exp_id == "exp_101"
    mock_client.get_experiment_by_name.assert_called_once_with("TestLab::Bioinformatics Workspace")
    mock_client.create_experiment.assert_not_called()


def test_get_or_create_experiment_create_new():
    mock_client = MagicMock()
    mock_client.get_experiment_by_name.return_value = None
    mock_client.create_experiment.return_value = "exp_202"

    config = MLflowConfig(enabled=True, experiment_prefix="TestLab::")
    tracker = MLflowTracker(config=config, client=mock_client)

    exp_id = tracker.get_or_create_experiment("Vision Transformer", "ws_02")
    assert exp_id == "exp_202"
    mock_client.create_experiment.assert_called_once()
    args, kwargs = mock_client.create_experiment.call_args
    assert kwargs["name"] == "TestLab::Vision Transformer"
    assert kwargs["tags"]["researchlens.workspace_id"] == "ws_02"


def test_create_and_terminate_run():
    mock_client = MagicMock()
    mock_run = MagicMock()
    mock_run.info.run_id = "mlflow_run_abc"
    mock_client.create_run.return_value = mock_run

    tracker = MLflowTracker(client=mock_client)
    run_id = tracker.create_run("exp_202", "run_test_01", tags={"env": "local"})

    assert run_id == "mlflow_run_abc"
    mock_client.create_run.assert_called_once_with(
        experiment_id="exp_202",
        run_name="run_test_01",
        tags={"env": "local"},
    )

    tracker.terminate_run(run_id, status="FINISHED")
    mock_client.set_terminated.assert_called_once_with(run_id="mlflow_run_abc", status="FINISHED")
