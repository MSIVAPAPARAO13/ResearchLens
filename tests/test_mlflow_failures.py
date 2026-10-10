"""Unit tests for MLflow synchronization failure handling and retry logic (Phase 13)."""

import os
import shutil
import tempfile
from unittest.mock import MagicMock

from researchlens.mlops.configuration import MLflowConfig
from researchlens.mlops.sync import MLflowSyncService
from researchlens.workspace.schema import (
    ExperimentPlan,
    ExperimentRun,
)
from researchlens.workspace.store import WorkspaceStore


def test_mlflow_sync_failure_preserves_local_run():
    temp_dir = tempfile.mkdtemp()
    try:
        store = WorkspaceStore(base_dir=os.path.join(temp_dir, "workspaces"))
        ws = store.create_workspace("Test Failure Workspace")
        ws_id = ws.workspace_id

        plan = ExperimentPlan("exp_fail_01", name="Fail Plan")
        store.save_experiment(plan, ws_id)

        run = ExperimentRun(
            run_id="run_fail_01",
            experiment_id=plan.experiment_id,
            status="COMPLETED",
        )
        store.save_run(run, ws_id)

        mock_client = MagicMock()
        mock_client.get_experiment_by_name.side_effect = ConnectionError("MLflow Tracking Server Down")

        config = MLflowConfig(enabled=True, retry_attempts=2, retry_backoff_factor=0.01)
        sync_svc = MLflowSyncService(store=store, config=config, client=mock_client)

        ok, err = sync_svc.sync_run(
            workspace_id=ws_id,
            run=run,
            experiment=plan,
        )

        assert ok is False
        assert "MLflow Tracking Server Down" in err
        assert run.status == "COMPLETED"  # Canonical ResearchLens run remains COMPLETED
        assert run.mlflow_sync_status == "FAILED"
        assert "MLflow Tracking Server Down" in run.mlflow_sync_error

        # Audit event MLFLOW_SYNC_FAILED must exist
        audit_events = store.list_audit(ws_id)
        assert any(a.action == "MLFLOW_SYNC_FAILED" for a in audit_events)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_mlflow_sync_retry_succeeds_on_second_attempt():
    temp_dir = tempfile.mkdtemp()
    try:
        store = WorkspaceStore(base_dir=os.path.join(temp_dir, "workspaces"))
        ws = store.create_workspace("Test Retry Workspace")
        ws_id = ws.workspace_id

        run = ExperimentRun(run_id="run_retry_01", experiment_id="exp_01", status="COMPLETED")
        store.save_run(run, ws_id)

        mock_client = MagicMock()
        mock_exp = MagicMock()
        mock_exp.experiment_id = "exp_ok_1"
        # First attempt fails, second attempt succeeds
        mock_client.get_experiment_by_name.side_effect = [RuntimeError("Temporary Glitch"), mock_exp]

        mock_run = MagicMock()
        mock_run.info.run_id = "mlflow_retry_run_id"
        mock_client.create_run.return_value = mock_run

        config = MLflowConfig(enabled=True, retry_attempts=2, retry_backoff_factor=0.01)
        sync_svc = MLflowSyncService(store=store, config=config, client=mock_client)

        ok, err = sync_svc.sync_run(workspace_id=ws_id, run=run)
        assert ok is True
        assert run.mlflow_sync_status == "SYNCED"
        assert run.mlflow_run_id == "mlflow_retry_run_id"

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
