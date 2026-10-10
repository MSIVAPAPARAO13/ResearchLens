"""Unit tests for MLflow synchronization service and bidirectional referencing (Phase 13)."""

import os
import shutil
import tempfile
from unittest.mock import MagicMock

from researchlens.mlops.configuration import MLflowConfig
from researchlens.mlops.sync import MLflowSyncService
from researchlens.workspace.schema import (
    ExperimentPlan,
    ExperimentResult,
    ExperimentRun,
)
from researchlens.workspace.store import WorkspaceStore


def test_mlflow_sync_run_success():
    temp_dir = tempfile.mkdtemp()
    try:
        store = WorkspaceStore(base_dir=os.path.join(temp_dir, "workspaces"))
        ws = store.create_workspace("Test Sync Workspace")
        ws_id = ws.workspace_id

        plan = ExperimentPlan("exp_sync_01", name="Sync Plan", dataset="SQuAD_v2", model="BERT")
        store.save_experiment(plan, ws_id)

        run = ExperimentRun(
            run_id="run_sync_01",
            experiment_id=plan.experiment_id,
            status="COMPLETED",
            seed=42,
            git_commit_sha="abc1234",
        )
        store.save_run(run, ws_id)

        res = ExperimentResult(
            result_id="res_01",
            run_id=run.run_id,
            model="BERT",
            dataset="SQuAD_v2",
            metric="f1",
            value=0.91,
        )

        mock_client = MagicMock()
        mock_exp = MagicMock()
        mock_exp.experiment_id = "exp_mlflow_99"
        mock_client.get_experiment_by_name.return_value = mock_exp

        mock_run = MagicMock()
        mock_run.info.run_id = "mlflow_run_77"
        mock_client.create_run.return_value = mock_run

        config = MLflowConfig(enabled=True)
        sync_svc = MLflowSyncService(store=store, config=config, client=mock_client)

        ok, err = sync_svc.sync_run(
            workspace_id=ws_id,
            run=run,
            experiment=plan,
            results=[res],
            artifacts=[],
        )

        assert ok is True
        assert err is None
        assert run.mlflow_run_id == "mlflow_run_77"
        assert run.mlflow_experiment_id == "exp_mlflow_99"
        assert run.mlflow_sync_status == "SYNCED"

        # Verify audit logs
        audit_events = store.list_audit(ws_id)
        actions = [a.action for a in audit_events]
        assert "MLFLOW_RUN_CREATED" in actions
        assert "MLFLOW_RUN_SYNCED" in actions

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
