"""ResearchLens ↔ MLflow synchronization service with bounded retries and audit logging (Phase 13)."""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

from researchlens.mlops.artifacts import MLflowArtifactManager
from researchlens.mlops.client import get_mlflow_client
from researchlens.mlops.configuration import MLflowConfig, get_mlflow_config
from researchlens.mlops.models import build_mlflow_tags
from researchlens.mlops.tracking import MLflowTracker
from researchlens.utils.logging import get_logger
from researchlens.workspace.schema import (
    ExperimentArtifact,
    ExperimentPlan,
    ExperimentResult,
    ExperimentRun,
)
from researchlens.workspace.store import WorkspaceStore

logger = get_logger("mlops_sync")


class MLflowSyncService:
    """Coordinates resilient bidirectional synchronization between ResearchLens runs and MLflow."""

    def __init__(
        self,
        store: WorkspaceStore,
        config: Optional[MLflowConfig] = None,
        client: Optional[Any] = None,
    ) -> None:
        self.store = store
        self.config = config or get_mlflow_config()
        self.client = client or get_mlflow_client(self.config)
        self.tracker = MLflowTracker(self.config, self.client)
        self.artifact_manager = MLflowArtifactManager(self.config, self.client)

    def sync_run(
        self,
        workspace_id: str,
        run: ExperimentRun,
        experiment: Optional[ExperimentPlan] = None,
        results: Optional[List[ExperimentResult]] = None,
        artifacts: Optional[List[ExperimentArtifact]] = None,
        manifest: Optional[Dict[str, Any]] = None,
        actor_id: str = "system",
    ) -> Tuple[bool, Optional[str]]:
        """Synchronize a completed or failed ResearchLens run with MLflow using bounded retries."""
        if not self.config.enabled:
            run.mlflow_sync_status = "DISABLED"
            self.store.save_run(run, workspace_id)
            return True, None

        if self.client is None:
            run.mlflow_sync_status = "FAILED"
            run.mlflow_sync_error = "MLflow client unavailable."
            self.store.save_run(run, workspace_id)
            self.store.log_audit(
                workspace_id,
                actor_id,
                "MLFLOW_SYNC_FAILED",
                "RUN",
                run.run_id,
                {"reason": "MLflow client unavailable"},
            )
            return False, "MLflow client unavailable."

        ws = self.store.get_workspace(workspace_id)
        ws_name = ws.name if ws else workspace_id

        last_error: Optional[str] = None
        for attempt in range(1, self.config.retry_attempts + 1):
            try:
                # 1. Resolve or create MLflow experiment
                mlflow_exp_id = self.tracker.get_or_create_experiment(ws_name, workspace_id)
                if not mlflow_exp_id:
                    raise RuntimeError(f"Failed to resolve MLflow experiment for workspace '{ws_name}'")

                # 2. Construct tags
                tags = build_mlflow_tags(workspace_id, run, experiment)
                run_name = f"{run.run_id}_{experiment.name if experiment else 'run'}"

                # 3. Create run
                mlflow_run_id = self.tracker.create_run(
                    experiment_id=mlflow_exp_id,
                    run_name=run_name,
                    tags=tags,
                )
                if not mlflow_run_id:
                    raise RuntimeError("Failed to create MLflow run entity.")

                run.mlflow_run_id = mlflow_run_id
                run.mlflow_experiment_id = mlflow_exp_id

                self.store.log_audit(
                    workspace_id,
                    actor_id,
                    "MLFLOW_RUN_CREATED",
                    "RUN",
                    run.run_id,
                    {"mlflow_run_id": mlflow_run_id, "mlflow_exp_id": mlflow_exp_id},
                )

                # 4. Log parameters
                params_to_log: Dict[str, Any] = {}
                if experiment and experiment.configuration:
                    params_to_log.update(experiment.configuration)
                if run.seed is not None:
                    params_to_log["seed"] = run.seed
                if experiment and experiment.timeout_seconds:
                    params_to_log["timeout_seconds"] = experiment.timeout_seconds
                if params_to_log:
                    self.tracker.log_parameters(mlflow_run_id, params_to_log)

                # 5. Log metrics
                if results:
                    metrics_dict: Dict[str, float] = {}
                    for r in results:
                        metrics_dict[r.metric] = float(r.value)
                    self.tracker.log_metrics(mlflow_run_id, metrics_dict)

                # 6. Log artifacts (with size filtering and SHA-256 pairing)
                if artifacts:
                    art_uris = self.artifact_manager.sync_artifacts(mlflow_run_id, artifacts)
                    if art_uris:
                        self.store.log_audit(
                            workspace_id,
                            actor_id,
                            "MLFLOW_ARTIFACT_LOGGED",
                            "RUN",
                            run.run_id,
                            {"count": len(art_uris)},
                        )

                # 7. Log canonical reproducibility manifest if present
                run_dir = self.store.get_run_dir(workspace_id, run.run_id)
                manifest_path = os.path.join(run_dir, "reports", "reproducibility.json")
                if not os.path.exists(manifest_path):
                    manifest_path = os.path.join(run_dir, "reproducibility.json")
                if os.path.exists(manifest_path):
                    self.artifact_manager.log_artifact(mlflow_run_id, manifest_path, artifact_path="reports")

                # 8. Terminate MLflow run matching ResearchLens status
                if run.status == "COMPLETED":
                    term_status = "FINISHED"
                elif run.status == "TIMEOUT":
                    term_status = "KILLED"
                    self.tracker.log_tags(mlflow_run_id, {"execution.timeout": "true"})
                else:
                    term_status = "FAILED"
                    if run.failure_category:
                        self.tracker.log_tags(mlflow_run_id, {"failure.category": run.failure_category})

                self.tracker.terminate_run(mlflow_run_id, status=term_status)

                run.mlflow_sync_status = "SYNCED"
                run.mlflow_sync_error = None
                self.store.save_run(run, workspace_id)

                self.store.log_audit(
                    workspace_id,
                    actor_id,
                    "MLFLOW_RUN_SYNCED",
                    "RUN",
                    run.run_id,
                    {"mlflow_run_id": mlflow_run_id, "status": term_status},
                )
                return True, None

            except Exception as e:
                last_error = str(e)
                logger.warning(f"MLflow sync attempt {attempt}/{self.config.retry_attempts} failed: {e}")
                if attempt < self.config.retry_attempts:
                    time.sleep(self.config.retry_backoff_factor * (2 ** (attempt - 1)))

        # Retries exhausted
        run.mlflow_sync_status = "FAILED"
        run.mlflow_sync_error = last_error
        self.store.save_run(run, workspace_id)

        self.store.log_audit(
            workspace_id,
            actor_id,
            "MLFLOW_SYNC_FAILED",
            "RUN",
            run.run_id,
            {"attempts": self.config.retry_attempts, "error": last_error},
        )
        return False, last_error

    def retry_sync(self, workspace_id: str, run_id: str, actor_id: str = "researcher") -> Tuple[bool, Optional[str]]:
        """Retry synchronization for a previously failed or pending run."""
        run = self.store.get_run(workspace_id, run_id)
        if not run:
            return False, f"Run '{run_id}' not found."

        experiment = self.store.get_experiment(workspace_id, run.experiment_id)
        results = self.store.get_results_for_run(workspace_id, run_id)
        artifacts = [self.store.get_artifact(workspace_id, a_id) for a_id in run.artifact_ids]
        valid_arts = [a for a in artifacts if a is not None]

        return self.sync_run(
            workspace_id=workspace_id,
            run=run,
            experiment=experiment,
            results=results,
            artifacts=valid_arts,
            actor_id=actor_id,
        )
