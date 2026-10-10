"""MLflow experiment tracking implementation for runs, metrics, and parameters (Phase 13)."""

import time
from typing import Any, Dict, Optional

from researchlens.mlops.client import get_mlflow_client
from researchlens.mlops.configuration import MLflowConfig, get_mlflow_config
from researchlens.mlops.models import sanitize_parameters
from researchlens.utils.logging import get_logger
from researchlens.workspace.experiment_service import normalize_and_validate_metric

logger = get_logger("mlops_tracking")


class MLflowTracker:
    """Orchestrates MLflow experiment setup, run tracking, metrics, and parameter logging."""

    def __init__(
        self,
        config: Optional[MLflowConfig] = None,
        client: Optional[Any] = None,
    ) -> None:
        self.config = config or get_mlflow_config()
        self.client = client or get_mlflow_client(self.config)

    def get_or_create_experiment(self, workspace_name: str, workspace_id: str) -> Optional[str]:
        """Resolve or create an MLflow experiment deterministically mapped from a ResearchLens workspace."""
        if self.client is None:
            return None

        exp_name = f"{self.config.experiment_prefix}{workspace_name}"
        try:
            exp = self.client.get_experiment_by_name(exp_name)
            if exp:
                return exp.experiment_id

            tags = {
                "researchlens.workspace_id": workspace_id,
                "researchlens.workspace_name": workspace_name,
            }
            return self.client.create_experiment(
                name=exp_name,
                artifact_location=self.config.artifact_location,
                tags=tags,
            )
        except Exception as e:
            logger.warning(f"Error getting/creating MLflow experiment '{exp_name}': {e}")
            raise

    def create_run(
        self,
        experiment_id: str,
        run_name: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """Start a new tracked MLflow run entity."""
        if self.client is None:
            return None

        try:
            run = self.client.create_run(
                experiment_id=experiment_id,
                run_name=run_name,
                tags=tags or {},
            )
            return run.info.run_id
        except Exception as e:
            logger.warning(f"Error creating MLflow run '{run_name}': {e}")
            raise

    def log_parameters(self, mlflow_run_id: str, parameters: Dict[str, Any]) -> None:
        """Sanitize parameters and log them to MLflow."""
        if self.client is None or not parameters:
            return

        clean_params = sanitize_parameters(parameters)
        for k, v in clean_params.items():
            try:
                self.client.log_param(mlflow_run_id, str(k), str(v))
            except Exception as e:
                logger.debug(f"Failed to log param '{k}' to MLflow run {mlflow_run_id}: {e}")

    def log_metrics(
        self,
        mlflow_run_id: str,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        timestamp: Optional[int] = None,
    ) -> None:
        """Validate and log empirical metrics to MLflow."""
        if self.client is None or not metrics:
            return

        ts = timestamp or int(time.time() * 1000)
        for metric_name, raw_val in metrics.items():
            is_valid, norm_val, _ = normalize_and_validate_metric(metric_name, raw_val)
            if is_valid:
                try:
                    self.client.log_metric(
                        run_id=mlflow_run_id,
                        key=metric_name,
                        value=float(norm_val),
                        timestamp=ts,
                        step=step or 0,
                    )
                except Exception as e:
                    logger.debug(f"Failed to log metric '{metric_name}' to MLflow: {e}")

    def log_tags(self, mlflow_run_id: str, tags: Dict[str, str]) -> None:
        """Attach metadata tags to an existing MLflow run."""
        if self.client is None or not tags:
            return

        for k, v in tags.items():
            if v is not None:
                try:
                    self.client.set_tag(mlflow_run_id, str(k), str(v))
                except Exception as e:
                    logger.debug(f"Failed to set tag '{k}' on MLflow run {mlflow_run_id}: {e}")

    def terminate_run(
        self,
        mlflow_run_id: str,
        status: str = "FINISHED",  # FINISHED, FAILED, KILLED
    ) -> None:
        """Update terminal status of MLflow run."""
        if self.client is None:
            return

        try:
            self.client.set_terminated(run_id=mlflow_run_id, status=status)
        except Exception as e:
            logger.warning(f"Failed to terminate MLflow run {mlflow_run_id} with status {status}: {e}")
