"""MLflow artifact synchronization and size-constrained upload manager (Phase 13)."""

import os
from typing import Any, Dict, List, Optional

from researchlens.execution.artifacts import calculate_sha256
from researchlens.mlops.client import get_mlflow_client
from researchlens.mlops.configuration import MLflowConfig, get_mlflow_config
from researchlens.utils.logging import get_logger
from researchlens.workspace.schema import ExperimentArtifact

logger = get_logger("mlops_artifacts")


class MLflowArtifactManager:
    """Manages secure artifact logging to MLflow with size thresholds and SHA-256 pairing."""

    def __init__(
        self,
        config: Optional[MLflowConfig] = None,
        client: Optional[Any] = None,
    ) -> None:
        self.config = config or get_mlflow_config()
        self.client = client or get_mlflow_client(self.config)

    def log_artifact(
        self,
        mlflow_run_id: str,
        local_path: str,
        artifact_path: Optional[str] = None,
    ) -> Optional[str]:
        """Validate and log a single local artifact file to an MLflow run."""
        if self.client is None or not os.path.exists(local_path):
            return None

        # Check path traversal and safety
        norm_path = os.path.normpath(local_path)
        if ".." in norm_path:
            logger.warning(f"Path traversal detected in artifact path: '{local_path}'. Skipping upload.")
            return None

        try:
            file_size_mb = os.path.getsize(norm_path) / (1024 * 1024)
            if file_size_mb > self.config.max_artifact_size_mb:
                logger.warning(
                    f"Artifact '{norm_path}' ({file_size_mb:.2f} MB) exceeds maximum allowed size "
                    f"({self.config.max_artifact_size_mb} MB). Skipping MLflow upload."
                )
                return None

            self.client.log_artifact(
                run_id=mlflow_run_id,
                local_path=norm_path,
                artifact_path=artifact_path,
            )

            # Construct canonical MLflow artifact URI
            basename = os.path.basename(norm_path)
            dest = f"{artifact_path}/{basename}" if artifact_path else basename
            return f"runs:/{mlflow_run_id}/{dest}"
        except Exception as e:
            logger.warning(f"Failed to log artifact '{local_path}' to MLflow run {mlflow_run_id}: {e}")
            return None

    def sync_artifacts(
        self,
        mlflow_run_id: str,
        artifacts: List[ExperimentArtifact],
    ) -> Dict[str, str]:
        """Upload registered artifacts and return mapping from artifact_id to MLflow artifact URI."""
        uri_mapping: Dict[str, str] = {}
        for art in artifacts:
            if not art.path or not os.path.exists(art.path):
                continue

            # Ensure SHA-256 is recorded
            if not art.sha256:
                art.sha256 = calculate_sha256(art.path)

            art_uri = self.log_artifact(mlflow_run_id, art.path)
            if art_uri:
                uri_mapping[art.artifact_id] = art_uri

        return uri_mapping
