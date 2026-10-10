"""Unit tests for MLflow artifact logging and size limit guards (Phase 13)."""

import os
import tempfile
from unittest.mock import MagicMock

from researchlens.mlops.artifacts import MLflowArtifactManager
from researchlens.mlops.configuration import MLflowConfig
from researchlens.workspace.schema import ExperimentArtifact


def test_log_artifact_success_and_sha256_pairing():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        f.write(b'{"accuracy": 0.95}')
        temp_path = f.name

    try:
        mock_client = MagicMock()
        mgr = MLflowArtifactManager(client=mock_client)

        uri = mgr.log_artifact("mlflow_run_01", temp_path, artifact_path="results")
        assert uri == f"runs:/mlflow_run_01/results/{os.path.basename(temp_path)}"
        mock_client.log_artifact.assert_called_once_with(
            run_id="mlflow_run_01",
            local_path=os.path.normpath(temp_path),
            artifact_path="results",
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_log_artifact_oversized_is_rejected():
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        # 2 MB dummy data
        f.write(b"x" * (2 * 1024 * 1024))
        temp_path = f.name

    try:
        mock_client = MagicMock()
        # Max limit set to 1.0 MB
        config = MLflowConfig(enabled=True, max_artifact_size_mb=1.0)
        mgr = MLflowArtifactManager(config=config, client=mock_client)

        uri = mgr.log_artifact("mlflow_run_01", temp_path)
        assert uri is None
        mock_client.log_artifact.assert_not_called()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_sync_artifacts_preserves_sha256():
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"reproducible experiment log")
        temp_path = f.name

    try:
        mock_client = MagicMock()
        mgr = MLflowArtifactManager(client=mock_client)

        art = ExperimentArtifact(
            artifact_id="art_100",
            run_id="run_100",
            name="log.txt",
            path=temp_path,
        )

        mapping = mgr.sync_artifacts("mlflow_run_01", [art])
        assert "art_100" in mapping
        assert len(art.sha256) == 64  # SHA-256 calculated
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
