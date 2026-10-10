"""MLflow configuration management for ResearchLens (Phase 13)."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class MLflowConfig:
    """Configuration settings for MLflow experiment tracking and model registry integration."""

    enabled: bool = False
    tracking_uri: str = "./mlruns"
    experiment_prefix: str = "ResearchLens::"
    artifact_location: Optional[str] = None
    auto_logging: bool = True
    max_artifact_size_mb: float = 50.0
    retry_attempts: int = 3
    retry_backoff_factor: float = 0.5


def get_mlflow_config() -> MLflowConfig:
    """Retrieve MLflow settings populated from environment variables with safe defaults."""
    env_enabled = os.getenv("RESEARCHLENS_MLFLOW_ENABLED", os.getenv("MLFLOW_ENABLED", "false")).lower() in (
        "true",
        "1",
        "yes",
    )
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", os.getenv("RESEARCHLENS_MLFLOW_TRACKING_URI", "./mlruns"))
    prefix = os.getenv("RESEARCHLENS_MLFLOW_EXPERIMENT_PREFIX", "ResearchLens::")
    art_loc = os.getenv("MLFLOW_ARTIFACT_LOCATION")
    auto_log = os.getenv("RESEARCHLENS_MLFLOW_AUTO_LOGGING", "true").lower() in ("true", "1", "yes")

    try:
        max_art_mb = float(os.getenv("MAX_MLFLOW_ARTIFACT_SIZE_MB", "50.0"))
    except ValueError:
        max_art_mb = 50.0

    try:
        retries = int(os.getenv("MLFLOW_RETRY_ATTEMPTS", "3"))
    except ValueError:
        retries = 3

    return MLflowConfig(
        enabled=env_enabled,
        tracking_uri=tracking_uri,
        experiment_prefix=prefix,
        artifact_location=art_loc,
        auto_logging=auto_log,
        max_artifact_size_mb=max_art_mb,
        retry_attempts=retries,
    )
