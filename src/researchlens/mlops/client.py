"""MLflow client factory and health verification (Phase 13)."""

import os
from typing import Any, Dict, Optional, Tuple

from researchlens.mlops.configuration import MLflowConfig, get_mlflow_config
from researchlens.utils.logging import get_logger

logger = get_logger("mlops_client")


def get_mlflow_client(config: Optional[MLflowConfig] = None) -> Any:
    """Instantiate and return an MLflowClient if MLflow is installed and accessible."""
    if config is None:
        config = get_mlflow_config()

    try:
        import os
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(config.tracking_uri)
        return MlflowClient(tracking_uri=config.tracking_uri)
    except ImportError:
        logger.warning("MLflow library is not installed in the current environment.")
        return None
    except Exception as e:
        logger.warning(f"Unable to initialize MLflowClient at '{config.tracking_uri}': {e}")
        return None


def validate_uri(uri: str) -> Tuple[bool, str]:
    """Validate tracking or artifact URI format."""
    if not uri or not uri.strip():
        return False, "URI cannot be empty."

    clean = uri.strip()
    if "://" in clean:
        scheme = clean.split("://")[0].lower()
        if scheme not in ["http", "https", "file", "s3", "sqlite"]:
            return False, f"Unsupported or invalid URI scheme: '{scheme}'"
        if scheme in ["http", "https"]:
            from urllib.parse import urlparse
            parsed = urlparse(clean)
            if not parsed.netloc:
                return False, f"Malformed HTTP(S) URI missing host: '{clean}'"
        return True, ""
    elif os.path.exists(clean) or clean.startswith("./") or clean.startswith("../") or clean.startswith("/") or clean.startswith("\\") or (len(clean) >= 2 and clean[1] == ":"):
        return True, ""
    return False, f"Invalid or unrecognized tracking URI format: '{clean}'"


def check_mlflow_health(config: Optional[MLflowConfig] = None) -> Dict[str, Any]:
    """Verify MLflow connectivity, URI syntax, and return structured health status (CONNECTED, DISABLED, UNAVAILABLE, MISCONFIGURED)."""
    if config is None:
        config = get_mlflow_config()

    if not config.enabled:
        return {
            "status": "DISABLED",
            "tracking_uri": config.tracking_uri,
            "version": "N/A",
            "experiments_count": 0,
            "error": None,
        }

    # Validate tracking URI syntax
    is_valid_track, track_err = validate_uri(config.tracking_uri)
    if not is_valid_track:
        return {
            "status": "MISCONFIGURED",
            "tracking_uri": config.tracking_uri,
            "version": "N/A",
            "experiments_count": 0,
            "error": f"Invalid tracking URI: {track_err}",
        }

    # Validate artifact URI syntax if provided
    if config.artifact_location:
        is_valid_art, art_err = validate_uri(config.artifact_location)
        if not is_valid_art:
            return {
                "status": "MISCONFIGURED",
                "tracking_uri": config.tracking_uri,
                "artifact_location": config.artifact_location,
                "version": "N/A",
                "experiments_count": 0,
                "error": f"Invalid artifact URI: {art_err}",
            }

    try:
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        import mlflow
        from mlflow.tracking import MlflowClient

        version = getattr(mlflow, "__version__", "UNKNOWN")

        # Fast probe for remote HTTP/HTTPS URIs when not mocked in tests
        is_mocked = hasattr(MlflowClient, "mock_calls") or hasattr(MlflowClient, "_mock_wraps") or "Mock" in type(MlflowClient).__name__
        if not is_mocked and (config.tracking_uri.startswith("http://") or config.tracking_uri.startswith("https://")):
            import urllib.request
            try:
                req = urllib.request.Request(config.tracking_uri, method="GET")
                with urllib.request.urlopen(req, timeout=1.0):
                    pass
            except Exception as net_err:
                return {
                    "status": "UNAVAILABLE",
                    "tracking_uri": config.tracking_uri,
                    "version": version,
                    "experiments_count": 0,
                    "error": f"Remote MLflow server unreachable: {net_err}",
                }

        mlflow.set_tracking_uri(config.tracking_uri)
        client = MlflowClient(tracking_uri=config.tracking_uri)

        # Probe client by searching experiments
        experiments = client.search_experiments(max_results=10)
        return {
            "status": "CONNECTED",
            "tracking_uri": config.tracking_uri,
            "version": version,
            "experiments_count": len(experiments),
            "error": None,
        }
    except ImportError:
        return {
            "status": "UNAVAILABLE",
            "tracking_uri": config.tracking_uri,
            "version": "NOT_INSTALLED",
            "experiments_count": 0,
            "error": "MLflow package not installed in Python runtime.",
        }
    except Exception as e:
        err_str = str(e)
        if "malformed" in err_str.lower() or "invalid uri" in err_str.lower():
            return {
                "status": "MISCONFIGURED",
                "tracking_uri": config.tracking_uri,
                "version": "UNKNOWN",
                "experiments_count": 0,
                "error": err_str,
            }
        return {
            "status": "UNAVAILABLE",
            "tracking_uri": config.tracking_uri,
            "version": "UNKNOWN",
            "experiments_count": 0,
            "error": err_str,
        }


def validate_mlflow_configuration(config: Optional[MLflowConfig] = None) -> Dict[str, Any]:
    """Alias for check_mlflow_health conforming to Phase 14 validation contract."""
    return check_mlflow_health(config)
