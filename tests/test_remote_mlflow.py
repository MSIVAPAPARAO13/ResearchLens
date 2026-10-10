"""Tests for Remote MLflow Tracking & Configuration Validation (Phase 14)."""

from researchlens.mlops.client import validate_mlflow_configuration
from researchlens.mlops.configuration import MLflowConfig


def test_mlflow_validation_disabled():
    """Verify DISABLED status when enabled=False."""
    cfg = MLflowConfig(enabled=False, tracking_uri="./mlruns")
    res = validate_mlflow_configuration(cfg)
    assert res["status"] == "DISABLED"


def test_mlflow_validation_misconfigured_uri():
    """Verify MISCONFIGURED status for malformed URIs."""
    # Malformed HTTP URI with no netloc
    cfg_bad_http = MLflowConfig(enabled=True, tracking_uri="http://")
    res = validate_mlflow_configuration(cfg_bad_http)
    assert res["status"] == "MISCONFIGURED"
    assert "Invalid tracking URI" in res["error"]

    # Malformed artifact URI
    cfg_bad_art = MLflowConfig(enabled=True, tracking_uri="./mlruns", artifact_location="invalid://?#%")
    res_art = validate_mlflow_configuration(cfg_bad_art)
    assert res_art["status"] == "MISCONFIGURED"


def test_mlflow_validation_local_connected_or_unavailable():
    """Verify local filesystem tracking validation."""
    cfg = MLflowConfig(enabled=True, tracking_uri="./mlruns")
    res = validate_mlflow_configuration(cfg)
    # Either CONNECTED or UNAVAILABLE (if mlflow not importable)
    assert res["status"] in ["CONNECTED", "UNAVAILABLE"]


def test_mlflow_remote_http_uri_support():
    """Verify HTTP tracking URI validation."""
    cfg = MLflowConfig(enabled=True, tracking_uri="http://127.0.0.1:54321")
    res = validate_mlflow_configuration(cfg)
    # It attempts to connect, returning UNAVAILABLE if host is unreachable, never crashing
    assert res["status"] in ["CONNECTED", "UNAVAILABLE"]
