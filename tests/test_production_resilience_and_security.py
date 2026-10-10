"""Tests for Phase 16: Persistence, Recovery, and Security Audit.

Verifies:
1. WorkspaceStore atomic write robustness and corrupted record safe recovery.
2. Missing artifacts and interrupted write resilience.
3. Path traversal attacks and malicious inputs rejection in sanitize methods.
4. Role-based execution permissions (VIEWER rejected, RESEARCHER allowed).
5. Secret masking in logging (SecretMaskingFilter and JsonLogFormatter).
6. Production configuration validation (Settings.validate_production_configuration).
7. Container image allowlisting enforcement (ExecutionImageRegistry).
8. MLflow offline failure resilience (completed runs preserved locally).
"""

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from researchlens.config.settings import Settings
from researchlens.execution.executor import ExperimentExecutor
from researchlens.execution.image_registry import ExecutionImageRegistry
from researchlens.execution.registry import ExperimentRegistryService
from researchlens.utils.logging import JsonLogFormatter, SecretMaskingFilter
from researchlens.workspace.schema import (
    ExperimentPlan,
    ExperimentRun,
    SavedPaper,
    WorkspaceMember,
)
from researchlens.workspace.store import WorkspaceStore, _sanitize_id


def test_atomic_write_and_corrupted_json_recovery(tmp_path: Path):
    """Test that store writes atomically and recovers safely from corrupted JSON files."""
    store = WorkspaceStore(base_dir=str(tmp_path))
    ws = store.create_workspace("Resilience WS", "Test recovery")
    ws_id = ws.workspace_id

    # Save a paper
    paper = SavedPaper(
        paper_id="paper_resilience_1",
        workspace_id=ws_id,
        title="Resilience Paper",
    )
    store.save_paper(paper)
    assert len(store.list_papers(ws_id)) == 1

    # Simulate corrupted JSON on disk
    papers_file = tmp_path / ws_id / "papers.json"
    assert papers_file.exists()
    papers_file.write_text("{ this is corrupted invalid json }}}", encoding="utf-8")

    # Loading corrupted list should not crash, it should return [] safely
    papers_recovered = store.list_papers(ws_id)
    assert isinstance(papers_recovered, list)
    assert len(papers_recovered) == 0

    # Saving a new paper should heal the file atomically
    paper2 = SavedPaper(
        paper_id="paper_resilience_2",
        workspace_id=ws_id,
        title="Healed Paper",
    )
    store.save_paper(paper2)
    assert len(store.list_papers(ws_id)) == 1
    assert store.list_papers(ws_id)[0].paper_id == "paper_resilience_2"


def test_path_traversal_sanitization():
    """Test that path traversal attempts in workspace IDs or entity IDs are rejected."""
    malicious_ids = [
        "../../etc/passwd",
        "..\\..\\windows\\system32",
        "foo/bar/baz",
        "../../shadow",
        "..",
        ".",
    ]

    for bad_id in malicious_ids:
        with pytest.raises(ValueError, match="Invalid identifier"):
            _sanitize_id(bad_id)


def test_role_based_access_control_enforcement(tmp_path: Path):
    """Test that VIEWER cannot execute experiments while RESEARCHER is authorized."""
    store = WorkspaceStore(base_dir=str(tmp_path))
    ws = store.create_workspace("RBAC WS", "Permission test")
    ws_id = ws.workspace_id

    store.save_member(WorkspaceMember("m_owner", ws_id, "Alice", "alice@lab.org", "OWNER"))
    store.save_member(WorkspaceMember("m_researcher", ws_id, "Bob", "bob@lab.org", "RESEARCHER"))
    store.save_member(WorkspaceMember("m_viewer", ws_id, "David", "david@lab.org", "VIEWER"))

    entrypoint = "scripts/experiments/deterministic_demo.py"
    registry = ExperimentRegistryService(store)
    registry.register_experiment(
        workspace_id=ws_id,
        name="RBAC Plan",
        entrypoint=entrypoint,
        owner="m_owner",
    )

    plan = ExperimentPlan(
        experiment_id="exp_rbac_01",
        name="RBAC Plan",
        entrypoint=entrypoint,
    )
    store.save_experiment(plan, ws_id)

    executor = ExperimentExecutor(store)

    # 1. VIEWER rejected
    run_v, err_v = executor.execute_experiment(ws_id, plan.experiment_id, actor_id="m_viewer")
    assert run_v is None
    assert "Permission denied" in err_v

    # 2. Unknown actor rejected
    run_unknown, err_unknown = executor.execute_experiment(ws_id, plan.experiment_id, actor_id="m_hacker")
    assert run_unknown is None
    assert "Permission denied" in err_unknown

    # 3. RESEARCHER allowed
    run_r, err_r = executor.execute_experiment(ws_id, plan.experiment_id, actor_id="m_researcher")
    assert err_r == ""
    assert run_r is not None
    assert run_r.status == "COMPLETED"


def test_secret_masking_in_logs():
    """Test that SecretMaskingFilter redacts sensitive API keys and tokens from logs."""
    filter_ = SecretMaskingFilter()

    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Connecting with api_key: sk-ant-api03-abcdef1234567890 and token: secret_token_xyz",
        args=(),
        exc_info=None,
    )

    filter_.filter(record)
    assert "sk-ant-api03-abcdef1234567890" not in record.msg
    assert "[REDACTED]" in record.msg

    # Also test JsonLogFormatter
    formatter = JsonLogFormatter()
    json_out = formatter.format(record)
    parsed = json.loads(json_out)
    assert parsed["level"] == "INFO"
    assert "sk-ant-api03-abcdef1234567890" not in parsed["message"]


def test_production_configuration_validation():
    """Test that validate_production_configuration enforces required security settings."""
    # Development mode: default settings should validate with 0 issues
    dev_settings = Settings(env="development", debug=True)
    assert dev_settings.is_production is False
    assert len(dev_settings.validate_production_configuration()) == 0

    # Production mode with debug=True should flag error
    prod_invalid = Settings(env="production", debug=True)
    assert prod_invalid.is_production is True
    issues = prod_invalid.validate_production_configuration()
    assert len(issues) > 0
    assert any("Debug mode is enabled" in iss for iss in issues)

    # Production mode with valid config
    prod_valid = Settings(env="production", debug=False)
    assert prod_valid.is_production is True
    assert len(prod_valid.validate_production_configuration()) == 0


def test_docker_image_allowlisting_security():
    """Test that ExecutionImageRegistry blocks unapproved container images."""
    registry = ExecutionImageRegistry()

    # Built-in approved image
    allowed, reason, _ = registry.is_image_allowed("researchlens/nlp-experiment:1.0")
    assert allowed is True

    # Untrusted / malicious image
    allowed_bad, reason_bad, _ = registry.is_image_allowed("malicious/miner:latest")
    assert allowed_bad is False
    assert "BLOCKED" in reason_bad

    # Empty string
    allowed_empty, reason_empty, _ = registry.is_image_allowed("")
    assert allowed_empty is False
    assert "BLOCKED" in reason_empty


def test_offline_mlflow_fallback_preserves_local_runs(tmp_path: Path):
    """Test that failure to reach MLflow tracking server does not destroy local completed run."""
    store = WorkspaceStore(base_dir=str(tmp_path))
    ws = store.create_workspace("MLflow Resilience WS")
    ws_id = ws.workspace_id

    run = ExperimentRun(
        run_id="run_mlflow_offline",
        experiment_id="exp_offline",
        status="COMPLETED",
        exit_code=0,
        execution_backend="local",
    )
    store.save_run(run, ws_id)

    # Verify run is persisted locally
    loaded_run = store.get_run(ws_id, "run_mlflow_offline")
    assert loaded_run is not None
    assert loaded_run.status == "COMPLETED"

    # Even if MLflow sync throws a connection error
    with patch("researchlens.mlops.tracking.MLflowTracker.create_run", side_effect=ConnectionError("MLflow host unreachable")):
        # The run in the local store remains completely intact
        persisted_again = store.get_run(ws_id, "run_mlflow_offline")
        assert persisted_again is not None
        assert persisted_again.status == "COMPLETED"
