"""Tests for Resource Limits Enforcement & Reporting (Phase 14)."""

import tempfile

from researchlens.execution.backends.docker import DockerExecutionBackend
from researchlens.execution.backends.local import LocalExecutionBackend
from researchlens.workspace.schema import ExecutionJob


def test_local_resource_limits_return_limit_not_enforced():
    """Verify local backend accurately reports LIMIT_NOT_ENFORCED for CPU/memory rather than fabricating enforcement."""
    backend = LocalExecutionBackend()
    with tempfile.TemporaryDirectory() as tmp_dir:
        job = ExecutionJob(
            job_id="j_local_lim",
            workspace_id="ws1",
            experiment_id="e1",
            run_id="r1",
            command=["python", "--version"],
            resources={"cpu_limit": 2, "memory_limit": "4GB"},
        )
        res = backend.start(job, tmp_dir)
        enforce = res["enforce_status"]
        assert enforce["cpu_limit"] == "LIMIT_NOT_ENFORCED"
        assert enforce["memory_limit"] == "LIMIT_NOT_ENFORCED"
        assert enforce["timeout"] == "ENFORCED"


def test_docker_resource_limits_unavailability_reporting(monkeypatch):
    """Verify docker backend returns LIMIT_NOT_ENFORCED when docker daemon is unavailable."""
    backend = DockerExecutionBackend()
    monkeypatch.setattr(backend, "is_docker_available", lambda: False)

    with tempfile.TemporaryDirectory() as tmp_dir:
        job = ExecutionJob(
            job_id="j_dock_lim",
            workspace_id="ws1",
            experiment_id="e1",
            run_id="r1",
            resources={"cpu_limit": 2, "memory_limit": "4GB"},
        )
        res = backend.start(job, tmp_dir)
        enforce = res["enforce_status"]
        assert enforce["cpu_limit"] == "LIMIT_NOT_ENFORCED"
        assert enforce["memory_limit"] == "LIMIT_NOT_ENFORCED"
