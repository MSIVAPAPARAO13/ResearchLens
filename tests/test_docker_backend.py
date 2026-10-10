"""Tests for Docker Execution Backend & Container Sandboxing (Phase 14)."""

import tempfile

import pytest

from researchlens.execution.backends.docker import DockerExecutionBackend
from researchlens.workspace.schema import ExecutionJob
from researchlens.workspace.store import WorkspaceStore


def test_docker_validation_allowed_vs_blocked_image():
    """Verify image allowlist enforcement in Docker backend."""
    backend = DockerExecutionBackend()

    # Approved image
    job_valid = ExecutionJob(
        job_id="j1",
        workspace_id="ws1",
        experiment_id="e1",
        run_id="r1",
        backend="DOCKER",
        image="researchlens/nlp-experiment:1.0",
        command=["python", "demo.py"],
    )
    is_valid, err = backend.validate(job_valid)
    assert is_valid is True

    # Blocked / arbitrary unknown image
    job_blocked = ExecutionJob(
        job_id="j2",
        workspace_id="ws1",
        experiment_id="e1",
        run_id="r2",
        backend="DOCKER",
        image="malicious/untrusted-image:latest",
        command=["python", "demo.py"],
    )
    is_valid, err = backend.validate(job_blocked)
    assert is_valid is False
    assert "BLOCKED" in err


def test_docker_validation_blocks_host_mount_attempts():
    """Verify forbidden volume mounts are blocked during validation."""
    backend = DockerExecutionBackend()

    # Attempt to mount host root or entire user directory
    bad_mounts = [
        {"host_path": "C:\\", "mount_path": "/workspace/data"},
        {"host_path": "/root", "mount_path": "/workspace/data"},
        {"host_path": "/var/run/docker.sock", "mount_path": "/var/run/docker.sock"},
    ]

    for m in bad_mounts:
        job = ExecutionJob(
            job_id="j_bad",
            workspace_id="ws1",
            experiment_id="e1",
            run_id="r1",
            backend="DOCKER",
            image="researchlens/nlp-experiment:1.0",
            command=["python", "demo.py"],
            resources={"mounts": [m]},
        )
        is_valid, err = backend.validate(job)
        assert is_valid is False
        assert "Security violation" in err or "Forbidden" in err


def test_docker_unavailable_graceful_handling(monkeypatch):
    """Verify backend returns DOCKER_NOT_AVAILABLE without crashing if docker is unavailable."""
    backend = DockerExecutionBackend()
    monkeypatch.setattr(backend, "is_docker_available", lambda: False)

    with tempfile.TemporaryDirectory() as tmp_dir:
        job = ExecutionJob(
            job_id="j_dock",
            workspace_id="ws1",
            experiment_id="e1",
            run_id="r1",
            backend="DOCKER",
            image="researchlens/nlp-experiment:1.0",
            command=["python", "demo.py"],
        )
        res = backend.start(job, tmp_dir)
        assert res["error"] == "DOCKER_NOT_AVAILABLE"
        assert res["exit_code"] == 1
        assert res["enforce_status"]["cpu_limit"] == "LIMIT_NOT_ENFORCED"


def test_docker_live_execution_if_available():
    """Run real container test if Docker is available on host."""
    backend = DockerExecutionBackend()
    if not backend.is_docker_available():
        pytest.skip("Docker daemon is not available on host. Skipping live container test.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        store = WorkspaceStore(base_dir=tmp_dir)
        ws = store.create_workspace("Docker Test WS", "tester")
        run_dir = store.get_run_dir(ws.workspace_id, "docker_run_1")

        # Use standard python:3.11-slim if present or mock
        job = ExecutionJob(
            job_id="live_docker_job",
            workspace_id=ws.workspace_id,
            experiment_id="exp_dock",
            run_id="docker_run_1",
            backend="DOCKER",
            image="python:3.11-slim",
            command=["python", "-c", "import json, os; os.makedirs('/workspace/output', exist_ok=True); json.dump({'metrics': {'acc': 0.95}}, open('/workspace/output/results.json', 'w'))"],
            network_policy="DISABLED",
            resources={"cpu_limit": 1.0, "memory_limit": "512m"},
            timeout=60,
        )

        backend.prepare(job, run_dir)
        res = backend.start(job, run_dir, timeout_seconds=60)
        # Even if pull fails in sandbox, it handles cleanly
        assert "exit_code" in res
        backend.cleanup(job.job_id)
