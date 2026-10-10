"""Tests for Execution Backend Abstraction & Local Backend (Phase 14)."""

import os
import tempfile

from researchlens.execution.backends import (
    DockerExecutionBackend,
    LocalExecutionBackend,
    RemoteExecutionBackend,
    get_execution_backend,
)
from researchlens.workspace.schema import (
    ExecutionJob,
)
from researchlens.workspace.store import WorkspaceStore


def test_execution_backend_factory():
    """Verify backend factory instantiation and fallback."""
    local_b = get_execution_backend("LOCAL")
    assert isinstance(local_b, LocalExecutionBackend)
    assert local_b.backend_type == "LOCAL"

    docker_b = get_execution_backend("DOCKER")
    assert isinstance(docker_b, DockerExecutionBackend)
    assert docker_b.backend_type == "DOCKER"

    remote_b = get_execution_backend("REMOTE")
    assert isinstance(remote_b, RemoteExecutionBackend)
    assert remote_b.backend_type == "REMOTE"

    # Default fallback is LOCAL
    default_b = get_execution_backend("unknown_backend")
    assert isinstance(default_b, LocalExecutionBackend)


def test_local_execution_backend_lifecycle():
    """Verify end-to-end local backend execution, logging, and artifact handling."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = WorkspaceStore(base_dir=tmp_dir)
        ws = store.create_workspace("Test WS", "tester")
        backend = LocalExecutionBackend()

        run_id = "test_run_123"
        run_dir = store.get_run_dir(ws.workspace_id, run_id)
        artifacts_dir = os.path.join(run_dir, "artifacts")

        job = ExecutionJob(
            job_id="job_123",
            workspace_id=ws.workspace_id,
            experiment_id="exp_123",
            run_id=run_id,
            backend="LOCAL",
            command=["python", "scripts/experiments/deterministic_demo.py", "--output-dir", artifacts_dir],
            timeout=30,
        )

        # 1. Validate
        is_valid, val_err = backend.validate(job)
        assert is_valid is True
        assert val_err == ""

        # 2. Prepare
        prep_ok, prep_err = backend.prepare(job, run_dir)
        assert prep_ok is True
        assert os.path.exists(artifacts_dir)

        # 3. Start
        res = backend.start(job, run_dir, timeout_seconds=30)
        assert res["exit_code"] == 0
        assert res["timed_out"] is False
        assert "Experiment completed successfully" in res["stdout"]

        # 4. Collect logs
        logs = backend.collect_logs(job.job_id, run_dir)
        assert "Experiment completed" in logs["stdout"]

        # 5. Collect artifacts
        arts = backend.collect_artifacts(job.job_id, artifacts_dir)
        assert len(arts) >= 2
        art_names = [a.name for a in arts]
        assert "results.json" in art_names
        assert "predictions.json" in art_names

        # 6. Cleanup
        backend.cleanup(job.job_id)


def test_execution_job_lifecycle_transitions():
    """Verify state transitions and serialization."""
    job = ExecutionJob(
        job_id="job_traj",
        workspace_id="ws_1",
        experiment_id="exp_1",
        run_id="run_1",
    )
    assert job.status == "CREATED"

    job.transition_to("VALIDATING")
    assert job.status == "VALIDATING"

    job.transition_to("RUNNING")
    assert job.status == "RUNNING"
    assert job.started_at is not None

    job.transition_to("COMPLETED")
    assert job.status == "COMPLETED"
    assert job.completed_at is not None
    assert len(job.transitions) == 3
