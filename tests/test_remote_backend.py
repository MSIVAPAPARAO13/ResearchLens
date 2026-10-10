"""Tests for Remote Execution Backend Contract (Phase 14)."""

from researchlens.execution.backends.remote import RemoteExecutionBackend
from researchlens.workspace.schema import ExecutionJob


def test_remote_backend_contract():
    """Verify RemoteExecutionBackend contract interface and default NOT_CONFIGURED state."""
    backend = RemoteExecutionBackend(provider_name="SLURM_OR_K8S")
    assert backend.backend_type == "REMOTE"

    job = ExecutionJob(
        job_id="job_remote_1",
        workspace_id="ws_remote",
        experiment_id="exp_remote",
        run_id="run_remote",
        backend="REMOTE",
        command=["python", "script.py"],
    )

    # 1. Submit
    submit_res = backend.submit(job)
    assert submit_res["status"] == "NOT_CONFIGURED"
    assert submit_res["job_id"] == "job_remote_1"

    # 2. Status
    st = backend.status("job_remote_1")
    assert st == "NOT_CONFIGURED"

    # 3. Logs
    logs = backend.logs("job_remote_1")
    assert "NOT_CONFIGURED" in logs["stderr"]

    # 4. Cancel
    cancelled, msg = backend.cancel("job_remote_1")
    assert cancelled is False
    assert "NOT_CONFIGURED" in msg

    # 5. Artifacts
    arts = backend.artifacts("job_remote_1")
    assert arts == []

    # 6. Validate & Prepare
    is_valid, val_err = backend.validate(job)
    assert is_valid is False
    assert "NOT_CONFIGURED" in val_err
