"""Tests for Container Security Enforcements & Sandbox Isolation (Phase 14)."""

from researchlens.execution.dataset_mount import DatasetMountValidator
from researchlens.execution.security import SecurityValidator
from researchlens.workspace.schema import ExecutionJob


def test_command_injection_rejection():
    """Verify dangerous shell metacharacters in command arguments are rejected."""
    bad_commands = [
        ["python", "script.py", "--dataset", "squad; rm -rf /"],
        ["python", "script.py", "--param", "val && whoami"],
        ["python", "script.py", "--flag", "`id`"],
        ["python", "script.py", "--flag", "test | cat /etc/passwd"],
    ]
    for cmd in bad_commands:
        is_safe, err = SecurityValidator.validate_command_arguments(cmd)
        assert is_safe is False
        assert "Dangerous shell character" in err


def test_docker_socket_mount_prevention():
    """Verify attempting to mount Docker socket is categorically rejected."""
    bad_socket_mounts = [
        "/var/run/docker.sock",
        "//./pipe/docker_engine",
    ]
    for sock in bad_socket_mounts:
        is_safe, err = DatasetMountValidator.validate_mount_safety(sock)
        assert is_safe is False
        assert "Forbidden Docker socket" in err


def test_host_root_traversal_mount_prevention():
    """Verify attempting to mount host root or parent directory traversal is rejected."""
    is_safe_root, _ = DatasetMountValidator.validate_mount_safety("/")
    assert is_safe_root is False

    is_safe_c, _ = DatasetMountValidator.validate_mount_safety("C:\\")
    assert is_safe_c is False


def test_network_policy_defaults_to_disabled():
    """Verify default network policy in ExecutionJob is DISABLED."""
    job = ExecutionJob(
        job_id="j_sec",
        workspace_id="ws_sec",
        experiment_id="e_sec",
        run_id="r_sec",
    )
    assert job.network_policy == "DISABLED"


def test_environment_secret_masking():
    """Verify sensitive environment credentials are not leaked into snapshots."""
    test_env = {
        "AWS_SECRET_ACCESS_KEY": "super_secret_key_12345",
        "MLFLOW_API_TOKEN": "token_abc_xyz",
        "SAFE_PARAM": "100",
    }
    sanitized = SecurityValidator.sanitize_env_vars(test_env)
    assert sanitized["AWS_SECRET_ACCESS_KEY"] == "[REDACTED_CREDENTIAL]"
    assert sanitized["MLFLOW_API_TOKEN"] == "[REDACTED_CREDENTIAL]"
    assert sanitized["SAFE_PARAM"] == "100"
