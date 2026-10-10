"""Security validation and sanitization for controlled experiment execution (Phase 12)."""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

# Prohibited file extensions for execution or registration
FORBIDDEN_EXTENSIONS = {
    ".exe",
    ".bat",
    ".ps1",
    ".cmd",
    ".vbs",
    ".sh",
    ".ipynb",
    ".dll",
    ".so",
    ".msi",
}

# Dangerous shell metacharacters that must never appear in command arguments
SHELL_INJECTION_PATTERN = re.compile(r"[;&|`$><\n\r]")

# Sensitive keywords that must be masked in environment variable snapshots
SENSITIVE_ENV_KEYS = {
    "secret",
    "token",
    "key",
    "password",
    "auth",
    "credential",
    "privkey",
    "api_key",
    "access_key",
}


class SecurityValidator:
    """Enforces strict execution sandboxing and input validation boundaries."""

    @staticmethod
    def validate_entrypoint(entrypoint: str, allowed_dirs: Optional[List[str]] = None) -> Tuple[bool, str]:
        """Validate that an experiment script entrypoint is safe and registered.

        Rejects arbitrary user-uploaded files, prohibited extensions, and path traversal.
        """
        if not entrypoint or not entrypoint.strip():
            return False, "Empty entrypoint path."

        clean_path = entrypoint.strip().replace("\\", "/")

        # 1. Reject path traversal
        if ".." in clean_path or clean_path.startswith("/"):
            return False, f"Path traversal or absolute path rejected: '{entrypoint}'"

        # 2. Reject forbidden extensions
        _, ext = os.path.splitext(clean_path.lower())
        if ext in FORBIDDEN_EXTENSIONS:
            return False, f"Prohibited executable or notebook extension '{ext}' in entrypoint."
        if ext != ".py":
            return False, f"Entrypoint must be a valid Python script (.py), got '{ext}'."

        # 3. Check allowed directories if specified
        if allowed_dirs:
            norm_allowed = [d.replace("\\", "/").rstrip("/") for d in allowed_dirs]
            matches_allowed = any(clean_path.startswith(f"{d}/") or clean_path == d for d in norm_allowed)
            if not matches_allowed:
                return False, f"Entrypoint '{entrypoint}' is not located in an approved directory: {allowed_dirs}."

        return True, ""

    @staticmethod
    def validate_command_arguments(args: List[str]) -> Tuple[bool, str]:
        """Verify command arguments for shell injection characters."""
        for arg in args:
            if not isinstance(arg, str):
                return False, f"Non-string command argument detected: {arg}"
            if SHELL_INJECTION_PATTERN.search(arg):
                return False, f"Dangerous shell character detected in argument: '{arg}'"
        return True, ""

    @staticmethod
    def validate_safe_path(path: str, base_dir: str) -> Tuple[bool, str]:
        """Ensure path resolves strictly within the intended base directory."""
        try:
            base_abs = os.path.abspath(base_dir)
            target_abs = os.path.abspath(os.path.join(base_dir, path))
            if not target_abs.startswith(base_abs):
                return False, f"Path '{path}' traverses outside base directory '{base_dir}'."
            return True, ""
        except Exception as e:
            return False, f"Invalid path resolution: {e}"

    @staticmethod
    def sanitize_env_vars(env_dict: Dict[str, Any]) -> Dict[str, str]:
        """Strip or mask secret keys, API tokens, and credentials from environment logs."""
        sanitized: Dict[str, str] = {}
        for k, v in env_dict.items():
            k_lower = str(k).lower()
            if any(sens in k_lower for sens in SENSITIVE_ENV_KEYS):
                sanitized[str(k)] = "[REDACTED_CREDENTIAL]"
            else:
                sanitized[str(k)] = str(v)
        return sanitized
