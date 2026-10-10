"""Dataset mounting and isolation validation for containerized runs (Phase 14)."""

import os
from typing import Optional, Tuple

from researchlens.execution.artifacts import calculate_sha256
from researchlens.workspace.schema import DatasetMount


class DatasetMountValidator:
    """Enforces strict isolation boundaries for container dataset mounts."""

    # Prohibited host paths that must never be mounted
    PROHIBITED_PREFIXES = [
        "/",
        "/root",
        "/home",
        "/etc",
        "/var",
        "/usr",
        "/bin",
        "/sbin",
        "/var/run/docker.sock",
        "//./pipe/docker_engine",
        "c:\\",
        "c:/",
        "c:\\windows",
        "c:\\users",
    ]

    @staticmethod
    def validate_mount_safety(host_path: str, allowed_base_dir: Optional[str] = None) -> Tuple[bool, str]:
        """Validate that a host path does not mount sensitive system directories or traverse paths."""
        if not host_path or not host_path.strip():
            return False, "Host mount path is empty."

        raw_norm = host_path.strip().replace("\\", "/").lower()

        # 1. Docker socket
        if "docker.sock" in raw_norm or "docker_engine" in raw_norm:
            return False, f"Forbidden Docker socket mount attempt: '{host_path}'."

        # 2. Host root attempts
        clean = os.path.abspath(host_path).replace("\\", "/").lower()
        if raw_norm.rstrip("/") in ["", "/", "c:", "c:/"] or clean.rstrip("/") in ["/", "c:", "c:/"]:
            return False, f"Forbidden host root mount attempt: '{host_path}'."

        # 3. Sensitive system directories
        system_roots = ["/etc", "/var", "/usr", "/bin", "/sbin", "c:/windows"]
        for s_root in system_roots:
            if clean == s_root or clean.startswith(f"{s_root}/"):
                return False, f"Forbidden system directory mount attempt: '{host_path}'."

        # 4. Entire user directory mount attempts (cross-platform check)
        raw_parts = [p for p in raw_norm.split("/") if p]
        if (len(raw_parts) <= 2 and len(raw_parts) >= 1 and raw_parts[0] in ["home", "root", "users"]) or \
           (len(raw_parts) <= 3 and len(raw_parts) >= 2 and raw_parts[0] in ["c:", "d:"] and raw_parts[1] in ["users", "home"]):
            if not os.path.isfile(host_path):
                return False, f"Forbidden entire user directory mount attempt: '{host_path}'."

        parts = [p for p in clean.split("/") if p]
        if (len(parts) <= 2 and len(parts) >= 1 and parts[0] in ["home", "root"]) or \
           (len(parts) <= 3 and len(parts) >= 2 and parts[0] in ["c:", "d:"] and parts[1] in ["users", "home"]):
            if not os.path.isfile(host_path):
                return False, f"Forbidden entire user directory mount attempt: '{host_path}'."

        if allowed_base_dir:
            base_abs = os.path.abspath(allowed_base_dir).replace("\\", "/").lower()
            if not clean.startswith(base_abs):
                return False, f"Mount path '{host_path}' traverses outside designated workspace data directory."

        return True, ""

    @staticmethod
    def prepare_dataset_mount(
        dataset_id: str,
        host_dataset_path: str,
        mount_target: str = "/workspace/data",
        read_only: bool = True,
    ) -> Tuple[Optional[DatasetMount], str]:
        """Create a validated DatasetMount definition."""
        is_safe, err = DatasetMountValidator.validate_mount_safety(host_dataset_path)
        if not is_safe:
            return None, err

        if not os.path.exists(host_dataset_path):
            return None, f"Dataset path does not exist on host: '{host_dataset_path}'."

        # Compute dataset hash if file
        ds_hash = "NOT_PROVIDED"
        if os.path.isfile(host_dataset_path):
            ds_hash = calculate_sha256(host_dataset_path)

        mount = DatasetMount(
            dataset_id=dataset_id,
            mount_path=mount_target,
            read_only=read_only,
            hash=ds_hash or "NOT_PROVIDED",
        )
        return mount, ""
