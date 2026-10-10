"""Containerized Docker Execution Backend (Phase 14)."""

import os
import subprocess
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple

from researchlens.execution.artifacts import discover_artifacts
from researchlens.execution.backends.base import BaseExecutionBackend
from researchlens.execution.dataset_mount import DatasetMountValidator
from researchlens.execution.image_registry import ExecutionImageRegistry
from researchlens.execution.process_monitor import ProcessMonitor
from researchlens.execution.security import SecurityValidator
from researchlens.utils.logging import get_logger
from researchlens.workspace.schema import (
    ExecutionJob,
    ExperimentArtifact,
    ExperimentPlan,
)
from researchlens.workspace.store import WorkspaceStore

logger = get_logger("docker_backend")


class DockerExecutionBackend(BaseExecutionBackend):
    """Executes experiments in isolated, sandboxed Docker containers."""

    def __init__(self, store: Optional[WorkspaceStore] = None):
        self.store = store
        self.image_registry = ExecutionImageRegistry(store)
        self._active_containers: Dict[str, str] = {}
        self._lock = threading.Lock()

    @property
    def backend_type(self) -> str:
        return "DOCKER"

    def is_docker_available(self) -> bool:
        """Check if Docker CLI and daemon are accessible."""
        try:
            res = subprocess.run(
                ["docker", "info"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False,
                timeout=5,
            )
            return res.returncode == 0
        except Exception:
            return False

    def validate(
        self,
        job: ExecutionJob,
        experiment: Optional[ExperimentPlan] = None,
    ) -> Tuple[bool, str]:
        """Validate Docker execution boundaries, image allowlist, mounts, and resource settings."""
        # 1. Validate image allowlist
        image_name = job.image or (experiment.execution_image if experiment else None) or "researchlens/nlp-experiment:1.0"
        is_allowed, reason, digest = self.image_registry.is_image_allowed(image_name, job.workspace_id)
        if not is_allowed:
            return False, reason

        # 2. Validate mounts against dangerous paths
        for mount in job.resources.get("mounts", []):
            host_path = mount.get("host_path", "")
            is_safe, mount_err = DatasetMountValidator.validate_mount_safety(host_path)
            if not is_safe:
                return False, f"Security violation in mount: {mount_err}"

        # 3. Validate command arguments against shell metacharacters
        if job.command:
            is_safe_cmd, cmd_err = SecurityValidator.validate_command_arguments(job.command)
            if not is_safe_cmd:
                return False, f"Security violation in command arguments: {cmd_err}"

        return True, ""

    def prepare(
        self,
        job: ExecutionJob,
        run_dir: str,
    ) -> Tuple[bool, str]:
        """Prepare local artifacts directory mapped to container /workspace/output."""
        try:
            artifacts_dir = os.path.join(run_dir, "artifacts")
            logs_dir = os.path.join(run_dir, "logs")
            os.makedirs(artifacts_dir, exist_ok=True)
            os.makedirs(logs_dir, exist_ok=True)
            return True, ""
        except Exception as e:
            return False, f"Failed to prepare Docker run directory: {e}"

    def start(
        self,
        job: ExecutionJob,
        run_dir: str,
        timeout_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute experiment inside sandboxed container with resource controls and strict mounting."""
        effective_timeout = timeout_seconds or job.timeout or 3600
        logs_dir = os.path.join(run_dir, "logs")
        artifacts_dir = os.path.join(run_dir, "artifacts")
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(artifacts_dir, exist_ok=True)

        if not self.is_docker_available():
            return {
                "exit_code": 1,
                "stdout": "",
                "stderr": "Docker daemon is not available or not running on host.",
                "duration_seconds": 0.0,
                "timed_out": False,
                "error": "DOCKER_NOT_AVAILABLE",
                "enforce_status": {
                    "cpu_limit": "LIMIT_NOT_ENFORCED",
                    "memory_limit": "LIMIT_NOT_ENFORCED",
                    "disk_limit": "LIMIT_NOT_ENFORCED",
                    "timeout": "LIMIT_NOT_ENFORCED",
                },
            }

        image_name = job.image or "researchlens/nlp-experiment:1.0"
        _, _, digest = self.image_registry.is_image_allowed(image_name, job.workspace_id)

        container_name = f"researchlens_run_{job.run_id}"

        # Network Policy (Section 8)
        network_flag = "none" if job.network_policy != "ENABLED" else "bridge"
        network_display = "DISABLED" if job.network_policy != "ENABLED" else "ENABLED"

        # Build Docker CLI arguments (shell=False)
        docker_args = [
            "docker", "run",
            "--name", container_name,
            "--rm",
            "--network", network_flag,
            "--security-opt", "no-new-privileges:true",
        ]

        # Security options: run as non-root where supported
        docker_args.extend(["--user", "1000:1000"])

        # Capabilities: drop unnecessary Linux capabilities
        docker_args.extend(["--cap-drop", "ALL"])

        # Resource Limits (Section 7)
        enforce_status = {
            "cpu_limit": "LIMIT_NOT_ENFORCED",
            "memory_limit": "LIMIT_NOT_ENFORCED",
            "disk_limit": "LIMIT_NOT_ENFORCED",
            "timeout": "ENFORCED",
            "network": f"Network: {network_display}",
        }

        cpu_limit = job.resources.get("cpu_limit")
        if cpu_limit:
            try:
                cpu_val = float(cpu_limit)
                docker_args.extend(["--cpus", str(cpu_val)])
                enforce_status["cpu_limit"] = f"ENFORCED ({cpu_val} CPUs)"
            except (ValueError, TypeError):
                enforce_status["cpu_limit"] = "LIMIT_NOT_ENFORCED"

        memory_limit = job.resources.get("memory_limit")
        if memory_limit:
            clean_mem = str(memory_limit).strip().lower()
            docker_args.extend(["-m", clean_mem])
            enforce_status["memory_limit"] = f"ENFORCED ({clean_mem})"

        # Artifact Mount (Section 19): /workspace/output strictly
        abs_artifacts = os.path.abspath(artifacts_dir).replace("\\", "/")
        docker_args.extend(["-v", f"{abs_artifacts}:/workspace/output:rw"])

        # Optional dataset mounts (Section 18)
        for mount in job.resources.get("mounts", []):
            h_path = mount.get("host_path")
            target = mount.get("mount_path", "/workspace/data")
            ro = "ro" if mount.get("read_only", True) else "rw"
            if h_path and os.path.exists(h_path):
                clean_h_path = os.path.abspath(h_path).replace("\\", "/")
                docker_args.extend(["-v", f"{clean_h_path}:{target}:{ro}"])

        # Environment variables (sanitized)
        for k, v in job.environment.items():
            docker_args.extend(["--env", f"{k}={v}"])

        # Image target
        docker_args.append(image_name)

        # Command arguments to run inside container
        # Translate local entrypoint/output-dir to container paths
        container_cmd = []
        skip_next = False
        for i, token in enumerate(job.command):
            if skip_next:
                skip_next = False
                continue
            if token == "--output-dir":
                container_cmd.extend(["--output-dir", "/workspace/output"])
                skip_next = True
            elif token.startswith("--output-dir="):
                container_cmd.append("--output-dir=/workspace/output")
            elif token == sys.executable or token.endswith("python") or token.endswith("python.exe"):
                container_cmd.append("python")
            else:
                container_cmd.append(token)

        docker_args.extend(container_cmd)

        monitor = ProcessMonitor(timeout_seconds=effective_timeout)
        monitor.start()

        with self._lock:
            self._active_containers[job.job_id] = container_name

        stdout_content = ""
        stderr_content = ""
        timed_out = False
        exit_code = 1
        error_msg = None

        try:
            res = subprocess.run(
                docker_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False,
                timeout=effective_timeout,
            )
            exit_code = res.returncode
            stdout_content = res.stdout or ""
            stderr_content = res.stderr or ""
        except subprocess.TimeoutExpired as e:
            timed_out = True
            stdout_content = e.stdout or ""
            stderr_content = e.stderr or f"Container timed out after {effective_timeout} seconds."
            exit_code = -1
            error_msg = f"Container execution exceeded timeout of {effective_timeout}s."
            self.cancel(job.job_id)
        except Exception as e:
            stderr_content = f"Docker execution error: {e}"
            exit_code = 1
            error_msg = str(e)
        finally:
            with self._lock:
                self._active_containers.pop(job.job_id, None)

        duration = monitor.stop()

        # Persist stdout and stderr logs
        stdout_path = os.path.join(logs_dir, "stdout.log")
        stderr_path = os.path.join(logs_dir, "stderr.log")
        with open(stdout_path, "w", encoding="utf-8") as f:
            f.write(stdout_content)
        with open(stderr_path, "w", encoding="utf-8") as f:
            f.write(stderr_content)

        return {
            "exit_code": exit_code,
            "stdout": stdout_content,
            "stderr": stderr_content,
            "duration_seconds": duration,
            "timed_out": timed_out,
            "error": error_msg,
            "enforce_status": enforce_status,
            "image": image_name,
            "image_digest": digest,
        }

    def cancel(self, job_id: str) -> Tuple[bool, str]:
        """Cancel and terminate active Docker container."""
        container_name = None
        with self._lock:
            container_name = self._active_containers.get(job_id)

        if container_name:
            try:
                subprocess.run(
                    ["docker", "rm", "-f", container_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=False,
                    timeout=5,
                )
                return True, f"Container '{container_name}' removed."
            except Exception as e:
                return False, f"Failed to stop container: {e}"
        return True, "No active container for job."

    def collect_logs(self, job_id: str, run_dir: str) -> Dict[str, str]:
        """Retrieve stdout and stderr from run directory."""
        logs = {"stdout": "", "stderr": ""}
        stdout_path = os.path.join(run_dir, "logs", "stdout.log")
        stderr_path = os.path.join(run_dir, "logs", "stderr.log")
        if os.path.exists(stdout_path):
            with open(stdout_path, "r", encoding="utf-8") as f:
                logs["stdout"] = f.read()
        if os.path.exists(stderr_path):
            with open(stderr_path, "r", encoding="utf-8") as f:
                logs["stderr"] = f.read()
        return logs

    def collect_artifacts(self, job_id: str, artifacts_dir: str) -> List[ExperimentArtifact]:
        """Collect artifacts generated in the mounted /workspace/output directory."""
        return discover_artifacts(artifacts_dir, run_id=job_id)

    def cleanup(self, job_id: str, run_dir: Optional[str] = None) -> None:
        """Ensure container is cleaned up."""
        self.cancel(job_id)
