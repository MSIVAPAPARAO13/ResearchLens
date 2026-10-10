"""Local Process Execution Backend (Phase 14)."""

import os
import subprocess
import threading
from typing import Any, Dict, List, Optional, Tuple

from researchlens.execution.artifacts import discover_artifacts
from researchlens.execution.backends.base import BaseExecutionBackend
from researchlens.execution.process_monitor import ProcessMonitor
from researchlens.execution.security import SecurityValidator
from researchlens.utils.logging import get_logger
from researchlens.workspace.schema import (
    ExecutionJob,
    ExperimentArtifact,
    ExperimentPlan,
)

logger = get_logger("local_backend")


class LocalExecutionBackend(BaseExecutionBackend):
    """Executes experiments as isolated subprocesses on the local host."""

    def __init__(self):
        self._active_processes: Dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    @property
    def backend_type(self) -> str:
        return "LOCAL"

    def validate(
        self,
        job: ExecutionJob,
        experiment: Optional[ExperimentPlan] = None,
    ) -> Tuple[bool, str]:
        """Validate execution arguments and safety boundaries."""
        if not job.command:
            return False, "Job command is empty."

        # Validate arguments against shell injection
        is_safe, err = SecurityValidator.validate_command_arguments(job.command)
        if not is_safe:
            return False, f"Local security validation failed: {err}"

        # Resource limits on local OS: CPU/memory limits on local OS cannot be enforced via standard subprocess
        return True, ""

    def prepare(
        self,
        job: ExecutionJob,
        run_dir: str,
    ) -> Tuple[bool, str]:
        """Prepare log and artifact directories for the local run."""
        try:
            os.makedirs(os.path.join(run_dir, "artifacts"), exist_ok=True)
            os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
            return True, ""
        except Exception as e:
            return False, f"Failed to prepare run directory: {e}"

    def start(
        self,
        job: ExecutionJob,
        run_dir: str,
        timeout_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute command safely via subprocess.run (shell=False)."""
        effective_timeout = timeout_seconds or job.timeout or 3600
        logs_dir = os.path.join(run_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        monitor = ProcessMonitor(timeout_seconds=effective_timeout)
        monitor.start()

        stdout_content = ""
        stderr_content = ""
        timed_out = False
        exit_code = 1
        error_msg = None

        # Resource enforcement notice for local backend
        enforce_status = {
            "cpu_limit": "LIMIT_NOT_ENFORCED",
            "memory_limit": "LIMIT_NOT_ENFORCED",
            "disk_limit": "LIMIT_NOT_ENFORCED",
            "timeout": "ENFORCED",
        }

        try:
            res = subprocess.run(
                job.command,
                cwd=".",
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
            stderr_content = e.stderr or f"Process timed out after {effective_timeout} seconds."
            exit_code = -1
            error_msg = f"Local process exceeded timeout of {effective_timeout}s."
        except Exception as e:
            stderr_content = f"Process invocation failure: {e}"
            exit_code = 1
            error_msg = str(e)

        duration = monitor.stop()

        # Persist stdout and stderr
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
        }

    def cancel(self, job_id: str) -> Tuple[bool, str]:
        """Cancel an executing local process if registered."""
        with self._lock:
            proc = self._active_processes.get(job_id)
            if proc:
                try:
                    proc.terminate()
                    return True, "Local process terminated."
                except Exception as e:
                    return False, f"Failed to terminate local process: {e}"
        return True, "No active subprocess to terminate."

    def collect_logs(self, job_id: str, run_dir: str) -> Dict[str, str]:
        """Read standard stdout and stderr logs."""
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
        """Discover and hash artifacts strictly within artifacts_dir."""
        return discover_artifacts(artifacts_dir, run_id=job_id)

    def cleanup(self, job_id: str, run_dir: Optional[str] = None) -> None:
        """Clean up active process references."""
        with self._lock:
            self._active_processes.pop(job_id, None)
