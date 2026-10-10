"""Remote Execution Backend Contract (Phase 14)."""

from typing import Any, Dict, List, Optional, Tuple

from researchlens.execution.backends.base import BaseExecutionBackend
from researchlens.workspace.schema import (
    ExecutionJob,
    ExperimentArtifact,
    ExperimentPlan,
)


class RemoteExecutionBackend(BaseExecutionBackend):
    """Remote execution backend contract preparing for cloud/cluster orchestration (Slurm, K8s, Ray, AWS, GCP, Azure).

    Initial implementation safely defaults to NOT_CONFIGURED without calling external cloud infrastructure.
    """

    def __init__(self, provider_name: str = "GENERIC_REMOTE"):
        self.provider_name = provider_name

    @property
    def backend_type(self) -> str:
        return "REMOTE"

    def submit(self, job: ExecutionJob) -> Dict[str, Any]:
        """Submit job to remote execution provider."""
        return {
            "status": "NOT_CONFIGURED",
            "job_id": job.job_id,
            "provider": self.provider_name,
            "message": "Remote cloud execution backend is not configured. Supported targets: AWS, Azure, GCP, Slurm, Kubernetes, Ray.",
        }

    def status(self, job_id: str) -> str:
        """Query remote job status."""
        return "NOT_CONFIGURED"

    def cancel(self, job_id: str) -> Tuple[bool, str]:
        """Cancel remote job."""
        return False, "Remote execution backend is NOT_CONFIGURED."

    def logs(self, job_id: str) -> Dict[str, str]:
        """Retrieve remote execution logs."""
        return {
            "stdout": "",
            "stderr": "Remote execution backend is NOT_CONFIGURED.",
        }

    def artifacts(self, job_id: str) -> List[ExperimentArtifact]:
        """Fetch remote execution artifacts."""
        return []

    # Implement BaseExecutionBackend interface methods
    def validate(
        self,
        job: ExecutionJob,
        experiment: Optional[ExperimentPlan] = None,
    ) -> Tuple[bool, str]:
        return False, "Remote execution backend is currently NOT_CONFIGURED."

    def prepare(
        self,
        job: ExecutionJob,
        run_dir: str,
    ) -> Tuple[bool, str]:
        return False, "Remote execution backend is currently NOT_CONFIGURED."

    def start(
        self,
        job: ExecutionJob,
        run_dir: str,
        timeout_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        return {
            "exit_code": 1,
            "stdout": "",
            "stderr": "Remote execution backend is NOT_CONFIGURED.",
            "duration_seconds": 0.0,
            "timed_out": False,
            "error": "REMOTE_BACKEND_NOT_CONFIGURED",
            "enforce_status": {
                "cpu_limit": "LIMIT_NOT_ENFORCED",
                "memory_limit": "LIMIT_NOT_ENFORCED",
                "disk_limit": "LIMIT_NOT_ENFORCED",
                "timeout": "LIMIT_NOT_ENFORCED",
            },
        }

    def collect_logs(self, job_id: str, run_dir: str) -> Dict[str, str]:
        return self.logs(job_id)

    def collect_artifacts(self, job_id: str, artifacts_dir: str) -> List[ExperimentArtifact]:
        return self.artifacts(job_id)

    def cleanup(self, job_id: str, run_dir: Optional[str] = None) -> None:
        pass
