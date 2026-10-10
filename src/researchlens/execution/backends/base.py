"""Base Execution Backend Interface (Phase 14)."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from researchlens.workspace.schema import (
    ExecutionJob,
    ExperimentArtifact,
    ExperimentPlan,
)


class BaseExecutionBackend(ABC):
    """Abstract base class for all ResearchLens execution backends."""

    @property
    @abstractmethod
    def backend_type(self) -> str:
        """Name of backend: LOCAL, DOCKER, REMOTE."""
        pass

    @abstractmethod
    def validate(
        self,
        job: ExecutionJob,
        experiment: Optional[ExperimentPlan] = None,
    ) -> Tuple[bool, str]:
        """Validate execution job specification against backend capabilities and constraints."""
        pass

    @abstractmethod
    def prepare(
        self,
        job: ExecutionJob,
        run_dir: str,
    ) -> Tuple[bool, str]:
        """Prepare execution environment, directories, configuration, or mounts."""
        pass

    @abstractmethod
    def start(
        self,
        job: ExecutionJob,
        run_dir: str,
        timeout_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute the job and return process execution summary.

        Expected return keys:
        - exit_code: int
        - stdout: str
        - stderr: str
        - duration_seconds: float
        - timed_out: bool
        - error: Optional[str]
        - enforce_status: Dict[str, str] (e.g. CPU, memory enforcement status)
        """
        pass

    @abstractmethod
    def cancel(self, job_id: str) -> Tuple[bool, str]:
        """Cancel an actively executing job."""
        pass

    @abstractmethod
    def collect_logs(self, job_id: str, run_dir: str) -> Dict[str, str]:
        """Collect captured stdout and stderr logs."""
        pass

    @abstractmethod
    def collect_artifacts(self, job_id: str, artifacts_dir: str) -> List[ExperimentArtifact]:
        """Collect and discover generated artifacts strictly in designated output directory."""
        pass

    @abstractmethod
    def cleanup(self, job_id: str, run_dir: Optional[str] = None) -> None:
        """Clean up active process, container, or temporary mounts."""
        pass
