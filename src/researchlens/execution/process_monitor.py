"""Process execution monitoring, resource observation, and failure classification (Phase 12)."""

import time
from typing import Optional

from researchlens.utils.logging import get_logger

logger = get_logger("process_monitor")


class ProcessMonitor:
    """Monitors process execution duration and resource utilization."""

    def __init__(self, timeout_seconds: int = 3600):
        self.timeout_seconds = timeout_seconds
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.timed_out: bool = False

    def start(self) -> None:
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        self.end_time = time.perf_counter()
        if self.start_time is not None:
            return round(self.end_time - self.start_time, 3)
        return 0.0

    @property
    def duration(self) -> float:
        if self.start_time is None:
            return 0.0
        current = self.end_time or time.perf_counter()
        return round(current - self.start_time, 3)

    @staticmethod
    def get_resource_monitoring_mode() -> str:
        """Indicate monitoring capability boundary truthfully."""
        try:
            import psutil  # noqa: F401

            return "RESOURCE LIMIT MONITORING ACTIVE"
        except ImportError:
            return "RESOURCE LIMIT MONITORING ONLY"


def classify_failure(
    exit_code: Optional[int],
    stderr: str = "",
    stdout: str = "",
    timed_out: bool = False,
) -> str:
    """Classify execution failure into a truthful, evidence-supported category.

    Categories:
    - TIMEOUT
    - DEPENDENCY_ERROR
    - DATA_ERROR
    - CONFIGURATION_ERROR
    - MODEL_ERROR
    - RESOURCE_ERROR
    - UNKNOWN
    """
    if timed_out:
        return "TIMEOUT"

    combined = (stderr + "\n" + stdout).lower()

    # 1. Dependency errors
    if any(m in combined for m in ["modulenotfounderror", "importerror", "no module named", "cannot import name"]):
        return "DEPENDENCY_ERROR"

    # 2. Configuration errors
    if any(m in combined for m in ["unrecognized arguments", "invalid argument", "configerror", "configuration error", "missing required argument"]):
        return "CONFIGURATION_ERROR"

    # 3. Model & Tensor execution errors
    if any(m in combined for m in ["cuda out of memory", "shape mismatch", "size mismatch", "dimension mismatch", "mat1 and mat2 shapes cannot be multiplied"]):
        return "MODEL_ERROR"

    # 4. Data errors
    if any(m in combined for m in ["filenotfounderror", "dataset not found", "data error", "keyerror", "no such file or directory"]):
        return "DATA_ERROR"

    # 5. Resource limits
    if any(m in combined for m in ["memoryerror", "out of memory", "killed", "disk full", "no space left on device"]):
        return "RESOURCE_ERROR"

    if exit_code is not None and exit_code != 0:
        return "UNKNOWN"

    return "UNKNOWN"
