"""Reproducible Experiment Execution & MLOps Integration Package (Phase 12-14)."""

from researchlens.execution.artifact_store import (
    BaseArtifactStore,
    LocalArtifactStore,
    S3CompatibleArtifactStore,
    get_artifact_store,
)
from researchlens.execution.artifacts import (
    calculate_sha256,
    discover_artifacts,
    verify_artifact_hash,
)
from researchlens.execution.backends import (
    BaseExecutionBackend,
    DockerExecutionBackend,
    LocalExecutionBackend,
    RemoteExecutionBackend,
    get_execution_backend,
)
from researchlens.execution.dataset_mount import DatasetMountValidator
from researchlens.execution.environment import (
    capture_environment_snapshot,
    capture_source_snapshot,
    get_git_commit_sha,
    get_git_provenance,
    get_gpu_info,
)
from researchlens.execution.executor import ExperimentExecutor
from researchlens.execution.image_registry import ExecutionImageRegistry
from researchlens.execution.matrix import ExperimentMatrixGenerator
from researchlens.execution.process_monitor import ProcessMonitor, classify_failure
from researchlens.execution.provenance import (
    calculate_reproducibility_completeness_score,
    generate_reproducibility_manifest,
)
from researchlens.execution.queue import ExecutionQueueManager
from researchlens.execution.registry import ExperimentRegistryService
from researchlens.execution.regression import RegressionDetector
from researchlens.execution.reproducibility import (
    ReproducibilityDiffService,
    ReproducibilityDriftDetector,
)
from researchlens.execution.security import SecurityValidator

__all__ = [
    "ExperimentExecutor",
    "ExperimentRegistryService",
    "SecurityValidator",
    "ProcessMonitor",
    "classify_failure",
    "capture_environment_snapshot",
    "get_git_commit_sha",
    "get_git_provenance",
    "capture_source_snapshot",
    "get_gpu_info",
    "calculate_sha256",
    "verify_artifact_hash",
    "discover_artifacts",
    "generate_reproducibility_manifest",
    "calculate_reproducibility_completeness_score",
    "BaseExecutionBackend",
    "LocalExecutionBackend",
    "DockerExecutionBackend",
    "RemoteExecutionBackend",
    "get_execution_backend",
    "ExecutionImageRegistry",
    "DatasetMountValidator",
    "BaseArtifactStore",
    "LocalArtifactStore",
    "S3CompatibleArtifactStore",
    "get_artifact_store",
    "ExecutionQueueManager",
    "ReproducibilityDiffService",
    "ReproducibilityDriftDetector",
    "ExperimentMatrixGenerator",
    "RegressionDetector",
]
