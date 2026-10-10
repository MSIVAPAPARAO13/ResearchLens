"""Execution Backends Package (Phase 14)."""

from typing import Optional

from researchlens.execution.backends.base import BaseExecutionBackend
from researchlens.execution.backends.docker import DockerExecutionBackend
from researchlens.execution.backends.local import LocalExecutionBackend
from researchlens.execution.backends.remote import RemoteExecutionBackend
from researchlens.workspace.store import WorkspaceStore

__all__ = [
    "BaseExecutionBackend",
    "LocalExecutionBackend",
    "DockerExecutionBackend",
    "RemoteExecutionBackend",
    "get_execution_backend",
]


def get_execution_backend(
    backend_type: str = "LOCAL",
    store: Optional[WorkspaceStore] = None,
) -> BaseExecutionBackend:
    """Factory retrieving the configured execution backend instance."""
    normalized = (backend_type or "LOCAL").strip().upper()
    if normalized == "DOCKER":
        return DockerExecutionBackend(store=store)
    elif normalized == "REMOTE":
        return RemoteExecutionBackend()
    else:
        return LocalExecutionBackend()
