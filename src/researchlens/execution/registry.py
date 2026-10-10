"""Experiment registry service managing approved execution entrypoints (Phase 12)."""

import os
from typing import List, Optional, Tuple

from researchlens.execution.security import SecurityValidator
from researchlens.utils.logging import get_logger
from researchlens.workspace.schema import ExperimentRegistryEntry, generate_id
from researchlens.workspace.store import WorkspaceStore

logger = get_logger("execution_registry")

# Approved directories where experiment scripts can reside
APPROVED_EXPERIMENT_DIRS = [
    "scripts/experiments",
    "scripts",
]


class ExperimentRegistryService:
    """Manages explicit registration and approval of experiment entrypoints."""

    def __init__(self, store: WorkspaceStore):
        self.store = store

    def register_experiment(
        self,
        workspace_id: str,
        name: str,
        entrypoint: str,
        description: str = "",
        command_template: str = "",
        owner: str = "system",
        allowed: bool = True,
        default_timeout_seconds: int = 3600,
        version: str = "1.0",
    ) -> Tuple[Optional[ExperimentRegistryEntry], str]:
        """Register a new experiment entrypoint after strict security validation."""
        is_safe, err = SecurityValidator.validate_entrypoint(entrypoint, allowed_dirs=APPROVED_EXPERIMENT_DIRS)
        if not is_safe:
            return None, f"Security validation rejected entrypoint: {err}"

        # Verify entrypoint exists on disk
        if not os.path.exists(entrypoint):
            return None, f"Registered entrypoint file does not exist: '{entrypoint}'"

        entry = ExperimentRegistryEntry(
            experiment_id=generate_id("reg"),
            name=name,
            description=description,
            entrypoint=entrypoint.replace("\\", "/"),
            version=version,
            owner=owner,
            allowed=allowed,
            command_template=command_template,
            default_timeout_seconds=default_timeout_seconds,
        )

        self.store.save_registry_entry(entry, workspace_id)
        self.store.log_audit(
            workspace_id=workspace_id,
            actor_id=owner,
            action="CREATE",
            target_type="EXPERIMENT_REGISTRY",
            target_id=entry.experiment_id,
            metadata={"name": name, "entrypoint": entrypoint, "version": version},
        )
        return entry, ""

    def list_experiments(self, workspace_id: str) -> List[ExperimentRegistryEntry]:
        return self.store.list_registry_entries(workspace_id)

    def list_registered_experiments(self, workspace_id: str) -> List[ExperimentRegistryEntry]:
        """List registered experiment entrypoints (alias for list_experiments for UI compatibility)."""
        return self.list_experiments(workspace_id)

    def toggle_allowed(self, workspace_id: str, experiment_id: str, allowed: bool) -> bool:
        """Toggle execution allowance of an experiment entrypoint."""
        entry = self.get_experiment(workspace_id, experiment_id)
        if not entry:
            return False
        entry.allowed = allowed
        self.store.save_registry_entry(entry, workspace_id)
        self.store.log_audit(
            workspace_id=workspace_id,
            actor_id="user",
            action="UPDATE",
            target_type="EXPERIMENT_REGISTRY",
            target_id=entry.experiment_id,
            metadata={"allowed": allowed, "action": "toggle_allowed"},
        )
        return True

    def get_experiment(self, workspace_id: str, experiment_id: str) -> Optional[ExperimentRegistryEntry]:
        return self.store.get_registry_entry(workspace_id, experiment_id)

    def validate_execution_allowed(self, workspace_id: str, entrypoint: str) -> Tuple[bool, str]:
        """Verify that an entrypoint is explicitly registered and permitted for execution."""
        clean_entrypoint = entrypoint.strip().replace("\\", "/")

        is_safe, err = SecurityValidator.validate_entrypoint(clean_entrypoint, allowed_dirs=APPROVED_EXPERIMENT_DIRS)
        if not is_safe:
            return False, err

        entries = self.list_experiments(workspace_id)
        matching = [e for e in entries if e.entrypoint == clean_entrypoint and e.allowed]

        if not matching:
            return False, f"Entrypoint '{clean_entrypoint}' is not registered or not permitted for execution."

        return True, ""
