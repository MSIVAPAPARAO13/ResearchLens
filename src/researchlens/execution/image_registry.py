"""Docker Execution Image Registry and Security Allowlist (Phase 14)."""

from typing import List, Optional, Tuple

from researchlens.workspace.schema import ExecutionImageRegistryEntry
from researchlens.workspace.store import WorkspaceStore

# Built-in approved images
BUILTIN_ALLOWED_IMAGES = [
    ExecutionImageRegistryEntry(
        image="researchlens/nlp-experiment:1.0",
        digest="sha256:d8a9f4e2b0c1e8736a5432109876543210abcdef0123456789abcdef01234567",
        allowed=True,
        owner="system",
        description="Official ResearchLens NLP experiment image",
    ),
    ExecutionImageRegistryEntry(
        image="researchlens/cpu-baseline:1.0",
        digest="sha256:a1b2c3d4e5f60718293a4b5c6d7e8f901234567890abcdef1234567890abcdef",
        allowed=True,
        owner="system",
        description="Official ResearchLens CPU baseline image",
    ),
    ExecutionImageRegistryEntry(
        image="python:3.11-slim",
        digest="NOT_PROVIDED",
        allowed=True,
        owner="system",
        description="Base Python 3.11 slim runtime for containerized evaluations",
    ),
]


class ExecutionImageRegistry:
    """Manages approved container images to prevent untrusted container execution."""

    def __init__(self, store: Optional[WorkspaceStore] = None):
        self.store = store

    def list_images(self, workspace_id: Optional[str] = None) -> List[ExecutionImageRegistryEntry]:
        """List all approved images from workspace store or built-in registry."""
        if self.store and workspace_id:
            stored = self.store.list_allowed_images(workspace_id)
            if stored:
                return stored
        return list(BUILTIN_ALLOWED_IMAGES)

    def is_image_allowed(
        self,
        image_name: str,
        workspace_id: Optional[str] = None,
    ) -> Tuple[bool, str, str]:
        """Verify whether an image is explicitly allowed in the registry.

        Returns (is_allowed, reason, digest).
        If unknown or not approved: returns (False, 'BLOCKED: ...', 'NOT_PROVIDED').
        """
        if not image_name or not image_name.strip():
            return False, "BLOCKED: Image name is empty.", "NOT_PROVIDED"

        clean_name = image_name.strip()
        images = self.list_images(workspace_id)

        for entry in images:
            # Match exact or tag-stripped name
            if entry.image == clean_name or entry.image.split(":")[0] == clean_name.split(":")[0]:
                if entry.allowed:
                    digest = entry.digest if entry.digest else "NOT_PROVIDED"
                    return True, "Image is approved in allowlist.", digest
                else:
                    return False, f"BLOCKED: Image '{clean_name}' is explicitly disabled in allowlist.", entry.digest

        return False, f"BLOCKED: Image '{clean_name}' is not in the approved ExecutionImageRegistry.", "NOT_PROVIDED"

    def register_image(
        self,
        entry: ExecutionImageRegistryEntry,
        workspace_id: Optional[str] = None,
    ) -> None:
        """Register or update an approved image."""
        if self.store and workspace_id:
            self.store.save_allowed_image(entry, workspace_id)
