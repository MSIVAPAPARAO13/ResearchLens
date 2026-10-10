"""Tests for Execution Image Allowlist & Registry (Phase 14)."""

import tempfile

from researchlens.execution.image_registry import ExecutionImageRegistry
from researchlens.workspace.schema import ExecutionImageRegistryEntry
from researchlens.workspace.store import WorkspaceStore


def test_builtin_allowed_images():
    """Verify built-in approved images pass allowlist check."""
    registry = ExecutionImageRegistry()

    is_allowed, reason, digest = registry.is_image_allowed("researchlens/nlp-experiment:1.0")
    assert is_allowed is True
    assert "approved" in reason.lower()
    assert digest.startswith("sha256:")


def test_unknown_image_blocked():
    """Verify arbitrary unknown image returns BLOCKED."""
    registry = ExecutionImageRegistry()

    is_allowed, reason, digest = registry.is_image_allowed("attacker/arbitrary-miner:latest")
    assert is_allowed is False
    assert "BLOCKED" in reason
    assert digest == "NOT_PROVIDED"


def test_register_and_disable_image_in_workspace():
    """Verify dynamic workspace image registration and disabling."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = WorkspaceStore(base_dir=tmp_dir)
        ws = store.create_workspace("Image WS", "tester")
        registry = ExecutionImageRegistry(store=store)

        # Register custom image
        custom_entry = ExecutionImageRegistryEntry(
            image="custom/academic-eval:2.0",
            digest="sha256:111222333444555",
            allowed=True,
            owner="tester",
        )
        registry.register_image(custom_entry, ws.workspace_id)

        # Now should be allowed
        ok, _, dig = registry.is_image_allowed("custom/academic-eval:2.0", ws.workspace_id)
        assert ok is True
        assert dig == "sha256:111222333444555"

        # Explicitly disable
        custom_entry.allowed = False
        registry.register_image(custom_entry, ws.workspace_id)

        ok2, reason2, _ = registry.is_image_allowed("custom/academic-eval:2.0", ws.workspace_id)
        assert ok2 is False
        assert "BLOCKED" in reason2
