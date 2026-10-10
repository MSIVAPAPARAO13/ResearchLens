"""Unit tests for audit trail logging and credential security filtering."""

import shutil
import tempfile

import pytest

from researchlens.workspace.store import WorkspaceStore


@pytest.fixture
def temp_ws_store():
    tmp_dir = tempfile.mkdtemp(prefix="rl_test_aud_")
    store = WorkspaceStore(base_dir=tmp_dir)
    yield store
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_audit_log_and_security_sanitization(temp_ws_store):
    store = temp_ws_store
    ws = store.create_workspace("Audit Security Lab")
    ws_id = ws.workspace_id

    # Test that secrets are filtered out from audit metadata
    tainted_metadata = {
        "export_format": "markdown",
        "api_key": "sk-secret-123456",
        "access_token": "bearer-private",
        "paper_count": 14,
    }

    event = store.log_audit(
        workspace_id=ws_id,
        actor_id="mem_admin",
        action="EXPORT",
        target_type="WORKSPACE",
        target_id=ws_id,
        metadata=tainted_metadata,
    )

    assert event.event_id.startswith("aud_")
    assert "export_format" in event.metadata
    assert "paper_count" in event.metadata
    # Security Rule: Secrets MUST NOT be stored in audit metadata
    assert "api_key" not in event.metadata

    audits = store.list_audit(ws_id)
    assert len(audits) == 1
    assert audits[0].action == "EXPORT"
