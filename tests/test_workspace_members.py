"""Unit tests for WorkspaceMember collaboration and permissions."""

import shutil
import tempfile

import pytest

from researchlens.workspace.schema import WorkspaceMember
from researchlens.workspace.store import WorkspaceStore


@pytest.fixture
def temp_ws_store():
    tmp_dir = tempfile.mkdtemp(prefix="rl_test_mem_")
    store = WorkspaceStore(base_dir=tmp_dir)
    yield store
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_workspace_members_and_permissions(temp_ws_store):
    store = temp_ws_store
    ws = store.create_workspace("Collaborative AI")
    ws_id = ws.workspace_id

    # 1. Add members with different roles
    owner = WorkspaceMember("mem_01", ws_id, "Alice Lead", "alice@lab.org", role="OWNER")
    researcher = WorkspaceMember("mem_02", ws_id, "Bob Scientist", "bob@lab.org", role="RESEARCHER")
    reviewer = WorkspaceMember("mem_03", ws_id, "Carol Reviewer", "carol@review.org", role="REVIEWER")
    viewer = WorkspaceMember("mem_04", ws_id, "David Observer", "david@guest.org", role="VIEWER")

    for m in [owner, researcher, reviewer, viewer]:
        store.save_member(m)

    members = store.list_members(ws_id)
    assert len(members) == 4

    # 2. Check permission hierarchy: OWNER > RESEARCHER > REVIEWER > VIEWER
    # Owner has full access
    assert store.check_permission(ws_id, "mem_01", "OWNER") is True
    assert store.check_permission(ws_id, "mem_01", "RESEARCHER") is True
    assert store.check_permission(ws_id, "mem_01", "VIEWER") is True

    # Researcher has researcher + reviewer + viewer access, but not owner
    assert store.check_permission(ws_id, "mem_02", "OWNER") is False
    assert store.check_permission(ws_id, "mem_02", "RESEARCHER") is True
    assert store.check_permission(ws_id, "mem_02", "REVIEWER") is True

    # Reviewer cannot perform researcher or owner actions
    assert store.check_permission(ws_id, "mem_03", "RESEARCHER") is False
    assert store.check_permission(ws_id, "mem_03", "REVIEWER") is True

    # Viewer has only read-only viewer access
    assert store.check_permission(ws_id, "mem_04", "REVIEWER") is False
    assert store.check_permission(ws_id, "mem_04", "VIEWER") is True

    # Unknown or inactive member
    assert store.check_permission(ws_id, "unknown_mem", "VIEWER") is False

    # 3. Delete member
    assert store.delete_member(ws_id, "mem_04") is True
    assert len(store.list_members(ws_id)) == 3
