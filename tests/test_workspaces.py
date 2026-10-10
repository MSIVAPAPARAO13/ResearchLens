"""Unit tests for ResearchWorkspace and WorkspaceStore CRUD lifecycle."""

import shutil
import tempfile

import pytest

from researchlens.workspace.schema import ResearchWorkspace
from researchlens.workspace.store import WorkspaceStore, _sanitize_id


@pytest.fixture
def temp_ws_store():
    tmp_dir = tempfile.mkdtemp(prefix="rl_test_ws_")
    store = WorkspaceStore(base_dir=tmp_dir)
    yield store
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_research_workspace_schema():
    ws = ResearchWorkspace(
        workspace_id="ws_test1",
        name="Multimodal Search",
        description="Investigation into multimodal search models",
    )
    assert ws.status == "ACTIVE"
    d = ws.to_dict()
    assert d["name"] == "Multimodal Search"
    restored = ResearchWorkspace.from_dict(d)
    assert restored.workspace_id == "ws_test1"
    assert restored.description == "Investigation into multimodal search models"


def test_workspace_store_lifecycle(temp_ws_store):
    store = temp_ws_store

    # 1. Create workspace
    ws = store.create_workspace("Quantum NLP", "Quantum computing for language modeling")
    assert ws.workspace_id.startswith("ws_")
    assert ws.name == "Quantum NLP"

    # 2. Get workspace
    fetched = store.get_workspace(ws.workspace_id)
    assert fetched is not None
    assert fetched.name == "Quantum NLP"

    # 3. List workspaces
    all_ws = store.list_workspaces()
    assert len(all_ws) == 1
    assert all_ws[0].workspace_id == ws.workspace_id

    # 4. Update workspace
    fetched.name = "Quantum NLP Updated"
    store.save_workspace(fetched)
    updated = store.get_workspace(ws.workspace_id)
    assert updated.name == "Quantum NLP Updated"

    # 5. Delete workspace
    deleted = store.delete_workspace(ws.workspace_id)
    assert deleted is True
    assert store.get_workspace(ws.workspace_id) is None
    assert len(store.list_workspaces()) == 0


def test_sanitize_id_prevents_traversal():
    assert _sanitize_id("valid_id-123") == "valid_id-123"
    with pytest.raises(ValueError):
        _sanitize_id("../traversal")
    with pytest.raises(ValueError):
        _sanitize_id("folder/name")
    with pytest.raises(ValueError):
        _sanitize_id("")
