"""Unit tests for team activity timeline event logging."""

import shutil
import tempfile

import pytest

from researchlens.workspace.store import WorkspaceStore


@pytest.fixture
def temp_ws_store():
    tmp_dir = tempfile.mkdtemp(prefix="rl_test_act_")
    store = WorkspaceStore(base_dir=tmp_dir)
    yield store
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_activity_event_logging(temp_ws_store):
    store = temp_ws_store
    ws = store.create_workspace("Activity Test")
    ws_id = ws.workspace_id

    # Log sequential team events
    e1 = store.log_activity(ws_id, "mem_alice", "PAPER_ADDED", "PAPER", "doi:10.1001/paper1", "Alice added new foundation paper.")
    e2 = store.log_activity(ws_id, "mem_bob", "RESULT_IMPORTED", "RESULT", "run_01", "Bob imported 12 SQuAD results.")
    e3 = store.log_activity(ws_id, "mem_carol", "REVIEW_COMPLETED", "FINDING", "fnd_01", "Carol approved primary benchmark finding.")

    assert e1.event_id.startswith("act_")
    assert e2.action == "RESULT_IMPORTED"
    assert e3.target_type == "FINDING"

    activities = store.list_activity(ws_id)
    assert len(activities) == 3
    # Most recent first
    assert activities[0].actor_id == "mem_carol"
    assert activities[1].actor_id == "mem_bob"
    assert activities[2].actor_id == "mem_alice"
