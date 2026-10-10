"""Unit tests for DiscussionThread and collaborative discussion messages."""

import shutil
import tempfile

import pytest

from researchlens.workspace.schema import DiscussionMessage, DiscussionThread
from researchlens.workspace.store import WorkspaceStore


@pytest.fixture
def temp_ws_store():
    tmp_dir = tempfile.mkdtemp(prefix="rl_test_thd_")
    store = WorkspaceStore(base_dir=tmp_dir)
    yield store
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_discussion_thread_and_messages(temp_ws_store):
    store = temp_ws_store
    ws = store.create_workspace("Vision Models")
    ws_id = ws.workspace_id

    # 1. Create thread
    thd = DiscussionThread(
        thread_id="thd_01",
        workspace_id=ws_id,
        target_type="EXPERIMENT",
        target_id="exp_vit_benchmark",
        title="Discrepancy in ImageNet Top-1 Reproduction",
    )
    store.save_thread(thd)

    threads = store.list_threads(ws_id)
    assert len(threads) == 1
    assert threads[0].status == "OPEN"

    # 2. Add messages to thread
    m1 = DiscussionMessage("msg_01", "thd_01", "mem_alice", "Are we evaluating on the test or val split?")
    m2 = DiscussionMessage("msg_02", "thd_01", "mem_bob", "We evaluated on val split, following original paper protocol.")

    assert store.add_message_to_thread(ws_id, "thd_01", m1) is True
    assert store.add_message_to_thread(ws_id, "thd_01", m2) is True

    # 3. Retrieve and verify messages
    updated_threads = store.list_threads(ws_id)
    assert len(updated_threads[0].messages) == 2
    assert updated_threads[0].messages[0].author_id == "mem_alice"
    assert updated_threads[0].messages[1].author_id == "mem_bob"

    # 4. Resolve thread
    updated_threads[0].status = "RESOLVED"
    store.save_thread(updated_threads[0])
    assert store.list_threads(ws_id)[0].status == "RESOLVED"
