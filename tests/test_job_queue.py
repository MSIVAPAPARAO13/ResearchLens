"""Tests for Execution Job Queue and Concurrency Management (Phase 14)."""

import tempfile
import time

from researchlens.execution.queue import ExecutionQueueManager
from researchlens.workspace.schema import ExecutionQueueItem, ExperimentRun
from researchlens.workspace.store import WorkspaceStore


def test_queue_ordering_by_priority_and_time():
    """Verify queue items are dequeued by priority (descending) and time (ascending)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = WorkspaceStore(base_dir=tmp_dir)
        ws = store.create_workspace("Queue WS", "tester")
        manager = ExecutionQueueManager(store=store)

        # Enqueue item with priority 1
        manager.enqueue(ws.workspace_id, run_id="r1", priority=1)
        time.sleep(0.01)
        # Enqueue item with priority 10
        manager.enqueue(ws.workspace_id, run_id="r2", priority=10)
        time.sleep(0.01)
        # Enqueue item with priority 5
        manager.enqueue(ws.workspace_id, run_id="r3", priority=5)

        # Dequeue should return item2 (priority 10) first
        next_job = manager.dequeue(ws.workspace_id)
        assert next_job is not None
        assert next_job.run_id == "r2"

        # Next should be item3 (priority 5)
        next_job2 = manager.dequeue(ws.workspace_id)
        assert next_job2 is not None
        assert next_job2.run_id == "r3"

        # Next should be item1 (priority 1)
        next_job3 = manager.dequeue(ws.workspace_id)
        assert next_job3 is not None
        assert next_job3.run_id == "r1"

        # No more queued items
        assert manager.dequeue(ws.workspace_id) is None


def test_queue_cancel_and_retry_lifecycle():
    """Verify queue cancel and retry operations."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = WorkspaceStore(base_dir=tmp_dir)
        ws = store.create_workspace("Queue WS", "tester")
        manager = ExecutionQueueManager(store=store)

        item = manager.enqueue(ws.workspace_id, run_id="r_cancel", priority=1)
        assert item.status == "QUEUED"

        # Cancel
        assert manager.cancel(ws.workspace_id, item.queue_id) is True
        items = manager.list_queue(ws.workspace_id)
        assert items[0].status == "CANCELLED"

        # Retry
        assert manager.retry(ws.workspace_id, item.queue_id) is True
        items = manager.list_queue(ws.workspace_id)
        assert items[0].status == "QUEUED"


def test_queue_concurrency_workers():
    """Verify bounded concurrent queue processing with 1, 2, and 4 workers."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = WorkspaceStore(base_dir=tmp_dir)
        ws = store.create_workspace("Workers WS", "tester")

        for workers in [1, 2, 4]:
            manager = ExecutionQueueManager(store=store, max_concurrent_runs=workers)
            for i in range(4):
                manager.enqueue(ws.workspace_id, run_id=f"r_w_{workers}_{i}", priority=1)

            def mock_run_fn(ws_id: str, q_item: ExecutionQueueItem):
                run = ExperimentRun(
                    run_id=q_item.run_id,
                    experiment_id="exp_test",
                    status="COMPLETED",
                )
                return run, ""

            results = manager.process_queue(ws.workspace_id, mock_run_fn)
            assert len(results) == 4
            for run_res, err in results:
                assert run_res is not None
                assert run_res.status == "COMPLETED"
