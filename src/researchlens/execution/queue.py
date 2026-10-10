"""Execution Queue and Concurrency Management (Phase 14)."""

import concurrent.futures
import threading
from typing import Callable, List, Optional, Tuple

from researchlens.utils.logging import get_logger
from researchlens.workspace.schema import (
    ExecutionQueueItem,
    ExperimentRun,
    current_iso_time,
    generate_id,
)
from researchlens.workspace.store import WorkspaceStore

logger = get_logger("execution_queue")


class ExecutionQueueManager:
    """Manages sequential and bounded-concurrency execution of scheduled experiment runs."""

    def __init__(self, store: WorkspaceStore, max_concurrent_runs: int = 1):
        self.store = store
        self.max_concurrent_runs = max(1, max_concurrent_runs)
        self._lock = threading.Lock()

    def enqueue(
        self,
        workspace_id: str,
        run_id: str,
        job_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        priority: int = 1,
        backend: str = "LOCAL",
    ) -> ExecutionQueueItem:
        """Enqueue an experiment run for scheduled execution."""
        item = ExecutionQueueItem(
            queue_id=generate_id("q"),
            run_id=run_id,
            job_id=job_id,
            experiment_id=experiment_id,
            priority=priority,
            backend=backend,
            status="QUEUED",
            created_at=current_iso_time(),
        )
        self.store.save_queue_item(item, workspace_id)
        return item

    def list_queue(self, workspace_id: str) -> List[ExecutionQueueItem]:
        """List queue items sorted by priority (descending) and created_at (ascending)."""
        items = self.store.list_queue_items(workspace_id)
        return sorted(items, key=lambda x: (-x.priority, x.created_at))

    def dequeue(self, workspace_id: str) -> Optional[ExecutionQueueItem]:
        """Atomically pop the next QUEUED job from the queue."""
        with self._lock:
            items = self.list_queue(workspace_id)
            for item in items:
                if item.status == "QUEUED":
                    item.status = "PREPARING"
                    self.store.save_queue_item(item, workspace_id)
                    return item
        return None

    def update_status(self, workspace_id: str, queue_id: str, status: str) -> bool:
        """Update status of a queue item."""
        with self._lock:
            items = self.store.list_queue_items(workspace_id)
            for item in items:
                if item.queue_id == queue_id:
                    item.status = status
                    self.store.save_queue_item(item, workspace_id)
                    return True
        return False

    def cancel(self, workspace_id: str, queue_id: str) -> bool:
        """Cancel a queued or running job."""
        return self.update_status(workspace_id, queue_id, "CANCELLED")

    def retry(self, workspace_id: str, queue_id: str) -> bool:
        """Re-queue a failed, cancelled, or timed-out job."""
        return self.update_status(workspace_id, queue_id, "QUEUED")

    def run_worker_task(
        self,
        workspace_id: str,
        item: ExecutionQueueItem,
        run_fn: Callable[[str, ExecutionQueueItem], Tuple[Optional[ExperimentRun], str]],
    ) -> Tuple[Optional[ExperimentRun], str]:
        """Execute a single queue item via the provided executor function."""
        self.update_status(workspace_id, item.queue_id, "RUNNING")
        try:
            run_obj, err = run_fn(workspace_id, item)
            if run_obj and run_obj.status == "COMPLETED":
                self.update_status(workspace_id, item.queue_id, "COMPLETED")
            elif run_obj and run_obj.status in ["FAILED", "TIMEOUT", "CANCELLED", "BLOCKED"]:
                self.update_status(workspace_id, item.queue_id, run_obj.status)
            else:
                self.update_status(workspace_id, item.queue_id, "FAILED" if err else "COMPLETED")
            return run_obj, err
        except Exception as e:
            self.update_status(workspace_id, item.queue_id, "FAILED")
            return None, str(e)

    def process_queue(
        self,
        workspace_id: str,
        run_fn: Callable[[str, ExecutionQueueItem], Tuple[Optional[ExperimentRun], str]],
        max_runs: Optional[int] = None,
    ) -> List[Tuple[Optional[ExperimentRun], str]]:
        """Process queued items using bounded concurrent execution (default: max_concurrent_runs)."""
        pending = [item for item in self.list_queue(workspace_id) if item.status == "QUEUED"]
        if max_runs is not None:
            pending = pending[:max_runs]

        if not pending:
            return []

        results: List[Tuple[Optional[ExperimentRun], str]] = []

        if self.max_concurrent_runs <= 1:
            # Sequential single-process queue
            for item in pending:
                res = self.run_worker_task(workspace_id, item, run_fn)
                results.append(res)
        else:
            # Bounded concurrent execution (tested with 1, 2, 4 workers)
            workers = min(self.max_concurrent_runs, len(pending))
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(self.run_worker_task, workspace_id, item, run_fn)
                    for item in pending
                ]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        results.append((None, str(e)))

        return results
