"""Unit tests for experiment timeout termination (Phase 12)."""

import os
import shutil
import tempfile

import pytest

from researchlens.execution.executor import ExperimentExecutor
from researchlens.execution.registry import ExperimentRegistryService
from researchlens.workspace.schema import ExperimentPlan
from researchlens.workspace.store import WorkspaceStore


@pytest.fixture
def temp_workspace():
    tmp = tempfile.mkdtemp(prefix="rl_test_timeout_")
    store = WorkspaceStore(base_dir=tmp)
    ws = store.create_workspace("Timeout Test Workspace")
    yield store, ws.workspace_id
    shutil.rmtree(tmp, ignore_errors=True)


def test_experiment_timeout_enforcement(temp_workspace):
    store, ws_id = temp_workspace
    registry = ExperimentRegistryService(store)
    executor = ExperimentExecutor(store)

    # 1. Create a script in approved directory that sleeps
    script_path = "scripts/experiments/sleep_mock.py"
    os.makedirs(os.path.dirname(script_path), exist_ok=True)
    with open(script_path, "w", encoding="utf-8") as f:
        f.write("import time\ntime.sleep(10)\n")

    try:
        # Register script
        registry.register_experiment(
            workspace_id=ws_id,
            name="Sleeping Script",
            entrypoint=script_path,
        )

        plan = ExperimentPlan(
            experiment_id="exp_timeout_01",
            name="Timeout Test Plan",
            entrypoint=script_path,
            timeout_seconds=1,  # 1 second timeout
        )
        store.save_experiment(plan, ws_id)

        # Execute with 1 second timeout
        run, err = executor.execute_experiment(
            workspace_id=ws_id,
            experiment_id=plan.experiment_id,
            actor_id="system",
        )

        assert run is not None
        assert run.status == "TIMEOUT"
        assert run.failure_category == "TIMEOUT"
        assert "timed out" in err.lower()
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)
