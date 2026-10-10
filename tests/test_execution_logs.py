"""Unit tests for execution stdout/stderr log capture (Phase 12)."""

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
    tmp = tempfile.mkdtemp(prefix="rl_test_logs_")
    store = WorkspaceStore(base_dir=tmp)
    ws = store.create_workspace("Logs Test Workspace")
    yield store, ws.workspace_id
    shutil.rmtree(tmp, ignore_errors=True)


def test_execution_log_capture(temp_workspace):
    store, ws_id = temp_workspace
    registry = ExperimentRegistryService(store)
    executor = ExperimentExecutor(store)

    entrypoint = "scripts/experiments/deterministic_demo.py"
    registry.register_experiment(
        workspace_id=ws_id,
        name="Deterministic Logging",
        entrypoint=entrypoint,
    )

    plan = ExperimentPlan(
        experiment_id="exp_log_01",
        name="Logging Test Plan",
        entrypoint=entrypoint,
        timeout_seconds=30,
    )
    store.save_experiment(plan, ws_id)

    run, err = executor.execute_experiment(
        workspace_id=ws_id,
        experiment_id=plan.experiment_id,
        actor_id="system",
    )

    assert run is not None
    assert run.status == "COMPLETED"

    run_dir = store.get_run_dir(ws_id, run.run_id)
    stdout_file = os.path.join(run_dir, "logs", "stdout.log")
    stderr_file = os.path.join(run_dir, "logs", "stderr.log")

    assert os.path.exists(stdout_file)
    assert os.path.exists(stderr_file)

    with open(stdout_file, "r", encoding="utf-8") as f:
        stdout_text = f.read()
    assert "Experiment completed successfully" in stdout_text
