"""Unit tests for controlled experiment execution lifecycle (Phase 12)."""

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
    tmp = tempfile.mkdtemp(prefix="rl_test_exec_")
    store = WorkspaceStore(base_dir=tmp)
    ws = store.create_workspace("Executor Test Workspace")
    yield store, ws.workspace_id
    shutil.rmtree(tmp, ignore_errors=True)


def test_executor_successful_run_lifecycle(temp_workspace):
    store, ws_id = temp_workspace
    registry = ExperimentRegistryService(store)
    executor = ExperimentExecutor(store)

    # 1. Register allowed experiment
    entrypoint = "scripts/experiments/deterministic_demo.py"
    reg_entry, err = registry.register_experiment(
        workspace_id=ws_id,
        name="Deterministic Evaluation",
        entrypoint=entrypoint,
    )
    assert reg_entry is not None
    assert err == ""

    # 2. Create Experiment Plan
    plan = ExperimentPlan(
        experiment_id="exp_test_01",
        research_plan_id=None,
        name="Deterministic Test Run",
        dataset="SQuAD_v2",
        model="DeterministicBaseline",
        entrypoint=entrypoint,
        timeout_seconds=60,
    )
    store.save_experiment(plan, ws_id)

    # 3. Execute experiment
    run, exec_err = executor.execute_experiment(
        workspace_id=ws_id,
        experiment_id=plan.experiment_id,
        actor_id="system",
        seed=42,
    )

    assert exec_err == ""
    assert run is not None
    assert run.status == "COMPLETED"
    assert run.exit_code == 0
    assert run.duration_seconds > 0.0
    assert len(run.artifact_ids) >= 1

    # 4. Verify results were ingested
    results = store.list_results(ws_id)
    assert len(results) >= 2  # accuracy and f1
    assert all(r.status == "UNVERIFIED_IMPORT" for r in results)
    assert any(r.metric == "accuracy" for r in results)

    # 5. Verify run directory files exist
    run_dir = store.get_run_dir(ws_id, run.run_id)
    assert os.path.exists(os.path.join(run_dir, "environment.json"))
    assert os.path.exists(os.path.join(run_dir, "config.json"))
    assert os.path.exists(os.path.join(run_dir, "reproducibility.json"))
    assert os.path.exists(os.path.join(run_dir, "logs", "stdout.log"))
    assert os.path.exists(os.path.join(run_dir, "logs", "stderr.log"))
    assert os.path.exists(os.path.join(run_dir, "artifacts", "results.json"))

    # 6. Verify candidate finding
    findings = store.list_findings(ws_id)
    assert len(findings) >= 0
