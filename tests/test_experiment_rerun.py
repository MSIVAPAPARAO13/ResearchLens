"""Unit tests for experiment re-runs and historical run preservation (Phase 12)."""

import shutil
import tempfile

import pytest

from researchlens.execution.executor import ExperimentExecutor
from researchlens.execution.registry import ExperimentRegistryService
from researchlens.workspace.schema import ExperimentPlan
from researchlens.workspace.store import WorkspaceStore


@pytest.fixture
def temp_workspace():
    tmp = tempfile.mkdtemp(prefix="rl_test_rerun_")
    store = WorkspaceStore(base_dir=tmp)
    ws = store.create_workspace("Rerun Test Workspace")
    yield store, ws.workspace_id
    shutil.rmtree(tmp, ignore_errors=True)


def test_experiment_rerun_creates_new_run_record(temp_workspace):
    store, ws_id = temp_workspace
    registry = ExperimentRegistryService(store)
    executor = ExperimentExecutor(store)

    entrypoint = "scripts/experiments/deterministic_demo.py"
    registry.register_experiment(
        workspace_id=ws_id,
        name="Rerun Deterministic Demo",
        entrypoint=entrypoint,
    )

    plan = ExperimentPlan(
        experiment_id="exp_rerun_01",
        name="Rerun Test Plan",
        entrypoint=entrypoint,
        dataset="SQuAD_v2",
    )
    store.save_experiment(plan, ws_id)

    # 1. Run 1
    run1, err1 = executor.execute_experiment(
        workspace_id=ws_id,
        experiment_id=plan.experiment_id,
        actor_id="system",
        seed=42,
    )
    assert run1 is not None
    assert err1 == ""
    assert run1.status == "COMPLETED"

    # 2. Re-run (Run 2) with identical seed
    run2, err2 = executor.execute_experiment(
        workspace_id=ws_id,
        experiment_id=plan.experiment_id,
        actor_id="system",
        seed=42,
    )
    assert run2 is not None
    assert err2 == ""
    assert run2.status == "COMPLETED"

    # 3. Verify runs are distinct entities and run 1 is preserved
    assert run1.run_id != run2.run_id
    all_runs = store.list_runs(ws_id)
    assert len(all_runs) == 2

    run_ids = {r.run_id for r in all_runs}
    assert run1.run_id in run_ids
    assert run2.run_id in run_ids

    # 4. Verify deterministic output (same metrics across both runs)
    results_run1 = store.get_results_for_run(ws_id, run1.run_id)
    results_run2 = store.get_results_for_run(ws_id, run2.run_id)

    metrics_run1 = {r.metric: r.value for r in results_run1}
    metrics_run2 = {r.metric: r.value for r in results_run2}

    assert metrics_run1["accuracy"] == metrics_run2["accuracy"]
    assert metrics_run1["f1"] == metrics_run2["f1"]
