"""Unit tests for role-based execution and cancellation permissions (Phase 12)."""

import shutil
import tempfile

import pytest

from researchlens.execution.executor import ExperimentExecutor
from researchlens.execution.registry import ExperimentRegistryService
from researchlens.workspace.schema import ExperimentPlan, WorkspaceMember
from researchlens.workspace.store import WorkspaceStore


@pytest.fixture
def temp_workspace():
    tmp = tempfile.mkdtemp(prefix="rl_test_perm_")
    store = WorkspaceStore(base_dir=tmp)
    ws = store.create_workspace("Permissions Test Workspace")

    # Add members with different roles
    store.save_member(WorkspaceMember("m_owner", ws.workspace_id, "Alice", "alice@lab.org", "OWNER"))
    store.save_member(WorkspaceMember("m_researcher", ws.workspace_id, "Bob", "bob@lab.org", "RESEARCHER"))
    store.save_member(WorkspaceMember("m_reviewer", ws.workspace_id, "Carol", "carol@lab.org", "REVIEWER"))
    store.save_member(WorkspaceMember("m_viewer", ws.workspace_id, "David", "david@lab.org", "VIEWER"))

    yield store, ws.workspace_id
    shutil.rmtree(tmp, ignore_errors=True)


def test_execution_permissions_by_role(temp_workspace):
    store, ws_id = temp_workspace
    registry = ExperimentRegistryService(store)
    executor = ExperimentExecutor(store)

    entrypoint = "scripts/experiments/deterministic_demo.py"
    registry.register_experiment(
        workspace_id=ws_id,
        name="Perm Check Experiment",
        entrypoint=entrypoint,
        owner="m_owner",
    )

    plan = ExperimentPlan(
        experiment_id="exp_perm_01",
        name="Perm Plan",
        entrypoint=entrypoint,
    )
    store.save_experiment(plan, ws_id)

    # 1. VIEWER rejected
    run_v, err_v = executor.execute_experiment(ws_id, plan.experiment_id, actor_id="m_viewer")
    assert run_v is None
    assert "Permission denied" in err_v

    # 2. REVIEWER rejected from executing (only review authority)
    run_rev, err_rev = executor.execute_experiment(ws_id, plan.experiment_id, actor_id="m_reviewer")
    assert run_rev is None
    assert "Permission denied" in err_rev

    # 3. RESEARCHER allowed
    run_r, err_r = executor.execute_experiment(ws_id, plan.experiment_id, actor_id="m_researcher")
    assert err_r == ""
    assert run_r is not None
    assert run_r.status == "COMPLETED"
