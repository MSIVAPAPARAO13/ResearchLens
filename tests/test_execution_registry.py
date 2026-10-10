"""Unit tests for Experiment Registry service (Phase 12)."""

import shutil
import tempfile

import pytest

from researchlens.execution.registry import ExperimentRegistryService
from researchlens.workspace.store import WorkspaceStore


@pytest.fixture
def temp_workspace():
    tmp = tempfile.mkdtemp(prefix="rl_test_reg_")
    store = WorkspaceStore(base_dir=tmp)
    ws = store.create_workspace("Registry Test Workspace")
    yield store, ws.workspace_id
    shutil.rmtree(tmp, ignore_errors=True)


def test_registry_registration_and_validation(temp_workspace):
    store, ws_id = temp_workspace
    registry = ExperimentRegistryService(store)

    # 1. Successful registration of valid script in approved directory
    entry, err = registry.register_experiment(
        workspace_id=ws_id,
        name="Deterministic Benchmark Demo",
        entrypoint="scripts/experiments/deterministic_demo.py",
        description="Deterministic baseline evaluator",
    )
    assert entry is not None
    assert err == ""
    assert entry.allowed is True

    # 2. Check execution allowed
    is_allowed, check_err = registry.validate_execution_allowed(
        ws_id, "scripts/experiments/deterministic_demo.py"
    )
    assert is_allowed is True
    assert check_err == ""

    # 3. Rejection of unapproved entrypoint
    is_allowed, check_err = registry.validate_execution_allowed(
        ws_id, "scripts/experiments/unregistered_script.py"
    )
    assert is_allowed is False
    assert "not registered" in check_err

    # 4. Rejection of non-existent file
    bad_entry, bad_err = registry.register_experiment(
        workspace_id=ws_id,
        name="Missing Script",
        entrypoint="scripts/experiments/does_not_exist.py",
    )
    assert bad_entry is None
    assert "does not exist" in bad_err


def test_registry_list_and_toggle_allowed(temp_workspace):
    store, ws_id = temp_workspace
    registry = ExperimentRegistryService(store)

    entry, err = registry.register_experiment(
        workspace_id=ws_id,
        name="Custom Version Benchmark",
        entrypoint="scripts/experiments/deterministic_demo.py",
        version="2.1.0",
        owner="lead_researcher",
    )
    assert entry is not None
    assert err == ""
    assert entry.version == "2.1.0"

    # Test list_registered_experiments alias
    all_entries = registry.list_registered_experiments(ws_id)
    assert len(all_entries) == 1
    assert all_entries[0].experiment_id == entry.experiment_id

    # Test toggle_allowed to False
    ok = registry.toggle_allowed(ws_id, entry.experiment_id, False)
    assert ok is True
    updated = registry.get_experiment(ws_id, entry.experiment_id)
    assert updated.allowed is False

    is_allowed, msg = registry.validate_execution_allowed(ws_id, "scripts/experiments/deterministic_demo.py")
    assert is_allowed is False

    # Test toggle_allowed back to True
    ok = registry.toggle_allowed(ws_id, entry.experiment_id, True)
    assert ok is True
    updated = registry.get_experiment(ws_id, entry.experiment_id)
    assert updated.allowed is True

    is_allowed, msg = registry.validate_execution_allowed(ws_id, "scripts/experiments/deterministic_demo.py")
    assert is_allowed is True
