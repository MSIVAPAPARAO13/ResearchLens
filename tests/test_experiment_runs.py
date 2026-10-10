"""Unit tests for ExperimentRun lifecycle and execution metadata tracking."""

import shutil
import tempfile

import pytest

from researchlens.workspace.schema import ExperimentRun
from researchlens.workspace.store import WorkspaceStore


@pytest.fixture
def temp_ws_store():
    tmp_dir = tempfile.mkdtemp(prefix="rl_test_run_")
    store = WorkspaceStore(base_dir=tmp_dir)
    yield store
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_experiment_run_lifecycle(temp_ws_store):
    store = temp_ws_store
    ws = store.create_workspace("RL Experiments")
    ws_id = ws.workspace_id

    run = ExperimentRun(
        run_id="run_ppo_01",
        experiment_id="exp_ppo_benchmark",
        run_number=1,
        status="COMPLETED",
        environment="Ubuntu 22.04 LTS, Python 3.11",
        hardware="NVIDIA A100 80GB PCIe",
        software_versions={"torch": "2.2.0", "gymnasium": "0.29.1"},
        seed=42,
        notes="Standard benchmark seed 42 run",
    )
    store.save_run(run, ws_id)

    runs = store.list_runs(ws_id)
    assert len(runs) == 1
    r = runs[0]
    assert r.status == "COMPLETED"
    assert r.hardware == "NVIDIA A100 80GB PCIe"
    assert r.seed == 42
    assert r.software_versions["torch"] == "2.2.0"

    fetched = store.get_run(ws_id, "run_ppo_01")
    assert fetched is not None
    assert fetched.run_id == "run_ppo_01"

    assert store.delete_run(ws_id, "run_ppo_01") is True
    assert len(store.list_runs(ws_id)) == 0
