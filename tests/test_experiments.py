"""Unit tests for ExperimentPlan setup and CSV export."""

import shutil
import tempfile

import pytest

from researchlens.workspace.planner import ResearchPlanner
from researchlens.workspace.schema import ExperimentPlan
from researchlens.workspace.store import WorkspaceStore


@pytest.fixture
def temp_ws_store():
    tmp_dir = tempfile.mkdtemp(prefix="rl_test_exp_")
    store = WorkspaceStore(base_dir=tmp_dir)
    yield store
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_experiment_plan_lifecycle_and_csv_export(temp_ws_store):
    store = temp_ws_store
    ws = store.create_workspace("Reinforcement Learning")
    ws_id = ws.workspace_id

    e1 = ExperimentPlan(
        experiment_id="exp_01",
        name="PPO Baseline Reproduction",
        hypothesis_id="hyp_01",
        dataset="MuJoCo HalfCheetah-v4",
        model="PPO",
        baseline="TRPO",
        metrics=["Episode Reward", "Sample Efficiency"],
        variables={"learning_rate": "3e-4", "clip_ratio": "0.2"},
        expected_outcome="Reproduce reported 4500 average reward within 1M steps",
        status="PLANNED",
    )
    e2 = ExperimentPlan(
        experiment_id="exp_02",
        name="PPO with Auxiliary Value Head",
        hypothesis_id="hyp_01",
        dataset="MuJoCo HalfCheetah-v4",
        model="PPO-Aux",
        baseline="PPO",
        metrics=["Episode Reward", "Sample Efficiency"],
        variables={"aux_loss_weight": "0.1"},
        expected_outcome="15% faster convergence",
        status="PLANNED",
    )
    store.save_experiment(e1, ws_id)
    store.save_experiment(e2, ws_id)

    exps = store.list_experiments(ws_id)
    assert len(exps) == 2

    # Export to CSV
    csv_str = ResearchPlanner.export_experiments_to_csv(exps)
    assert "Experiment ID,Name,Hypothesis ID" in csv_str
    assert "PPO Baseline Reproduction" in csv_str
    assert "PPO-Aux" in csv_str
    assert "MuJoCo HalfCheetah-v4" in csv_str

    assert store.delete_experiment(ws_id, "exp_01") is True
    assert len(store.list_experiments(ws_id)) == 1
