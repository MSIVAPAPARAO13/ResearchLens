"""Unit tests for output contract validation and result ingestion (Phase 12)."""

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
    tmp = tempfile.mkdtemp(prefix="rl_test_res_")
    store = WorkspaceStore(base_dir=tmp)
    ws = store.create_workspace("Results Test Workspace")
    yield store, ws.workspace_id
    shutil.rmtree(tmp, ignore_errors=True)


def test_missing_results_json_fails_output_contract(temp_workspace):
    store, ws_id = temp_workspace
    registry = ExperimentRegistryService(store)
    executor = ExperimentExecutor(store)

    # Script that exits cleanly but does NOT produce results.json
    script_path = "scripts/experiments/no_output_script.py"
    os.makedirs(os.path.dirname(script_path), exist_ok=True)
    with open(script_path, "w", encoding="utf-8") as f:
        f.write("print('Hello without creating results.json')\n")

    try:
        registry.register_experiment(
            workspace_id=ws_id,
            name="No Output Experiment",
            entrypoint=script_path,
        )

        plan = ExperimentPlan(
            experiment_id="exp_contract_01",
            name="Missing Output Plan",
            entrypoint=script_path,
        )
        store.save_experiment(plan, ws_id)

        run, err = executor.execute_experiment(
            workspace_id=ws_id,
            experiment_id=plan.experiment_id,
            actor_id="system",
        )

        assert run is not None
        assert run.status == "FAILED"
        assert run.failure_category == "FAILED_OUTPUT_CONTRACT"
        assert "Output contract failed" in err
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)
