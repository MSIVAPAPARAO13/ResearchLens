"""Unit tests for GPU detection and device requirement handling (Phase 12)."""

import shutil
import tempfile

import pytest

from researchlens.execution.environment import get_gpu_info
from researchlens.execution.executor import ExperimentExecutor
from researchlens.execution.registry import ExperimentRegistryService
from researchlens.workspace.schema import ExecutionProfile, ExperimentPlan
from researchlens.workspace.store import WorkspaceStore


@pytest.fixture
def temp_workspace():
    tmp = tempfile.mkdtemp(prefix="rl_test_gpu_")
    store = WorkspaceStore(base_dir=tmp)
    ws = store.create_workspace("GPU Test Workspace")
    yield store, ws.workspace_id
    shutil.rmtree(tmp, ignore_errors=True)


def test_get_gpu_info_structure():
    info = get_gpu_info()
    assert "available" in info
    assert "name" in info
    assert "vram_gb" in info
    assert "cuda_version" in info
    assert isinstance(info["available"], bool)


def test_gpu_unavailable_blocks_run_when_required(temp_workspace):
    store, ws_id = temp_workspace
    registry = ExperimentRegistryService(store)
    executor = ExperimentExecutor(store)

    entrypoint = "scripts/experiments/deterministic_demo.py"
    registry.register_experiment(
        workspace_id=ws_id,
        name="GPU Dependent Experiment",
        entrypoint=entrypoint,
    )

    plan = ExperimentPlan(
        experiment_id="exp_gpu_01",
        name="GPU Demanding Plan",
        entrypoint=entrypoint,
    )
    store.save_experiment(plan, ws_id)

    # Create profile that strictly demands a non-existent specialized GPU if host has no GPU
    profile = ExecutionProfile(
        profile_id="prof_strict_gpu",
        name="NVIDIA H100 Required",
        gpu="NVIDIA H100 80GB SXM5",
    )
    store.save_execution_profile(profile, ws_id)

    # If host has no GPU available, executor must mark BLOCKED
    gpu_state = get_gpu_info()
    if not gpu_state.get("available", False):
        run, err = executor.execute_experiment(
            workspace_id=ws_id,
            experiment_id=plan.experiment_id,
            profile_id=profile.profile_id,
            actor_id="system",
        )
        assert run is not None
        assert run.status == "BLOCKED"
        assert "GPU is not available" in err
