"""Tests for Reproducibility Diff and Drift Detection (Phase 14)."""

import tempfile

from researchlens.execution.executor import ExperimentExecutor
from researchlens.execution.reproducibility import (
    ReproducibilityDiffService,
    ReproducibilityDriftDetector,
)
from researchlens.workspace.schema import (
    ExperimentPlan,
    ExperimentRun,
)
from researchlens.workspace.store import WorkspaceStore


def test_reproducibility_diff_identical_runs():
    """Verify differential comparison of two identical runs returns MATCH."""
    run1 = ExperimentRun(
        run_id="run_1",
        experiment_id="exp_1",
        seed=42,
        git_commit_sha="abcdef123456",
        environment="Linux Python 3.11",
        execution_backend="LOCAL",
    )
    run2 = ExperimentRun(
        run_id="run_2",
        experiment_id="exp_1",
        seed=42,
        git_commit_sha="abcdef123456",
        environment="Linux Python 3.11",
        execution_backend="LOCAL",
    )

    manifest = {
        "dataset_hash": "hash_dataset_123",
        "model_version": "1.0",
        "artifact_hashes": {"results.json": "hash_res_123"},
    }
    metrics = {"accuracy": 0.885, "f1": 0.870}

    diff = ReproducibilityDiffService.compare_runs(
        original_run=run1,
        replay_run=run2,
        original_manifest=manifest,
        replay_manifest=manifest,
        original_metrics=metrics,
        replay_metrics=metrics,
    )

    assert diff["overall_reproduced"] is True
    assert diff["categories"]["seed"]["status"] == "MATCH"
    assert diff["categories"]["code_commit"]["status"] == "MATCH"
    assert diff["categories"]["dataset_hash"]["status"] == "MATCH"
    assert diff["metrics"]["accuracy"]["status"] == "MATCH"


def test_reproducibility_diff_detects_differences():
    """Verify differential comparison flags parameter and metric deviations."""
    run1 = ExperimentRun(run_id="r1", experiment_id="e1", seed=42, git_commit_sha="c1")
    run2 = ExperimentRun(run_id="r2", experiment_id="e1", seed=123, git_commit_sha="c2")

    diff = ReproducibilityDiffService.compare_runs(
        original_run=run1,
        replay_run=run2,
        original_metrics={"accuracy": 0.90},
        replay_metrics={"accuracy": 0.80},
    )

    assert diff["overall_reproduced"] is False
    assert diff["categories"]["seed"]["status"] == "DIFFERENCE"
    assert diff["categories"]["code_commit"]["status"] == "DIFFERENCE"
    assert diff["metrics"]["accuracy"]["status"] == "DIFFERENCE"


def test_reproducibility_drift_detection():
    """Verify drift detector identifies dataset, code, model, and config modifications."""
    manifest = {
        "dataset_hash": "orig_ds_hash",
        "code_commit": "orig_git_sha",
        "model_version": "1.0",
        "configuration": {"learning_rate": 0.001},
    }

    drift_events = ReproducibilityDriftDetector.detect_drift(
        original_manifest=manifest,
        current_dataset_hash="new_ds_hash",
        current_code_commit="new_git_sha",
        current_model_version="2.0",
        current_configuration={"learning_rate": 0.01},
    )

    drift_statuses = [d["status"] for d in drift_events]
    assert "DATASET_CHANGED" in drift_statuses
    assert "CODE_CHANGED" in drift_statuses
    assert "MODEL_CHANGED" in drift_statuses
    assert "CONFIGURATION_CHANGED" in drift_statuses


def test_reproduce_run_creates_new_run_never_mutates_original():
    """Verify reproduce_run creates a brand new run record without mutating original."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = WorkspaceStore(base_dir=tmp_dir)
        ws = store.create_workspace("Reproduce WS")
        from researchlens.execution.registry import ExperimentRegistryService
        reg_service = ExperimentRegistryService(store)
        reg_service.register_experiment(
            workspace_id=ws.workspace_id,
            name="Deterministic Baseline",
            entrypoint="scripts/experiments/deterministic_demo.py",
        )
        exp = ExperimentPlan(
            experiment_id="exp_rep",
            name="Deterministic Baseline",
            entrypoint="scripts/experiments/deterministic_demo.py",
        )
        store.save_experiment(exp, ws.workspace_id)

        executor = ExperimentExecutor(store=store)

        # 1. Run original
        orig_run, err1 = executor.execute_experiment(
            workspace_id=ws.workspace_id,
            experiment_id=exp.experiment_id,
            seed=42,
        )
        assert orig_run is not None
        assert orig_run.status == "COMPLETED"

        # 2. Reproduce run
        replay_run, err2 = executor.reproduce_run(
            workspace_id=ws.workspace_id,
            original_run_id=orig_run.run_id,
        )
        assert replay_run is not None
        assert replay_run.run_id != orig_run.run_id
        assert replay_run.seed == orig_run.seed

        # Check original run is completely unmodified
        reloaded_orig = store.get_run(ws.workspace_id, orig_run.run_id)
        assert reloaded_orig.run_id == orig_run.run_id
        assert "Replay" not in reloaded_orig.notes
