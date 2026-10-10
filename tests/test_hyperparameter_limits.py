"""Tests for Hyperparameter Limits and Safety Bounds (Phase 14)."""

from researchlens.execution.matrix import (
    MAX_ALLOWED_RUNS,
    ExperimentMatrixGenerator,
)


def test_bounded_hyperparameter_grid():
    """Verify bounded grid under MAX_RUNS limit succeeds."""
    models = ["ModelA"]
    datasets = ["DatasetX"]
    seeds = [42]
    param_grid = {
        "learning_rate": [0.001, 0.0001],
        "batch_size": [16, 32],
    }

    # 1 * 1 * 1 * 2 * 2 = 4 runs <= 20
    is_valid, msg, total = ExperimentMatrixGenerator.validate_matrix(
        models, datasets, seeds, param_grid, max_runs_limit=MAX_ALLOWED_RUNS
    )
    assert is_valid is True
    assert total == 4

    plans, _ = ExperimentMatrixGenerator.generate_matrix_plans(
        workspace_id="ws_hp",
        base_name="Sweep",
        models=models,
        datasets=datasets,
        seeds=seeds,
        param_grid=param_grid,
        confirmed=True,
    )
    assert len(plans) == 4
    for p in plans:
        assert "learning_rate" in p.configuration
        assert "batch_size" in p.configuration


def test_reject_unrestricted_large_hyperparameter_matrix():
    """Verify matrices exceeding max_runs_limit are rejected."""
    models = ["ModelA", "ModelB", "ModelC"]  # 3
    datasets = ["Dataset1", "Dataset2"]       # 2
    seeds = [1, 2, 3, 4]                      # 4
    # 3 * 2 * 4 = 24 runs > 20

    is_valid, msg, total = ExperimentMatrixGenerator.validate_matrix(
        models, datasets, seeds, max_runs_limit=20
    )
    assert is_valid is False
    assert total == 24
    assert "exceeds maximum allowable limit" in msg

    # Generating plans also fails
    plans, gen_err = ExperimentMatrixGenerator.generate_matrix_plans(
        workspace_id="ws_hp",
        base_name="Sweep",
        models=models,
        datasets=datasets,
        seeds=seeds,
        confirmed=True,
        max_runs_limit=20,
    )
    assert len(plans) == 0
    assert "exceeds maximum allowable limit" in gen_err
