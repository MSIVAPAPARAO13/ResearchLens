"""Tests for Experiment Matrix Generation (Phase 14)."""

from researchlens.execution.matrix import ExperimentMatrixGenerator


def test_experiment_matrix_combinations_and_count():
    """Verify 3 models x 2 datasets x 2 seeds produces exactly 12 planned runs with no duplicates."""
    models = ["ModelA", "ModelB", "ModelC"]
    datasets = ["DatasetX", "DatasetY"]
    seeds = [42, 123]

    count = ExperimentMatrixGenerator.calculate_matrix_size(models, datasets, seeds)
    assert count == 12

    is_valid, msg, total = ExperimentMatrixGenerator.validate_matrix(models, datasets, seeds)
    assert is_valid is True
    assert total == 12
    assert "12 experiments will be created" in msg

    # Without confirmation, plans are not generated
    plans_unconfirmed, msg_unconf = ExperimentMatrixGenerator.generate_matrix_plans(
        workspace_id="ws_matrix",
        base_name="Grid Test",
        models=models,
        datasets=datasets,
        seeds=seeds,
        confirmed=False,
    )
    assert len(plans_unconfirmed) == 0
    assert "Explicit confirmation required" in msg_unconf

    # With confirmation, plans are generated
    plans, msg_conf = ExperimentMatrixGenerator.generate_matrix_plans(
        workspace_id="ws_matrix",
        base_name="Grid Test",
        models=models,
        datasets=datasets,
        seeds=seeds,
        confirmed=True,
    )
    assert len(plans) == 12

    # Verify no duplicates
    combos = set((p.model, p.dataset, p.variables["seed"]) for p in plans)
    assert len(combos) == 12
