"""Controlled Experiment Matrices and Bounded Hyperparameter Sweeps (Phase 14)."""

import itertools
from typing import Any, Dict, List, Optional, Tuple

from researchlens.workspace.schema import (
    ExperimentPlan,
    current_iso_time,
    generate_id,
)

MAX_ALLOWED_RUNS = 20


class ExperimentMatrixGenerator:
    """Generates controlled Cartesian product experiment matrices with safety limits and explicit confirmation."""

    @staticmethod
    def calculate_matrix_size(
        models: List[str],
        datasets: List[str],
        seeds: List[int],
        param_grid: Optional[Dict[str, List[Any]]] = None,
    ) -> int:
        """Calculate total number of runs for the proposed combination."""
        base_count = max(1, len(models)) * max(1, len(datasets)) * max(1, len(seeds))
        grid_combinations = 1
        if param_grid:
            for vals in param_grid.values():
                if vals:
                    grid_combinations *= len(vals)
        return base_count * grid_combinations

    @staticmethod
    def validate_matrix(
        models: List[str],
        datasets: List[str],
        seeds: List[int],
        param_grid: Optional[Dict[str, List[Any]]] = None,
        max_runs_limit: int = MAX_ALLOWED_RUNS,
    ) -> Tuple[bool, str, int]:
        """Validate proposed matrix against upper bound limit."""
        if not models:
            return False, "At least one model must be specified.", 0
        if not datasets:
            return False, "At least one dataset must be specified.", 0
        if not seeds:
            return False, "At least one seed must be specified.", 0

        total_runs = ExperimentMatrixGenerator.calculate_matrix_size(models, datasets, seeds, param_grid)
        if total_runs > max_runs_limit:
            return (
                False,
                f"Proposed matrix size ({total_runs} runs) exceeds maximum allowable limit ({max_runs_limit} runs). "
                "Rejecting large unrestricted hyperparameter sweeps.",
                total_runs,
            )

        return True, f"{total_runs} experiments will be created.", total_runs

    @staticmethod
    def generate_matrix_plans(
        workspace_id: str,
        base_name: str,
        models: List[str],
        datasets: List[str],
        seeds: List[int],
        param_grid: Optional[Dict[str, List[Any]]] = None,
        entrypoint: str = "scripts/experiments/deterministic_demo.py",
        backend: str = "LOCAL",
        confirmed: bool = False,
        max_runs_limit: int = MAX_ALLOWED_RUNS,
    ) -> Tuple[List[ExperimentPlan], str]:
        """Generate ExperimentPlan configurations for confirmed matrices."""
        is_valid, msg, total_runs = ExperimentMatrixGenerator.validate_matrix(
            models, datasets, seeds, param_grid, max_runs_limit
        )
        if not is_valid:
            return [], msg

        if not confirmed:
            return [], f"Explicit confirmation required: {total_runs} experiments will be created. Confirmation: FALSE."

        plans: List[ExperimentPlan] = []

        # Flatten parameter grid into list of dicts
        grid_keys = list(param_grid.keys()) if param_grid else []
        grid_values = [param_grid[k] for k in grid_keys] if param_grid else []
        param_combinations = [dict(zip(grid_keys, combo)) for combo in itertools.product(*grid_values)] if grid_combinations_exist(param_grid) else [{}]

        idx = 1
        for model, dataset, seed, p_dict in itertools.product(models, datasets, seeds, param_combinations):
            config = dict(p_dict)
            config["execution_backend"] = backend

            name = f"{base_name} - {model} on {dataset} (Seed {seed})"
            if p_dict:
                param_suffix = ", ".join(f"{k}={v}" for k, v in p_dict.items())
                name += f" [{param_suffix}]"

            plan = ExperimentPlan(
                experiment_id=generate_id("exp"),
                name=name,
                model=model,
                dataset=dataset,
                variables={"seed": str(seed), "matrix_index": str(idx)},
                configuration=config,
                entrypoint=entrypoint,
                execution_backend=backend,
                status="PLANNED",
                created_at=current_iso_time(),
            )
            plans.append(plan)
            idx += 1

        return plans, f"Successfully generated {len(plans)} experiment configurations."


def grid_combinations_exist(param_grid: Optional[Dict[str, List[Any]]]) -> bool:
    if not param_grid:
        return False
    return any(bool(v) for v in param_grid.values())
