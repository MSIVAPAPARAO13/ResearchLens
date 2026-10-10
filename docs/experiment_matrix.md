# Experiment Matrices & Hyperparameter Sweeps

## 1. Overview

ResearchLens supports controlled Cartesian grid experiment matrices across models, datasets, seeds, and bounded hyperparameters.

## 2. Guardrails & Ceilings

To prevent accidental resource exhaustion or runaway cloud/compute bills:
- `MAX_RUNS = 20`: Any matrix generating more than 20 runs is automatically rejected.
- **Explicit Confirmation**: Matrices calculate `number_of_runs` prior to creation and require an explicit `confirm=True` confirmation before generating `ExperimentRun` records.
- **No Unrestricted Sweeps**: Continuous Bayesian/random search algorithms that could launch unbounded runs are prohibited. Sweeps are strictly bounded discrete grids.

## 3. Matrix Generation Flow

```
Models: [A, B, C] (3)
Datasets: [X, Y] (2)
Seeds: [42, 123] (2)
→ Total: 3 × 2 × 2 = 12 runs (≤ 20)
→ Researcher confirms
→ Generates 12 linked ExperimentRuns
```
