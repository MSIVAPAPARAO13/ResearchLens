# Continuous Integration & Regression Detection

## 1. Overview

ResearchLens integrates automated evaluation pipelines into Continuous Integration (CI) to ensure algorithmic correctness and guard against metric degradation across Pull Requests without incurring expensive GPU training cycles.

## 2. GitHub Actions Workflow

Located at `.github/workflows/researchlens-evaluation.yml`:
1. `python -m compileall src app tests scripts evaluation` (Bytecode integrity)
2. `python -m ruff check .` (Lint and import verification)
3. `pytest` test suite execution across backends, security validators, and tracking modules
4. Targeted CPU evaluation benchmark (`scripts/run_evaluation.py`)

## 3. Regression Detection Thresholds

Managed by `RegressionDetector` in `src/researchlens/execution/regression.py`:
- `mrr`: Maximum permissible drop `0.05`
- `f1`: Maximum permissible drop `0.05`
- `accuracy`: Maximum permissible drop `0.05`
- `precision`: Maximum permissible drop `0.05`
- `recall`: Maximum permissible drop `0.05`

If a metric drops beyond the permitted threshold:
- Status: `REGRESSION_DETECTED`
- Warnings recorded in `evaluation.json`
- Code is flagged for human researcher review rather than silently accepted or prematurely rejected.
