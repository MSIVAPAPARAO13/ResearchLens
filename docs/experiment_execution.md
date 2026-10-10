# Safe & Reproducible Experiment Execution

## 1. Overview

ResearchLens Phase 12 introduces a controlled, sandboxed, and auditable experiment execution layer. The execution framework bridges research planning, empirical run management, output artifact preservation, and scientific findings.

```
Experiment Plan & Entrypoint
            ↓
Environment & Hardware Validation
            ↓
Controlled Subprocess Execution (shell=False)
            ↓
Run Tracking & Log Capturing
            ↓
Output Contract Validation (results.json)
            ↓
Artifact Hashing (SHA-256)
            ↓
Canonical Manifest Generation (reproducibility.json)
            ↓
Baseline Comparison & Candidate Finding (PENDING_RESEARCHER_REVIEW)
```

---

## 2. Core Components

### 2.1 Experiment Registry (`ExperimentRegistryService`)
To prevent the execution of arbitrary or untrusted code, all experiments must be registered in the `ExperimentRegistry`.
- Allowed entrypoint locations: `scripts/experiments/`, `experiments/`, `src/researchlens/experiments/`.
- Allowed extensions: `.py` only.
- Flag `allowed: bool`: Must be set to `True` by an authorized researcher or owner.

### 2.2 Execution Profiles (`ExecutionProfile`)
Defines the target environment:
- Environment Types: `LOCAL`, `VIRTUALENV`, `DOCKER`.
- Hardware specification: `cpu` or `cuda`.
- Device check: If `cuda` is requested on a CPU-only host, execution is marked `BLOCKED`.

### 2.3 Sandboxed Execution (`ExperimentExecutor`)
- Commands are invoked as explicit argument lists with `shell=False`.
- Configurable per-experiment `timeout_seconds` (default: 3600 seconds). On timeout, process trees are terminated cleanly to prevent orphaned background tasks.
- `stdout` and `stderr` streams are captured and saved to:
  `data/workspaces/<workspace_id>/runs/<run_id>/logs/stdout.log`
  `data/workspaces/<workspace_id>/runs/<run_id>/logs/stderr.log`

### 2.4 Output Contract (`results.json`)
Every experiment must produce a `results.json` adhering to:
```json
{
  "metrics": {
    "accuracy": 0.92,
    "f1": 0.91
  },
  "metadata": {},
  "artifacts": []
}
```
Missing or invalid `results.json` fails the run with `FAILED_OUTPUT_CONTRACT`.

---

## 3. Historical Run Preservation & Re-runs

When an experiment is re-executed:
- A new, distinct `ExperimentRun` entity is created with a fresh `run_id`.
- Historical runs and their logs, manifests, and artifacts are never overwritten.
- Multi-run analytics compute empirical aggregations (`mean`, `std`, `min`, `max`) without making unverified statistical significance claims.

---

## 4. MLflow Experiment Tracking Integration (Phase 13)

Following local execution validation, runs are asynchronously synchronized to MLflow if enabled (`MLFLOW_ENABLED=true`):
- Deterministic experiment namespaces: `ResearchLens::<WorkspaceName>`
- Clean bidirectional linkage via `mlflow_run_id` and MLflow run tags (`researchlens.run_id`, `researchlens.workspace_id`).
- Sanitized parameter logging strips passwords, API keys, and credentials.
- Validated metrics logged via `ExperimentService.normalize_and_validate_metric`.
- Artifacts discoverable under run artifacts directory logged to MLflow with size capping (`MAX_MLFLOW_ARTIFACT_SIZE_MB`).
- Fault tolerance ensures execution status `COMPLETED` is maintained even if the tracking server is unreachable.

---

## 5. Execution Backends & Isolation (Phase 14)

Experiments are dispatched via configurable execution backends (`LOCAL`, `DOCKER`, `REMOTE`):
- **Local Process Backend (`LOCAL`)**: Subprocess execution with process isolation and resource monitoring.
- **Docker Backend (`DOCKER`)**: Non-root containerized execution (`--user 1000:1000`), `--cap-drop ALL`, `no-new-privileges`, CPU/memory limits, disabled network by default, and strict image allowlisting via `ExecutionImageRegistry`. Runs strictly mount `/workspace/output`.
- **Remote Backend (`REMOTE`)**: Future-ready cloud/cluster contract returning `NOT_CONFIGURED`.
- **Execution Job Specification (`ExecutionJob`)**: Persisted to `execution_job.json` prior to execution with full lifecycle tracking (`CREATED` -> `VALIDATING` -> `QUEUED` -> `PREPARING` -> `RUNNING` -> `UPLOADING` -> `COMPLETED`/`FAILED`).


