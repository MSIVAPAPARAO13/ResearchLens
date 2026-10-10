# MLflow Integration Architecture in ResearchLens

## 1. Overview & Separation of Concerns

ResearchLens integrates with MLflow as a specialized experiment tracking and model registry backend. Crucially, **MLflow does NOT replace ResearchLens domain entities or execution engines**.

| Dimension | ResearchLens Role (Authoritative) | MLflow Role (Integration Backend) |
|---|---|---|
| **Scientific Governance** | Research questions, hypotheses, gaps, findings, human review | None (MLflow does not decide scientific comparability) |
| **Execution Authority** | Controlled subprocess execution, sandboxing, environment capture, output contract validation | Metadata consumer post-execution |
| **Provenance & Hashing** | Authoritative SHA-256 artifact hashing, reproducibility manifest | Secondary artifact storage and retrieval URI |
| **Entity Storage** | `data/workspaces/<ws_id>/` JSON datastore | Local file store (`./mlruns`) or remote tracking server |
| **Model Registry** | Scientific stage governance (`CANDIDATE` -> `VALIDATED` -> `PRODUCTION`), human review mandatory | Model tracking, parameter/metric association |

---

## 2. Bidirectional Mapping Architecture

ResearchLens maintains persistent cross-navigation references to MLflow:

```
ResearchLens Workspace (e.g. "Alzheimer MRI")
       │
       ▼
MLflow Experiment ("ResearchLens::Alzheimer MRI")
       │
       ├──────────────────────────────────────────────┐
       ▼                                              ▼
ResearchLens ExperimentRun                     MLflow Run
   ├── run_id: "run_abc"                          ├── run_id: "mlflow_xyz"
   ├── mlflow_run_id: "mlflow_xyz"                └── tags:
   ├── mlflow_experiment_id: "exp_1"                   ├── researchlens.workspace_id
   └── mlflow_sync_status: "SYNCED"                   ├── researchlens.experiment_id
                                                       └── researchlens.run_id
```

### Required MLflow Tags
Every run synced to MLflow is stamped with:
- `researchlens.workspace_id`: Local workspace identifier
- `researchlens.experiment_id`: Experiment plan identifier
- `researchlens.run_id`: Authoritative ResearchLens run ID
- `researchlens.hypothesis_id`: Associated hypothesis (when available)
- `researchlens.question_id`: Associated research question (when available)
- `researchlens.dataset_id`: Dataset name or identifier
- `researchlens.model_id`: Model name or HuggingFace ID
- `researchlens.code_commit`: Git commit SHA

*Note: Identifiers are never fabricated. Only extant values are logged.*

---

## 3. Configuration & Local-First Mode

MLflow configuration is managed via `MLflowConfig` with environment variable overrides:

| Setting | Default | Description |
|---|---|---|
| `MLFLOW_ENABLED` | `false` | Master toggle. When false, MLflow is completely bypassed. |
| `MLFLOW_TRACKING_URI` | `./mlruns` | File-based local tracking directory or server URL. |
| `MLFLOW_EXPERIMENT_PREFIX` | `ResearchLens::` | Prefix for deterministic experiment namespaces. |
| `MLFLOW_ARTIFACT_LOCATION` | `""` | Optional custom artifact repository location. |
| `MAX_MLFLOW_ARTIFACT_SIZE_MB` | `100.0` | Maximum artifact file size allowed for upload. |
| `MLFLOW_RETRY_ATTEMPTS` | `3` | Bounded retry attempts for sync failures. |
| `MLFLOW_RETRY_BACKOFF_FACTOR`| `1.0` | Exponential backoff base factor (seconds). |

ResearchLens starts cleanly when `MLFLOW_ENABLED=false` or when the MLflow package is not installed.

---

## 4. Fault Tolerance & Offline Mode

When MLflow is enabled but the tracking URI is unreachable or encounters network failure:
1. ResearchLens execution proceeds to completion.
2. Canonical local execution outputs, results, logs, and artifacts are fully saved to `data/workspaces/`.
3. Bounded exponential backoff retry is executed (e.g., 3 attempts).
4. If sync cannot complete, the run status remains `COMPLETED` while `mlflow_sync_status` is marked `FAILED` with `mlflow_sync_error`.
5. An audit event `MLFLOW_SYNC_FAILED` is recorded.
6. The user can retry synchronization at any time via the UI or `MLflowSyncService.sync_run()`.

---

## 5. Security & Sanitization

1. **Parameter Sanitization**: All hyperparameters and run metadata pass through `sanitize_parameters()` before reaching MLflow. Any keys containing `password`, `secret`, `token`, `api_key`, `auth`, `credential`, or `private_key` are automatically masked with `[MASKED]`.
2. **Artifact Path Traversal**: Artifact upload paths are validated to prevent `../` traversal or escaping safe boundaries.
3. **Artifact Size Capping**: Any artifact exceeding `MAX_MLFLOW_ARTIFACT_SIZE_MB` (default 100 MB) is rejected from MLflow upload while preserved locally.
4. **Role-Based Access Control**: Model promotion requires `REVIEWER`, `RESEARCHER`, or `OWNER` workspace privileges. Unauthorized attempts are rejected and audited.

---

## 6. Remote Tracking & Health Validation (Phase 14)

### 6.1 Remote HTTP(S) Tracking
ResearchLens supports remote MLflow tracking servers (e.g. `http://mlflow.internal:5000` or PostgreSQL + S3 artifact backends). Tracking and artifact URIs are validated against canonical schemes (`http`, `https`, `file`, `s3`, `sqlite`).

### 6.2 Health Check & Probing (`validate_mlflow_configuration`)
Returns structured connectivity status without hanging or crashing ResearchLens startup:
- `CONNECTED`: Remote server or local tracking directory accessible.
- `DISABLED`: Tracking disabled via `MLFLOW_ENABLED=false`.
- `UNAVAILABLE`: Unreachable server or missing Python package.
- `MISCONFIGURED`: Malformed URI or unsupported scheme.

### 6.3 Model Registry Docker Provenance
Models registered from containerized runs automatically include `execution_image` and `image_digest` within model metadata and MLflow tags.

