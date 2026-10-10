# Computational Reproducibility & Provenance

## 1. Principles of Computational Reproducibility

Scientific progress depends on the verifiable reproducibility of empirical observations. ResearchLens guarantees computational provenance by capturing all ten canonical dimensions of research runs:

1. **Exact Code Version**: Git commit SHA (`git rev-parse HEAD`). If Git is not initialized, `NOT_AVAILABLE` is explicitly stored; commit hashes are never fabricated.
2. **Dataset Provenance**: Dataset identifier, dataset version, and SHA-256 hash. If unavailable, `NOT PROVIDED` is marked.
3. **Model Provenance**: Model architecture, checkpoint path or hash, and configuration. Model weights are never duplicated into workspace storage.
4. **Configuration & Hyperparameters**: Versioned configuration mapping (`learning_rate`, `batch_size`, `epochs`, `optimizer`, `dropout`).
5. **Random Seed**: Explicit seed specification. If absent, `SEED NOT PROVIDED` is marked.
6. **Python Runtime**: Exact interpreter version and architecture.
7. **Package Environment**: Complete snapshot of installed pip packages and versions.
8. **Hardware Specification**: Host OS, CPU cores, RAM, and GPU device / CUDA version if present.
9. **Execution Entrypoint & Arguments**: Command executed and working directory.
10. **Output Artifact Hashes**: SHA-256 cryptographic hashes for all generated models, metrics, plots, and tables.

---

## 2. Canonical Reproducibility Manifest (`reproducibility.json`)

Stored under `data/workspaces/<workspace_id>/runs/<run_id>/reports/reproducibility.json`:
```json
{
  "code_commit": "4166520da1f65aebb76efe3484ed4db83d10d244",
  "dataset": "SQuAD_v2",
  "dataset_hash": "NOT PROVIDED",
  "model": "DistilBERT-QA",
  "model_version": "1.0",
  "configuration": {
    "batch_size": 32,
    "epochs": 3
  },
  "seed": 42,
  "python_version": "3.13.13",
  "hardware": {
    "os": "Windows",
    "cpu": "16 cores",
    "ram_total_gb": 31.85,
    "gpu": "CPU Only",
    "cuda_version": "NOT_AVAILABLE"
  },
  "artifacts": [
    {
      "name": "results.json",
      "type": "METRICS",
      "sha256": "...",
      "size_bytes": 102
    }
  ]
}
```

---

## 3. Reproducibility Completeness Score

Rather than assigning an arbitrary or misleading "probability" to reproducibility, ResearchLens computes a **Reproducibility Completeness Score**:
- Format: `8/10 metadata fields available`
- Categorization:
  - `COMPLETE`: 10/10 canonical fields documented.
  - `PARTIAL`: 6 to 9 canonical fields documented.
  - `INSUFFICIENT`: Fewer than 6 canonical fields documented.

---

## 4. MLflow Reproducibility & Model Lineage Integration (Phase 13)

- **Manifest as MLflow Artifact**: `reproducibility.json` is logged as a first-class artifact alongside `environment.json` and requirements snapshots.
- **Double Identity Artifact Verification**: Each artifact retains its authoritative ResearchLens SHA-256 hash paired with its MLflow storage URI.
- **Model Registry Lineage**: Every registered model preserves source run ID, Git commit SHA, dataset identifier, evaluation metrics, configuration dictionary, and artifact SHA-256 hash. Overwriting existing model versions is strictly prohibited.

---

## 5. Reproducibility Re-Run, Diff & Drift Detection (Phase 14)

### 5.1 Non-Destructive Re-run (`reproduce_run`)
Researchers can trigger a rerun using the exact parameters, seed, configuration, dataset, image, and backend of an original run:
- A brand new `ExperimentRun` is instantiated.
- The original run and its artifacts remain immutable.

### 5.2 Differential Comparison (`ReproducibilityDiffService`)
Compares the original and reproduced runs across:
- Metrics
- Parameters
- Dataset Hash
- Model Version
- Code Commit
- Environment / Software Dependencies
- Artifact SHA-256 Hashes

Each dimension evaluates to `MATCH`, `DIFFERENCE`, or `MISSING`.

### 5.3 Provenance Drift Detection (`ReproducibilityDriftDetector`)
Detects underlying drift across:
- `DATASET_CHANGED`: Dataset content or hash has changed since original run.
- `CODE_CHANGED`: Git commit SHA or dirty worktree state has drifted.
- `DEPENDENCIES_CHANGED`: Software dependencies or python runtime modified.
- `MODEL_CHANGED`: Model checkpoint or version modified.
- `CONFIGURATION_CHANGED`: Hyperparameters or runtime arguments modified.


