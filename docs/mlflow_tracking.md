# MLflow Tracking & NLP/RAG Experiment Logging

## 1. Parameter & Metric Tracking

ResearchLens experiments log configuration hyperparameters and evaluated metrics to MLflow runs.

### Parameter Logging
Parameters are extracted from:
- `ExperimentPlan.configuration` (e.g., `learning_rate`, `batch_size`, `epochs`, `optimizer`, `dropout`, `seed`).
- NLP & RAG-specific parameters (e.g., `embedding_model`, `chunk_size`, `chunk_overlap`, `top_k`, `temperature`).
- Execution seeds and dataset splits.

Sensitive keys are stripped and masked prior to MLflow API calls.

### Metric Validation & Normalization
Metrics are processed through `ExperimentService.normalize_and_validate_metric()`:
- Bounded metrics (`accuracy`, `f1`, `precision`, `recall`, `roc_auc`, `mrr`) are verified within `[0.0, 1.0]`. Percentages (e.g., `85.0%`) are automatically converted to `0.850`.
- Latency and loss metrics (`loss`, `latency_ms`, `perplexity`) are validated as non-negative floats.
- Invalid or corrupted metrics are rejected from logging to maintain telemetry hygiene.

### Step-Based Metrics
MLflow step logging is supported for iterative training and evaluation:
```python
tracker.log_metrics(
    mlflow_run_id="run_123",
    metrics={"loss": 0.42, "accuracy": 0.88},
    step=epoch_number,
)
```

---

## 2. NLP & RAG Experiment Tracking

Because ResearchLens is an NLP and scientific research workbench, specialized logging captures RAG pipeline attributes:

1. **Exact Model Identifiers**:
   - Explicit HuggingFace IDs (e.g., `sentence-transformers/all-MiniLM-L6-v2`, `facebook/bart-large-cnn`) are logged, avoiding generic labels like "BERT".
2. **Retrieval Metrics**:
   - `retrieval_precision`, `retrieval_recall`, `mrr`, `answerability`, `citation_accuracy`, `rouge1`, `rouge2`, `rougeL`, `f1`.
3. **Prompt Versioning**:
   - Prompt template version identifiers are logged.
   - Exact prompt templates or schemas are SHA-256 hashed (`prompt_sha256`) to ensure prompt reproducibility without risking secret leakage.
4. **Dataset Provenance**:
   - `dataset_name`, `dataset_version`, `dataset_hash`, and evaluation slice identifiers.

---

## 3. Artifact Synchronization & Hashing Parity

ResearchLens artifacts are synchronized to MLflow with double-identity verification:

1. **Canonical Identifier**: SHA-256 checksum computed on disk before upload.
2. **Storage Identifier**: MLflow artifact URI (e.g., `runs:/<mlflow_run_id>/predictions.json`).

Supported artifacts include:
- `results.json`: Output contract containing normalized metrics and metadata.
- `reproducibility.json`: Canonical Phase 12 manifest.
- `environment.json` & `requirements.lock`: Environment snapshots.
- Confusion matrices, prediction CSVs, loss curves, and diagnostic plots.

Artifacts exceeding `MAX_MLFLOW_ARTIFACT_SIZE_MB` remain on local disk and are marked with `[OVERSIZED_LOCAL_ONLY]`.
