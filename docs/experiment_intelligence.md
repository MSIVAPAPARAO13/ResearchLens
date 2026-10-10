# Experiment Intelligence in ResearchLens

## 1. Principles
ResearchLens is an **experiment-aware research environment**, not an arbitrary code execution runner. It does not train models or execute GPU tasks directly. Instead, it ingests, validates, compares, and audits empirical outputs generated in external environments.

## 2. Ingestion & Validation Lifecycle
```
External Benchmark / Cluster Run
               ↓
CSV / JSON Evaluation Log
               ↓
ResearchLens Preview & Validation
(Row schema checks, metric ranges, percentage normalization)
               ↓
Confirmation by Researcher
               ↓
Stored ExperimentResult (Marked 'UNVERIFIED_IMPORT')
```

### Supported Formats
1. **Wide CSV**: Columns represent metrics (e.g. `model,dataset,accuracy,f1,latency,seed`).
2. **Narrow JSON / CSV**: Records represent key-value metric assignments (e.g. `{"model": "BERT", "dataset": "SQuAD", "metric": "f1", "value": 0.915}`).

### Metric Normalization Rules
- Bounded metrics ($[0, 1]$ or $[0, 100\%]$): `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `mrr`, `map`, `ndcg`. Percentages $> 1.0$ and $\le 100.0$ are automatically normalized to fractions (e.g. $91.5\% \rightarrow 0.915$).
- Non-negative metrics ($\ge 0$): `rmse`, `mae`, `bleu`, `rouge`, `loss`, `latency`.

## 3. Scientific Comparability & Baseline Comparison
In accordance with Phase 6 scientific comparability:
- Absolute and relative deltas ($\Delta = \text{val}_{\text{proposed}} - \text{val}_{\text{baseline}}$) are computed **only** when `dataset` and `metric` match identically.
- Cross-dataset or cross-metric comparisons are flagged as `NO_BASELINE_MATCH` or `NOT_DIRECTLY_COMPARABLE`.

## 4. Reproducibility Assessment
Each execution run is audited against 4 critical reproducibility factors:
1. Execution environment (OS, Python version).
2. Hardware specification (GPU/CPU model, memory).
3. Software dependencies (libraries with exact versions).
4. Random seed.

Completeness tiers:
- **`COMPLETE`**: All 4 metadata elements specified.
- **`PARTIAL`**: 2–3 metadata elements specified.
- **`INSUFFICIENT`**: 0–1 metadata elements specified. Missing fields are explicitly reported as `"NOT PROVIDED"`, never fabricated.

## 5. Scientific Integrity Safeguards
- **No Automatic Hypothesis Promotion**: AI suggestions or high metric values never automatically switch a hypothesis to `SUPPORTED`. A human researcher must confirm.
- **Negative Results Support**: Underperforming models and failed hypotheses are first-class findings (`NEGATIVE_RESULT`).
- **Explicit Provenance Chain**: Every empirical finding traces back to an experiment run, experiment plan, hypothesis, question, and literature evidence quote.

## 6. Phase 12: Controlled Execution & Reproducibility Pipeline
In addition to external result ingestion, Phase 12 adds an explicit, sandboxed local execution layer:
- **Experiment Whitelist**: Execution is restricted to scripts registered in `ExperimentRegistry`.
- **Sandboxed Subprocesses**: Runs use typed argument arrays and `shell=False`.
- **Output Contract**: Every run must produce `results.json`, which is automatically ingested via `ExperimentService`.
- **SHA-256 Hashing**: All outputs in `artifacts/`, `metrics/`, and `reports/` are cryptographically hashed.
- **Multi-Run & Re-run Tracking**: Re-runs produce new distinct run records; aggregate statistics (`mean`, `std`, `min`, `max`) are calculated across seeds and runs without inferring unproven statistical significance.
