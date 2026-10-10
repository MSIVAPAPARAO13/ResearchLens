# Research Monitoring & Literature Evolution Intelligence (Phase 9)

## 1. Overview & Objective

ResearchLens Phase 9 transitions the platform from on-demand search ("discover and analyze literature when the user asks") to longitudinal intelligence: **"Track a research topic over time and identify what changed in the literature."**

The monitoring system functions strictly as an evidence-grounded **decision-support tool**, keeping users systematically informed of emerging paradigms, method novelties, dataset shifts, and the temporal lifecycle of research gaps without substituting for human scientific judgment.

---

## 2. Architecture & Data Model

### 2.1 Saved Research Topics (`ResearchTopic`)
A topic defines a continuous interest boundary:
- `topic_id`: Unique deterministic or generated UUID string.
- `name`: User-facing display title (e.g., "AI-based Alzheimer MRI diagnosis").
- `query`: Exact scholarly query string passed to external search engines.
- `description`: Explanatory context for the topic.
- `providers`: List of targeted literature sources (`semantic_scholar`, `arxiv`, `crossref`, `pubmed`).
- `year_from` / `year_to`: Optional temporal filters.
- `max_results`: Per-provider retrieval limit.
- `created_at` / `last_checked_at`: ISO timestamp tracking.
- `active`: Boolean toggle allowing disabling without data deletion.

### 2.2 Search Configuration (`MonitoringConfig`)
Encapsulates execution parameters:
- `search_query`: Query string.
- `providers`: Target provider list.
- `max_results`: Result threshold.
- `date_filter`: Bounded publication year ranges.
- `open_access_only`: Filter for open-access licensing.
- `similarity_threshold`: Deduplication threshold (default: 0.85).
- `ranking_config`: Config dictionary reused directly by Phase 7 hybrid ranking.

### 2.3 Monitoring Runs (`MonitoringRun`)
Captures point-in-time snapshot of discovery:
- `run_id`: Unique run identifier.
- `topic_id`: Parent topic reference.
- `started_at` / `completed_at`: Execution timestamps.
- `status`: Lifecycle state (`RUNNING`, `COMPLETED`, `PARTIAL`, `FAILED`).
- `query` / `providers`: Execution metadata.
- `results_count`: Total retrieved raw papers.
- `new_papers_count`: Newly discovered papers.
- `updated_papers_count`: Papers with detected metadata changes.
- `duplicate_count`: Deduplicated results.
- `errors`: List of provider failure strings (graceful degradation ensures partial runs complete if at least one provider succeeds).

### 2.4 Paper History (`PaperHistoryEntry`)
Persistent tracking structure ensuring longitudinal identity across monitoring runs:
- `paper_id`: Stable identifier (prefers DOI > arXiv ID > provider ID > normalized title).
- `first_seen` / `last_seen`: Timestamp bounds.
- `providers`: Observed source providers.
- `title`, `authors`, `year`, `doi`, `arxiv_id`, `citation_count`, `url`.
- `previous_metadata`: Prior run snapshot dictionary.
- `current_metadata`: Current run snapshot dictionary.

---

## 3. Change Detection Engine

The change detection pipeline partitions incoming deduplicated search results against historical records:
1. **NEW**: No prior identifier or normalized title match exists in `PaperHistoryEntry`.
2. **EXISTING**: Unchanged metadata and persistent citation signal.
3. **UPDATED**: Changes detected in citation count, abstract, venue, authors, or open-access PDF availability. Recorded as `(field, old_value, new_value, detected_at)`.
4. **REMOVED_FROM_RESULTS**: Previously observed papers absent from the latest top search window.

---

## 4. Longitudinal Gap Evolution

Implemented in `src/researchlens/analysis/gap_evolution.py`.

### 4.1 Gap Lifecycle Statuses
- `NEW`: Newly observed research gap limitation in the current run.
- `PERSISTENT`: Unchanged gap appearing across consecutive runs without significant support delta.
- `STRENGTHENING`: Limitation acknowledged by a growing number of newly analyzed papers.
- `WEAKENING`: Fewer recent papers emphasize the gap, or preliminary solutions emerge without complete verification.
- `POTENTIALLY_ADDRESSED`: At least one newly discovered paper explicitly introduces evidence or benchmarks addressing the limitation.
- `UNVERIFIED`: Gap identified from abstract/metadata without full-text verification.

*Rule:* ResearchLens never marks a gap as definitively "CLOSED" without conclusive, verified empirical evidence.

---

## 5. Emerging Trends & Theme Evolution

Implemented in `src/researchlens/analysis/trend_analysis.py`.

1. **Emerging Methods**: Compares `ResearchProfile.models` across runs. Identifies newly observed architectures, algorithms, and paradigms.
2. **Emerging Datasets**: Compares `ResearchProfile.datasets` to identify newly adopted benchmarks and modalities.
3. **Emerging Metrics**: Compares `ResearchProfile.metrics` to track novel evaluation criteria.
4. **Theme Evolution**: Reuses embedding cosine similarities across semantic clusters to classify research themes into `NEW_THEME`, `PERSISTENT_THEME`, `EMERGING_THEME`, or `DECLINING_THEME`.

---

## 6. Alerts & Notification System

Implemented in `src/researchlens/analysis/monitoring_service.py`.

- **Alert Types**: `NEW_PAPER`, `NEW_METHOD`, `NEW_DATASET`, `NEW_RESEARCH_GAP`, `GAP_STATUS_CHANGED`, `POTENTIALLY_ADDRESSED_GAP`, `SIGNIFICANT_LITERATURE_CHANGE`.
- **Severity Levels**:
  - `INFO`: Ordinary relevant new paper discoveries.
  - `IMPORTANT`: Multiple new papers or emergence of novel methods/datasets.
  - `HIGH`: Potentially addressed major research gap.
- **Notification Provider Foundation**: `NotificationProvider` abstract interface with `ConsoleNotificationProvider` default. Extensible for Slack, Email, or Webhooks in future phases.

---

## 7. Reporting & Export

Implemented in `src/researchlens/analysis/monitoring_report.py`.
Exports longitudinal monitoring digests into:
- **Markdown (`.md`)**: Human-readable literature review update.
- **JSON (`.json`)**: Machine-readable full state dump for automation.
- **CSV (`.csv`)**: Tabular list of new papers and their provenance for bibliometric tools.

---

## 8. Scientific Integrity Guarantees

All generated summaries and alerts adhere strictly to the Scientific Integrity Rule:
- Phrased strictly as: *"Among the literature retrieved by ResearchLens..."* or *"Within the monitored corpus..."*
- Explicitly distinguishes **literature observations** from **scientific conclusions**.
- Prohibits unfounded universal claims like *"The field has solved this problem"* or *"State of the art"*.
