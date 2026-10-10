# ResearchLens UI Component Inventory

## Overview

This document catalogs the component inventory, reusable design elements, view structures, state ownership, and service contracts for the redesigned **ResearchLens** platform.

---

## 1. Design System & CSS Tokens ([app/styles/custom.css](file:///c:/Users/msiva/Music/ResearchLens/app/styles/custom.css))

| Token / Class | Category | Purpose & Description |
|---|---|---|
| `--lens-bg-root` | Color Token | Root application background (`#0B0F19`, Cosmic Obsidian) |
| `--lens-surface-primary` | Color Token | Primary container background (`#111827`) |
| `--lens-surface-card` | Color Token | Card and panel elevated background (`#182234`) |
| `--lens-border-subtle` | Color Token | High-definition card border (`#263554`) |
| `--lens-border-focus` | Color Token | Accessible focus ring indicator (`#3B82F6`) |
| `--lens-text-primary` | Color Token | Primary high-contrast text (`#F8FAFC`) |
| `--lens-text-secondary` | Color Token | Muted body and label text (`#94A3B8`) |
| `--lens-status-success` | Semantic Token | Confirmed evidence, passed tests (`#10B981`) |
| `--lens-status-warning` | Semantic Token | Approval gates, pending review (`#F59E0B`) |
| `--lens-status-danger` | Semantic Token | Refusals, errors, timeouts (`#EF4444`) |
| `.lens-header` | Layout Component | Top application/section banner with title, subtitle, and badges |
| `.lens-card` | Layout Component | Elevated container with subtle borders and hover transitions |
| `.lens-stat-box` | Data Display | High-impact metric card with label, prominent number, and subtext |
| `.lens-citation-panel` | Provenance | Strict evidentiary quotation panel with page, section, confidence, and quote |
| `.lens-badge` | Status Display | Semantic pills (`badge-success`, `badge-warning`, `badge-danger`, `badge-info`) |
| `.tldr-card` | Editorial Callout | High-contrast single-sentence synthesis block |
| `.exec-card` | Editorial Callout | Structured multi-paragraph executive summary block |
| `.approval-gate-banner` | Integrity Gate | Prominent human researcher authorization gate callout |
| `.lens-empty-state` | State Display | Informative empty state card with icon, title, description, and action hint |

---

## 2. Reusable UI Components ([app/components/ui_elements.py](file:///c:/Users/msiva/Music/ResearchLens/app/components/ui_elements.py))

| Component Function | Inputs | Output / Rendering |
|---|---|---|
| `render_page_header` | `title`, `subtitle`, `workspace_context`, `badge_label`, `badge_variant` | Renders the standard scientific banner with active workspace context and platform status pill. |
| `render_stat_card` | `label`, `number`, `subtext` | Renders a styled metric card with clear numeric hierarchy. |
| `render_empty_state` | `title`, `description`, `icon` | Renders an informative empty state card without visual clutter. |
| `render_citation_panel` | `page_number`, `section_name`, `source_text`, `confidence`, `label`, `chunk_id` | Renders a verbatim excerpt panel with exact page, section, and confidence provenance. |
| `render_approval_gate_banner` | `stage_name`, `approval_status`, `approved_by`, `timestamp`, `required_role` | Renders the human-in-the-loop approval gate callout requiring researcher authorization. |

---

## 3. Core View Components ([app/components/](file:///c:/Users/msiva/Music/ResearchLens/app/components/))

| View Component | File Reference | Upstream Services & Backend Models | Primary Capabilities |
|---|---|---|---|
| **Dashboard View** | [dashboard_view.py](file:///c:/Users/msiva/Music/ResearchLens/app/components/dashboard_view.py) | `WorkspaceStore`, `check_system_health()` | High-level research statistics, recent papers, activity feed, diagnostic probes, and quick-action shortcuts. |
| **Upload Component** | [upload.py](file:///c:/Users/msiva/Music/ResearchLens/app/components/upload.py) | `get_settings()` | Validates PDF format, enforces 100 MB upload limit, detects empty uploads. |
| **Summary View** | [summary_view.py](file:///c:/Users/msiva/Music/ResearchLens/app/components/summary_view.py) | `Document`, `summarize_section` | Document metrics, TL;DR, executive summary, and expandable section summaries. |
| **Paper Intelligence** | [metadata_view.py](file:///c:/Users/msiva/Music/ResearchLens/app/components/metadata_view.py) | `ResearchProfile`, `ExtractedItem` | Structured facts: problem, objective, methods, datasets, metrics, limitations with page/section provenance. |
| **Research Gaps** | [gap_view.py](file:///c:/Users/msiva/Music/ResearchLens/app/components/gap_view.py) | `ResearchGapReport`, `CandidateGap` | Identified gaps, explicit vs inferred, confidence scores, supporting evidence accordions. |
| **Evidence Q&A** | [qa_view.py](file:///c:/Users/msiva/Music/ResearchLens/app/components/qa_view.py) | `ScientificRetrievalPipeline`, `LocalQuestionAnswerer` | Hybrid dense-BM25 retrieval, Cross-Encoder reranking, citation cards, explicit refusal on insufficient evidence. |
| **Literature Intelligence** | [literature_view.py](file:///c:/Users/msiva/Music/ResearchLens/app/components/literature_view.py) | `LiteratureCollection`, `generate_literature_report` | Batch PDF ingestion, external literature discovery (arXiv, Semantic Scholar), comparability checks, conflict detection. |
| **Cross-Paper Comparison** | [comparison_view.py](file:///c:/Users/msiva/Music/ResearchLens/app/components/comparison_view.py) | `PaperComparisonMatrix`, `PaperProfile` | Comparative matrix across models, datasets, metrics, and limitations. |
| **Research Monitor** | [monitoring_view.py](file:///c:/Users/msiva/Music/ResearchLens/app/components/monitoring_view.py) | `TopicStore`, `ResearchMonitoringService` | Continuous monitoring of dynamic research topics, paper change detection, literature evolution reports. |
| **Collaborative Workspace** | [workspace_view.py](file:///c:/Users/msiva/Music/ResearchLens/app/components/workspace_view.py) | `WorkspaceStore`, `ExperimentService` | 12 subtabs: Overview, Team, Papers, Evidence Bookmarks, Discussions, Questions, Directions, Plans, Lab, Findings, Risks, Export. |
| **Sandboxed Execution** | [execution_view.py](file:///c:/Users/msiva/Music/ResearchLens/app/components/execution_view.py) | `ExperimentRegistryService`, `ExperimentExecutor` | Local sandboxed execution, environment snapshots, artifact verification, reproducibility scoring. |
| **MLflow Tracking** | [mlops_view.py](file:///c:/Users/msiva/Music/ResearchLens/app/components/mlops_view.py) | `MLflowSyncService`, `ResearchModelRegistry` | Parameter tracking, metric normalization, MLflow run inspection, model stage governance (`CANDIDATE`, `VALIDATED`, `PRODUCTION`). |
| **MLOps Infrastructure** | [mlops_infra_view.py](file:///c:/Users/msiva/Music/ResearchLens/app/components/mlops_infra_view.py) | `DockerExecutionBackend`, `ExecutionQueueManager` | Backends (`LOCAL`, `DOCKER`, `REMOTE`), job queue, image allowlist, matrix generator, reproducibility diff, CI regression guards. |
| **Research Automation** | [automation_view.py](file:///c:/Users/msiva/Music/ResearchLens/app/components/automation_view.py) | `ResearchWorkflowOrchestrator` | 13-stage guided workflow, strict approval gate, evaluation, finding candidate generation, and formal human peer review. |

---

## 4. State Ownership & Persistence Semantics

| State Key | Lifetime | Storage Layer | Purpose |
|---|---|---|---|
| `authenticated` | Session | In-Memory (`st.session_state`) | Guards production application when `api_secret_key` is configured. |
| `current_document` | Session | In-Memory (`Document`) | Holds the active parsed research paper and section mapping. |
| `current_profile` | Session | In-Memory (`ResearchProfile`) | Extracted structured scientific entities and metadata. |
| `current_gap_report` | Session | In-Memory (`ResearchGapReport`) | Extracted candidate limitations and directional research gaps. |
| `current_collection` | Session | In-Memory (`LiteratureCollection`) | Multi-paper literature batch. |
| `current_pipeline` | Session | In-Memory (`ScientificRetrievalPipeline`) | Active hybrid retriever and reranker. |
| `comparison_matrix` | Session | In-Memory (`PaperComparisonMatrix`) | Cross-paper comparison rows. |
| Workspaces & Entities | Persistent | Filesystem (`data/workspaces/<id>/*.json`) | Atomic JSON transactions via `WorkspaceStore`. |
| Topics & Monitoring | Persistent | Filesystem (`data/topics/<id>/*.json`) | Stored topic definitions and crawl history via `TopicStore`. |
| MLflow Runs | Persistent | SQLite / Filesystem (`./mlruns`) | Experiment parameters, empirical metrics, and model artifacts. |
