# Research Workspace & Research Planning Intelligence (Phase 10)

## 1. Overview & Objective

ResearchLens Phase 10 transitions the platform from a literature intelligence viewer into an active **Researcher Workspace and Research Planning Decision-Support System**:

```text
Literature Discovery & Monitoring
              ↓
  Saved Papers & Collections
              ↓
 Evidence Bookmarks & Annotations
              ↓
Research Questions & Hypotheses
              ↓
Prioritized Research Directions
              ↓
  Grounded Research Planning
              ↓
Benchmark Experiment Builder & Risk Register
              ↓
Traceable Multi-Format Export (Markdown / JSON / CSV)
```

The system remains strictly an evidence-grounded **decision-support tool**. It assists researchers in organizing literature, articulating testable hypotheses, drafting structured benchmark protocols, and mapping potential risks, without substituting for independent human scientific reasoning.

---

## 2. Workspace Data Model & Schema

All models reside in [`src/researchlens/workspace/schema.py`](file:///c:/Users/msiva/Music/ResearchLens/src/researchlens/workspace/schema.py):

### 2.1 Research Workspace (`ResearchWorkspace`)
- `workspace_id`: Unique sanitized identifier (`ws_<uuid>`).
- `name`: User-defined project title.
- `description`: Research context and scope.
- `topic_id`: Optional link to a Phase 9 longitudinal research topic.
- `status`: Lifecycle status (`ACTIVE`, `ARCHIVED`).
- `created_at`, `updated_at`: ISO UTC timestamps.

### 2.2 Paper Collections & Saved Papers (`PaperCollection`, `SavedPaper`)
- **`PaperCollection`**: Thematic folders (`collection_id`, `name`, `description`, `paper_ids`).
- **`SavedPaper`**: Papers marked by the researcher with reading priority (`LOW`, `MEDIUM`, `HIGH`), status (`UNREAD`, `READ`, `REVIEWED`), custom tags, and personal notes.

### 2.3 Evidence Bookmarks & Annotations (`EvidenceBookmark`, `ResearchAnnotation`)
- **`EvidenceBookmark`**: Point-in-paper evidentiary citations preserving `paper_id`, `page_number`, `section_name`, `chunk_id`, `source_text`, and label.
- **`ResearchAnnotation`**: Margin notes categorized into `IMPORTANT`, `QUESTION`, `LIMITATION`, `METHOD`, `RESULT`, `IDEA`, and `FOLLOW_UP`.

### 2.4 Research Questions & Hypotheses (`ResearchQuestion`, `ResearchHypothesis`)
- **`ResearchQuestion`**: Formal inquiries categorized into `PRIMARY`, `SECONDARY`, or `EXPLORATORY`, linked to supporting gaps and evidence IDs.
- **`ResearchHypothesis`**: Testable variable relationships (`independent_variables`, `dependent_variables`, `expected_relationship`, `supporting_evidence`).
  - *Scientific Integrity Rule:* Newly created hypotheses strictly default to `DRAFT` or `PROPOSED`. The AI cannot mark a hypothesis as `SUPPORTED`. Only empirical verification by the researcher can transition this status.

### 2.5 Research Directions (`ResearchDirection`)
- Potential investigation angles scored via a transparent priority formula across 6 dimensions:
  - `evidence_strength` (weight: 0.25)
  - `research_gap_strength` (weight: 0.20)
  - `feasibility` (weight: 0.15)
  - `impact` (weight: 0.15)
  - `data_availability` (weight: 0.15)
  - `evaluation_feasibility` (weight: 0.10)

### 2.6 Benchmark Experiment Builder (`ExperimentPlan`)
- Empirical evaluation plans specifying `model`, `baseline`, `dataset`, `metrics`, `variables`, and `expected_outcome` with statuses (`PLANNED`, `RUNNING`, `COMPLETED`, `BLOCKED`). Planning only; does not execute GPU workloads.

### 2.7 Risk Register & Milestones (`ResearchRisk`, `ResearchMilestone`)
- **`ResearchRisk`**: Evidence-grounded risks inferred from reported literature limitations (`GENERALIZATION`, `COMPUTE`, `DATA_LEAKAGE`, `REPRODUCIBILITY`, `BIAS`).
- **`ResearchMilestone`**: Target timeline tracking (`NOT_STARTED`, `IN_PROGRESS`, `COMPLETED`, `BLOCKED`).

---

## 3. Storage Architecture

Implemented in [`src/researchlens/workspace/store.py`](file:///c:/Users/msiva/Music/ResearchLens/src/researchlens/workspace/store.py):
- **Local Persistence:** Data stored under `data/workspaces/<workspace_id>/`.
- **Atomic File Operations:** Uses temporary file creation and atomic replacement (`os.replace`) to guard against disk corruption.
- **Path Sanitization:** Validates workspace and paper IDs against regular expression `^[a-zA-Z0-9_\-]+$` to block directory traversal attacks.
- **Workspace Search:** In-memory keyword filtering across all entity fields in sub-2ms latency.

---

## 4. Evidence-to-Plan Traceability

Implemented in [`ResearchPlanner`](file:///c:/Users/msiva/Music/ResearchLens/src/researchlens/workspace/planner.py):
- Every literature-derived element in a synthesized plan maintains an auditable provenance chain:
  `Plan -> Research Question -> Hypothesis -> Dataset / Baseline -> Evidence Bookmark -> Paper Excerpt`
- Missing provenance links trigger automated validation warnings.

---

## 5. Scientific Integrity Boundaries

ResearchLens strictly enforces:
1. **No Auto-Validation:** AI never marks hypotheses as `SUPPORTED` or gaps as `CLOSED`.
2. **Explicit Labeling:** Distinguishes `LITERATURE EVIDENCE` from `USER INTERPRETATION` from `AI-GENERATED SUGGESTION` from `EXPERIMENTAL RESULT`.
3. **No Hallucinated Claims:** Mandates scientific integrity disclaimer notices on all exported plans.

---

## 6. Phase 11 Extensions: Collaboration & Experiment Intelligence

In Phase 11, the workspace expands to support multi-researcher collaboration and external experiment analysis:
- **`WorkspaceMember`**: Roles (`OWNER`, `RESEARCHER`, `REVIEWER`, `VIEWER`) for granular access control.
- **Shared Evidence**: Single persistent record per citation with author tracking (`created_by`) and privacy toggles (`visibility`).
- **Contextual Discussions**: `ResearchComment` attached to papers, evidence, hypotheses, and runs; plus multi-turn `DiscussionThread` topics.
- **Experiment Runs & Result Ingestion**: Ingests external CSV and JSON logs into typed `ExperimentResult` entities marked `UNVERIFIED_IMPORT`.
- **Baseline Comparison**: Direct comparability restricted to matching `(dataset, metric)` tuples.
- **Reproducibility Checks**: Audits environmental completeness (`COMPLETE`, `PARTIAL`, `INSUFFICIENT`) without inventing missing parameters.
- **Peer Review & Finding Traceability**: `ResearchFinding` (including `NEGATIVE_RESULT`) and `ResearchReview` approval flows. Hypotheses require manual researcher confirmation before transition to `SUPPORTED`.
- **Security & Audit Logs**: Immutable `AuditEvent` log with automated secret filtering.
