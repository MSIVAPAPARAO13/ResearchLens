# ResearchLens Architecture Documentation

## 1. System Overview

**ResearchLens** is a modular, AI-powered research paper summarization and intelligence platform. It transforms dense, unstructured academic PDFs into structured, explainable, and hierarchical insights while maintaining strict factual boundaries.

Unlike opaque end-to-end prompting, ResearchLens implements a transparent, multi-stage pipeline:
1. **Ingestion:** Robust extraction preserving document structure, metadata, and page numbers.
2. **Preprocessing:** Noise reduction, explainable regex-based section detection, and sentence-boundary preserving token chunking.
3. **Summarization:** Cached BART-large transformer models operating in a two-tier hierarchical architecture (chunk-level condensation followed by section-level synthesis).
4. **Information Extraction:** Rule- and entity-based identification of authors, methods, datasets, metrics, and quantitative findings.
5. **Research Gap Analysis:** Heuristic extraction of explicit limitations and potential research directions.
6. **Evaluation Framework:** Standard ROUGE (1, 2, L) metric computation and retrieval benchmark contracts.

---

## 2. Target Architecture Diagram

```mermaid
graph TD
    subgraph UI Layer ["User Interface (Streamlit)"]
        UI[app/streamlit_app.py]
        CompUpload[app/components/upload.py]
        CompSumm[app/components/summary_view.py]
        CompMeta[app/components/metadata_view.py]
        CompComp[app/components/comparison_view.py]
    end

    subgraph Service Layer ["Application / Orchestration Layer"]
        Orch[Summarization & Intelligence Service]
    end

    subgraph Core NLP ["src/researchlens/"]
        Config[config/settings.py]
        
        subgraph Ingestion ["Ingestion Layer"]
            PDFLoader[ingestion/pdf_loader.py]
            DocModel[ingestion/document.py]
        end

        subgraph Preprocessing ["Preprocessing Layer"]
            Cleaner[preprocessing/cleaner.py]
            SecParser[preprocessing/section_parser.py]
            Chunker[preprocessing/chunker.py]
        end

        subgraph Summarization ["Summarization Layer"]
            ModelMgr[summarization/model.py]
            SummEngine[summarization/summarizer.py]
            HierSumm[summarization/hierarchical.py]
        end

        subgraph Extraction ["Extraction Layer"]
            Keywords[extraction/keywords.py]
            Entities[extraction/entities.py]
            Methods[extraction/methods.py]
            Datasets[extraction/datasets.py]
            Metrics[extraction/metrics.py]
        end

        subgraph Analysis ["Analysis Layer"]
            Gaps[analysis/research_gaps.py]
            PaperComp[analysis/paper_comparison.py]
            Evidence[analysis/evidence.py]
        end

        subgraph RetrievalQA ["Retrieval & QA (Prepared)"]
            Embeddings[retrieval/embeddings.py]
            VectorStore[retrieval/vector_store.py]
            Retriever[retrieval/retriever.py]
            QA[qa/question_answering.py]
            Prompts[qa/prompts.py]
        end

        subgraph Evaluation ["Evaluation Layer"]
            RougeEval[evaluation/summarization_metrics.py]
            RetrEval[evaluation/retrieval_metrics.py]
            EvalRun[evaluation/evaluation_runner.py]
        end
    end

    UI --> CompUpload
    UI --> CompSumm
    UI --> CompMeta
    UI --> CompComp
    UI --> Orch

    Orch --> PDFLoader
    PDFLoader --> DocModel
    DocModel --> Cleaner
    Cleaner --> SecParser
    SecParser --> Chunker
    Chunker --> HierSumm
    HierSumm --> ModelMgr
    ModelMgr --> SummEngine

    Orch --> Keywords
    Orch --> Entities
    Orch --> Methods
    Orch --> Datasets
    Orch --> Metrics
    Orch --> Gaps
```

---

## 3. Module Responsibilities

### `src/researchlens/config/`
- **`settings.py`**: Central source of truth for runtime configurations, paths, tokenizer limits, chunk sizes, beam parameters, and device resolution (CPU / CUDA). Supports environment variable overrides without hardcoding secrets.

### `src/researchlens/ingestion/`
- **`document.py`**: Strongly typed data representations (`Document`, `Page`, `Section`, `Chunk`, `DocumentMetadata`).
- **`pdf_loader.py`**: Robust PyMuPDF wrapper supporting file paths, binary streams, and temporary file handles. Guarantees file resource closure (`doc.close()`) and page text extraction with page numbers.

### `src/researchlens/preprocessing/`
- **`cleaner.py`**: Academic noise reduction. Cleans citation artifacts, p-values, table/figure references while explicitly protecting valid scientific acronyms (e.g. BERT, LSTM, GPT, CNN).
- **`section_parser.py`**: Explainable, pattern-based academic section classifier mapping noisy text lines to canonical sections (Abstract, Introduction, Literature Review, Methodology, Results, Discussion, Ethics, Conclusion, References).
- **`chunker.py`**: Sentence-aware, token-bounded chunker with configurable overlap and chunk metadata. Eliminates repeated quadratic tokenization.

### `src/researchlens/summarization/`
- **`model.py`**: Thread-safe lazy loading and memory caching for `transformers.BartForConditionalGeneration` and `BartTokenizer`. Implements CPU/CUDA detection.
- **`summarizer.py`**: High-level summarization interface (`summarize_text`, `summarize_section`, `summarize_document`, `generate_tldr`, `generate_executive_summary`).
- **`hierarchical.py`**: Multi-chunk hierarchical summarizer that condenses chunk-level summaries into coherent section summaries.

### `src/researchlens/extraction/`
- **`schema.py`**: Strong typed dataclasses (`ExtractedItem`, `QuantitativeResultItem`, `BaselineComparison`, `ResearchProfile`) preserving verbatim evidence, 1-indexed page number, canonical section name, chunk ID, heuristic confidence, and extraction method.
- **`metadata.py`**: Extraction of paper title, authors, organizations/affiliations, publication year, and author keywords.
- **`problem_objective.py`**: Extraction of research problem, research objective, explicit research questions (`RQ1`), and hypotheses (`H1`).
- **`methods.py`**: Method, model, and algorithm extraction combining POS tagging, NER, noun phrase analysis, and known scientific model patterns.
- **`datasets.py`**: Benchmark dataset extraction with sample size, data modality, split ratios, and source provenance.
- **`experimental_setup.py`**: Hyperparameter, data split, training setup, and hardware configuration extraction.
- **`metrics.py`**: Extraction of metric names, quantitative results (`Metric = Value`), and baseline comparative items with difference calculation.
- **`contributions.py`**: Identification of explicit paper contributions.
- **`gaps_limitations.py`**: Extraction of limitations, weaknesses, and explicit future work directions.
- **`validator.py`**: Entity deduplication, numerical consistency validation, and profile completeness validation.
- **`pipeline.py`**: `ExtractionPipeline` orchestrator coordinating all scientific extractors into a single `ResearchProfile`.

### `src/researchlens/analysis/`
- **`literature_collection.py`**: `LiteratureDocument` and `LiteratureCollection` dataclasses managing multi-paper collections with deterministic SHA-256 document hashing.
- **`paper_matrix.py`**: `PaperComparisonMatrix` generator building method, dataset, metric, baseline, and result comparison tables.
- **`comparability.py`**: Scientific comparability logic enforcing dataset/task alignment before comparing numerical results (`"Comparable"` vs. `"Not directly comparable"`).
- **`conflict_detector.py`**: `CrossPaperConflict` detection identifying empirical result contradictions (`RESULT_CONFLICT`) and evaluation differences (`EVALUATION_DIFFERENCE`).
- **`semantic_clustering.py`**: `LimitationCluster` grouping limitations and future work across papers into recurring themes using `SentenceTransformer` cosine similarity.
- **`cross_paper_gap_engine.py`**: `CrossPaperResearchGapEngine` formulating literature-level research gaps with cross-paper recurrence scoring and author independence checks.
- **`external_provider.py`**: Abstract `ExternalLiteratureProvider` interface contract preparing for future external discovery (Phase 7 readiness).
- **`literature_report.py`**: `LiteratureIntelligenceReport` aggregator supporting Markdown, JSON, and CSV matrix exports.
- **`research_gap_schema.py`**: Strongly typed `ResearchGap` and `ResearchHypothesis` schemas with provenance, confidence scores, and multi-paper fields.
- **`gap_engine.py`**: `ResearchGapEngine` analyzing single-paper `ResearchProfile` inputs.
- **`gap_scorer.py`**: Transparent multi-attribute heuristic scoring and confidence calibration.
- **`research_gap_report.py`**: Single-paper `ResearchGapReport` aggregator.
- **`paper_comparison.py`**: Multi-paper comparison schema and comparison matrix aggregator.
- **`evidence.py`**: Evidence provenance tracking linking summaries and claims back to page/chunk sources.

### `src/researchlens/retrieval/` & `qa/`
- **`embeddings.py`**: Local `SentenceTransformerEmbeddings` utilizing `sentence-transformers/all-MiniLM-L6-v2` with lazy loading, device selection, and L2 normalization.
- **`vector_store.py`**: Local `FAISSVectorStore` executing normalized inner product (exact cosine similarity) with metadata payload indexing.
- **`bm25.py`**: Local lexical search via `BM25Okapi` with scientific tokenization preserving model names (e.g. `resnet-50`), abbreviations, and numbers.
- **`hybrid_retriever.py`**: `HybridRetriever` and `ScientificRetrievalPipeline` combining dense semantic embeddings with BM25 keyword matching via min-max score normalization and configurable convex combination (`0.6` dense / `0.4` BM25).
- **`reranker.py`**: `CrossEncoderReranker` using `cross-encoder/ms-marco-MiniLM-L6-v2` to score candidate `(query, passage)` pairs via joint cross-attention.
- **`retriever.py`**: `DenseRetriever` and `SearchResult` data contract preserving complete provenance (`chunk_id`, `document_id`, `filename`, `page_number`, `section_name`, `retrieval_score`, `reranker_score`).
- **`question_answering.py`**: `LocalQuestionAnswerer` producing strict evidence-grounded answers. Rejects out-of-context queries with `"I could not find enough evidence in the uploaded paper to answer this question."`
- **`prompts.py`**: Grounded scientific prompt contracts.

---

## 4. Phase 3: Two-Stage Hybrid Retrieval & Reranking Architecture

```mermaid
flowchart TD
    A["Raw Research Paper (PDF)"] --> B["PyMuPDF Streaming Ingestion"]
    B --> C["Pages (Page Numbers & Text)"]
    C --> D["Section Parser (Canonical Headers)"]
    D --> E["Token-Aware Chunker (O(N) Complexity)"]
    E --> F["Chunks (chunk_id, page_num, section, text)"]
    
    F --> G["FAISS Vector Store (IndexFlatIP)"]
    F --> H["BM25 Lexical Index (BM25Okapi)"]
    
    Q["User Question"] --> DR["Dense Retriever (Top 10 candidates)"]
    Q --> BR["BM25 Retriever (Top 10 candidates)"]
    G --> DR
    H --> BR
    
    DR --> NORM["Score Normalization (Min-Max)"]
    BR --> NORM
    NORM --> FUSION["Linear Score Fusion (0.6 Dense + 0.4 BM25)"]
    FUSION --> POOL["Hybrid Candidate Pool (Top 10 Chunks)"]
    
    POOL --> CE["Cross-Encoder Reranker (ms-marco-MiniLM-L6-v2)"]
    Q --> CE
    CE --> RERANK["Reranked Top Evidence (Top 5 Chunks)"]
    
    RERANK --> QA["Local Question Answerer (Anti-Hallucination)"]
    Q --> QA
    QA --> ANS["Grounded Answer + Source Citations"]
```

---

## 5. Separation of Concerns & Clean Code Guarantees

1. **No NLP in Streamlit:** The Streamlit UI does not perform tokenization, vector search, or direct inference. It delegates to `researchlens` services.
2. **Deterministic Processing:** Preprocessing and indexing algorithms are explainable and reproducible without stochastic hallucinations.
3. **Lazy Model Initialization:** Importing any module does not trigger gigabyte model downloads or GPU allocations.
4. **Fast Testing:** The test suite mocks heavy models, allowing sub-second test execution on any platform.
5. **No Blind Hallucinations:** Questions without evidential support in the indexed paper return strict refusal statements rather than fabricated assertions.

---

## 6. Phase 6: Multi-Paper Literature Intelligence Architecture

```mermaid
flowchart TD
    A["Batch PDFs (2 to 10+ Papers)"] --> B["In-Memory Streaming Ingestion"]
    B --> C["Section Parsing & ResearchProfile Extraction"]
    C --> D["LiteratureCollection (Deterministic document_ids)"]
    
    D --> M["Paper Matrix Generator (CrossPaperComparisonMatrix)"]
    D --> SC["Semantic Lim/FW Clustering (SentenceTransformer Embeddings)"]
    D --> CD["Conflict Detector (RESULT_CONFLICT, EVALUATION_DIFF)"]
    D --> CP["Scientific Comparability Engine"]
    
    SC --> GE["CrossPaperResearchGapEngine"]
    CD --> GE
    CP --> GE
    
    GE --> REP["LiteratureIntelligenceReport"]
    REP --> EXP1["Markdown Export"]
    REP --> EXP2["JSON Export"]
    REP --> EXP3["Matrix CSV Export"]
```

- **`LiteratureCollection`**: In-memory aggregation of `LiteratureDocument` objects containing extracted `ResearchProfile` metadata.
- **`semantic_clustering.py`**: Embeds limitation and future-work statements with `SentenceTransformer`, groups them by cosine similarity, and assigns categorical theme labels (`EFFICIENCY`, `GENERALIZATION`, `DATASET`, `METHODOLOGY`, `BASELINE`, `ROBUSTNESS`).
- **`comparability.py`**: Ensures numerical results from distinct datasets or metrics are never treated as comparable, preventing unscientific ranking.
- **`conflict_detector.py`**: Flags empirical conflicts when identical models and benchmarks produce diverging metrics.
- **`cross_paper_gap_engine.py`**: Identifies literature-wide gaps requiring multi-paper evidence recurrence with transparent heuristic confidence scoring.

---

## 7. Phase 9: Research Monitoring & Literature Evolution Intelligence

```mermaid
flowchart TD
    T["Saved ResearchTopic & MonitoringConfig"] --> E["UnifiedLiteratureSearchService (S2, arXiv, Crossref, PubMed)"]
    E --> D["Deduplication Engine (DOI / Title Cosine >= 0.85)"]
    D --> CD["detect_paper_changes(Current vs PaperHistoryEntry)"]
    
    CD --> NP["New Papers (NEW)"]
    CD --> UP["Updated Papers (UPDATED)"]
    CD --> EP["Existing / Removed"]
    
    NP --> RP["ResearchProfile & Extraction Pipeline"]
    RP --> TA["TrendAnalyzer (Methods, Datasets, Metrics, Themes)"]
    RP --> GE["GapEvolutionEngine (compare_gap_sets & classify_gap_evolution)"]
    
    TA --> AS["ResearchAlert Generator"]
    GE --> AS
    
    AS --> MR["MonitoringReport (Markdown / JSON / CSV)"]
    AS --> UI["Streamlit Monitoring Dashboard Tab"]
```

- **`ResearchTopic` & `TopicStore`**: Local JSON persistence in `data/topics/` without external database bloat.
- **`PaperHistoryEntry`**: Tracks paper identity across runs using DOI, arXiv ID, provider ID, and normalized title.
- **`GapEvolutionEngine`**: Manages the longitudinal lifecycle of research gaps (`NEW`, `PERSISTENT`, `STRENGTHENING`, `WEAKENING`, `POTENTIALLY_ADDRESSED`, `UNVERIFIED`).
- **`TrendAnalyzer`**: Extracts and classifies emerging algorithms, modalities, and metrics from structured `ResearchProfile` objects.
- **`ResearchMonitoringService`**: Coordinates resilient multi-provider searches, error-tolerant partial executions, alert generation, and reporting.

---

## 8. Phase 10: Research Workspace & Research Planning Intelligence

```mermaid
flowchart TD
    W["ResearchWorkspace (data/workspaces/<ws_id>/)"] --> PC["PaperCollection & SavedPaper (Priority/Tags/Status)"]
    W --> EB["EvidenceBookmark (Page, Section, Chunk, Source Text)"]
    W --> RA["ResearchAnnotation (IMPORTANT, QUESTION, LIMITATION, etc.)"]
    
    EB --> RQ["ResearchQuestion (PRIMARY, SECONDARY, EXPLORATORY)"]
    RA --> RQ
    
    RQ --> RH["ResearchHypothesis (Variables, Expected Relationship)"]
    
    RQ --> RD["ResearchDirection (6-Dimension Priority Score)"]
    RH --> RD
    
    RD --> RP["ResearchPlanner (generate_draft_plan)"]
    RP --> VAL["Plan Validator (Completeness & Missing Component Checks)"]
    RP --> TRACE["Literature-to-Plan Traceability Engine (100% Provenance)"]
    RP --> EXP["Multi-Format Export (Markdown / JSON / CSV)"]
    
    RP --> EXP_PL["ExperimentPlan (Benchmark Protocol Setup)"]
    RP --> RISK["ResearchRisk Register (Inferred from Literature Limitations)"]
    RP --> MILE["ResearchMilestones (Timeline Tracking)"]
```

- **`WorkspaceStore`**: Atomic file-backed JSON persistence in `data/workspaces/<workspace_id>/` with path sanitization preventing traversal attacks.
- **`ResearchPlanner`**: Synthesizes evidence-grounded draft plans, calculates heuristic direction priority scores, infers literature risks, and exports structured plans.
- **`Scientific Integrity Boundary`**: Strictly prevents automated hypothesis confirmation and enforces provenance on all literature-derived suggestions.

---

## 9. Phase 11: Research Collaboration & Experiment Intelligence

```mermaid
flowchart TD
    WS["Research Workspace"] --> TM["Workspace Members (OWNER, RESEARCHER, REVIEWER, VIEWER)"]
    WS --> SP["Shared Papers & Shared Evidence (EvidenceBookmark)"]
    WS --> COMM["Contextual Comments & Threaded Discussions"]
    
    SP --> RQ["Research Questions & Hypotheses"]
    RQ --> EXP_PL["Experiment Plans (v1, v2, v3 Non-Destructive Versioning)"]
    EXP_PL --> EXT["External Experiment Execution (Python / Cluster / Cloud)"]
    
    EXT --> RES_IMP["Experiment Result Ingestion (CSV / JSON)"]
    RES_IMP --> PREV["Preview & Range Validation (Bounds & Percentage Normalization)"]
    PREV --> RES["ExperimentResult (status='UNVERIFIED_IMPORT')"]
    
    RES --> COMP["Scientific Baseline Comparison (Dataset & Metric Alignment)"]
    RES --> REP["Reproducibility Assessment (COMPLETE, PARTIAL, INSUFFICIENT)"]
    
    COMP --> FIND["Research Findings (OBSERVATION, COMPARISON, NEGATIVE_RESULT)"]
    FIND --> REV["Peer Review Workflow (APPROVED, REQUIRES_REVISION, REJECTED)"]
    REV --> HYP_UP["Researcher Hypothesis Update (Human-in-the-Loop Confirmation)"]
    
    FIND --> REP_OUT["Experiment & Collaborative Research Reports"]
    WS --> AUDIT["Security Audit Log & Activity Stream"]
```

- **`WorkspaceMember`**: Role hierarchy (`OWNER > RESEARCHER > REVIEWER > VIEWER`) governing granular permissions.
- **`ExperimentService`**: Ingestion, validation, baseline delta calculation, reproducibility auditing, and finding generation.
- **`Scientific Comparability`**: Results are only compared across matching datasets and metrics, rejecting cross-dataset conflation.
- **`Strict Human Confirmation`**: Hypothesis status cannot be altered to `SUPPORTED` without human researcher sign-off.
- **`Audit & Activity Logging`**: Structured JSON logging with automated credential sanitization.

---

## 11. Phase 12: Controlled Experiment Execution & MLOps Architecture

```mermaid
graph TD
    REG[ExperimentRegistry] -->|Validate Entrypoint| SEC[SecurityValidator]
    SEC -->|Allowed Entrypoint| PRE[Pre-flight Validation & GPU Detection]
    PRE -->|Snapshot Environment| EXE[ExperimentExecutor]
    EXE -->|subprocess shell=False| PROC[Controlled Process]
    PROC -->|timeout & exit code| LOGS[stdout.log & stderr.log]
    PROC -->|results.json contract| OUT[Output Contract Validator]
    OUT -->|Discover Artifacts| ART[Artifact Manager & SHA-256 Hasher]
    ART --> MANIFEST[reproducibility.json Manifest & Completeness Score]
    OUT --> INGEST[ExperimentService Result Ingestion]
    INGEST --> COMP[Scientific Baseline Comparison]
    COMP --> FIND[Candidate ResearchFinding: PENDING_RESEARCHER_REVIEW]
```

- **`ExperimentRegistry`**: Whitelist of permitted local experiment scripts residing in approved directories (`scripts/experiments/`).
- **`SecurityValidator`**: Validates argument arrays against shell metacharacters and enforces strict directory traversal containment.
- **`ExperimentExecutor`**: Subprocess runner executing with `shell=False`, capturing duration, logs, exit codes, and enforcing timeout termination.
- **`Artifact Manager`**: Discovers output files and hashes them with SHA-256 for bitwise tamper detection.
- **`Canonical Reproducibility Manifest`**: Assembles 10 canonical metadata dimensions into `reproducibility.json` and evaluates completeness score.

---

## 12. Phase 13: MLflow Experiment Tracking & Model Registry Integration

```mermaid
graph TD
    RL_WS[ResearchLens Workspace] -->|Deterministic Mapping| ML_EXP[MLflow Experiment]
    RL_RUN[ResearchLens ExperimentRun] -->|Bidirectional Sync| ML_RUN[MLflow Run]
    RL_PARAMS[ExperimentPlan Configuration] -->|Sanitize Credentials| ML_PARAMS[MLflow Parameters]
    RL_METRICS[Normalized Results] -->|Range Validation| ML_METRICS[MLflow Metrics]
    RL_ART[Artifacts with SHA-256] -->|Size Capping <= 100MB| ML_ART[MLflow Artifacts]
    
    RL_RUN --> MANIFEST[reproducibility.json]
    MANIFEST --> ML_ART
    
    RL_RUN --> REG_PRE[Eligibility Validation: Run COMPLETED + SHA-256 Hash]
    REG_PRE --> REG_CAND[ResearchModelRegistration: CANDIDATE]
    REG_CAND -->|Peer Reviewer Human Approval| REG_VAL[Stage: VALIDATED]
    REG_VAL -->|Lead Researcher Human Approval| REG_PROD[Stage: PRODUCTION]
```

- **`MLflowTracker`**: Orchestrates experiment namespaces (`ResearchLens::<Workspace>`), run creation, and parameter/metric logging.
- **`MLflowSyncService`**: Coordinates bidirectional references, bounded retries (exponential backoff), and graceful fault tolerance when MLflow is offline.
- **`ResearchModelRegistry`**: Governs scientific model lifecycle (`NONE`, `CANDIDATE`, `VALIDATED`, `PRODUCTION`, `ARCHIVED`) strictly requiring human review for stage promotions.
- **`Fault Tolerance Boundary`**: When MLflow is offline or disabled (`MLFLOW_ENABLED=false`), local canonical execution completes normally and sync status is marked `FAILED` with retry support.

---

## 13. Phase 14: Remote MLOps & Automated Experiment Infrastructure

```mermaid
graph TD
    RL[ResearchLens Experiment Request] --> REG[Experiment Registry]
    REG --> JOB[ExecutionJob Specification]
    JOB --> QUEUE[ExecutionQueue: Priority & Bounded Workers]
    QUEUE --> BACKEND{Execution Backend}
    
    BACKEND -->|LOCAL| BE_LOC[LocalExecutionBackend: Isolated Subprocess]
    BACKEND -->|DOCKER| BE_DOC[DockerExecutionBackend: Non-root Container + CapDrop]
    BACKEND -->|REMOTE| BE_REM[RemoteExecutionBackend: Cloud/Cluster Contract]
    
    BE_LOC --> RUN[ExperimentRun]
    BE_DOC --> RUN
    BE_REM --> RUN
    
    RUN --> MLFLOW[Remote/Local MLflow Tracking]
    RUN --> ART_STORE[ArtifactStore: LOCAL / S3_COMPATIBLE]
    RUN --> DIFF[Reproducibility Diff & Drift Detection]
    RUN --> REGRESS[Automated Regression Detection]
    RUN --> FINDING[Candidate Research Finding: Human Review Required]
```

- **`Execution Backend Abstraction`**: Decouples execution mechanisms (`LOCAL`, `DOCKER`, `REMOTE`) via `BaseExecutionBackend` while preserving scientific logic.
- **`Docker Sandboxing & Image Registry`**: Non-root execution (`1000:1000`), `--cap-drop ALL`, `no-new-privileges`, CPU/memory limits, disabled network by default, and strict image allowlisting via `ExecutionImageRegistry`.
- **`Remote Execution Contract`**: `RemoteExecutionBackend` contract returning `NOT_CONFIGURED` without cloud lock-in.
- **`Execution Queue & Concurrency`**: Priority-ordered single-process queue supporting bounded worker pools (`max_concurrent_runs` 1, 2, 4).
- **`Artifact Storage Abstraction`**: Unified `BaseArtifactStore` supporting `LOCAL` and optional `S3_COMPATIBLE` (AWS S3, MinIO, Ceph).
- **`Reproducibility Re-Run & Drift Detection`**: Non-destructive re-run generating comparative diffs (`MATCH`, `DIFFERENCE`, `MISSING`) and detecting drift across data, code, model, and configuration.
- **`Controlled Experiment Matrices`**: Bounded Cartesian grids (`MAX_RUNS = 20`) requiring explicit researcher confirmation.
- **`Continuous Integration & Regression Guards`**: Automated CI workflow with configurable metric regression thresholds.

---

## 14. Phase 15: Advanced Research Automation & Guided Scientific Workflows

```mermaid
graph TD
    A[1. Literature Discovery] --> B[2. Evidence Collection]
    B --> C[3. Gap Validation]
    C --> D[4. Research Question]
    D --> E[5. Hypothesis Proposal]
    E --> F[6. Experiment Plan]
    F --> G{7. Researcher Approval Gate}
    G -- Approved --> H[8. Bounded Matrix]
    G -- Rejected --> F
    H --> I[9. Controlled Execution]
    I --> J[10. Automated Evaluation]
    J --> K[11. Scientific Comparability]
    K --> L[12. Finding Candidate]
    L --> M[13. Human Review & Confirmation]
```

- **`EvidenceGroundedProposer`**: Synthesizes literature evidence bookmarks, quotations, and confirmed gaps to propose testable hypotheses with explicit assumptions, missing evidence, and measurable criteria. Emits strict scientific integrity notices.
- **`Researcher Approval Gate`**: Enforces strict RBAC check preventing autonomous execution until an authorized researcher (`RESEARCHER` or `OWNER`) provides explicit approval.
- **`FindingCandidateGenerator`**: Synthesizes validated empirical run metrics, baseline comparisons, limitations, and uncertainty into candidate findings with `PENDING_RESEARCHER_REVIEW` status.
- **`ResearchWorkflowOrchestrator`**: Manages the end-to-end 13-stage lifecycle, transitions history, failure recovery, cancellation, and audit trail.
- **`Scientific Integrity Demarcation`**: Strictly enforces `EXECUTED` != `REPRODUCED` != `EVALUATED` != `COMPARABLE` != `RESEARCH FINDING` != `SCIENTIFIC CONCLUSION`. Never claims unverified statistical significance without multi-seed testing.




