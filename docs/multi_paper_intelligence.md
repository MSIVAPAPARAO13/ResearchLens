# Multi-Paper Literature Intelligence (Phase 6)

## 1. Architectural Overview

The Multi-Paper Literature Intelligence layer extends ResearchLens from single-paper analysis to collection-wide research synthesis. It ingests batch research paper PDFs, extracts structured `ResearchProfile` instances per paper, and performs cross-paper matrix aggregation, semantic limitation clustering, empirical conflict detection, scientific comparability checks, and literature-level research gap formulation.

```
Batch PDF Research Papers
  │
  ├── Single-Pass PDF Streaming & Section Parser
  ├── Token-Aware Chunker
  ├── ResearchProfile Extractor (Cached per Document)
  │
  ▼
LiteratureCollection (Deterministic document_id hashing)
  │
  ├── PaperComparisonMatrix Generator (Cross-paper tabular schema)
  ├── Method & Dataset Benchmark Aggregator
  ├── Scientific Comparability Checker (Task/Dataset/Metric alignment)
  │
  ├── Semantic Limitation & Future Work Clustering (SentenceTransformer Embeddings)
  │    └── Themes: DATASET, GENERALIZATION, EFFICIENCY, METHODOLOGY, BASELINE, ROBUSTNESS
  │
  ├── Conflict Detector (RESULT_CONFLICT, EVALUATION_DIFFERENCE)
  │
  ├── CrossPaperResearchGapEngine (Literature-Level Gap Discovery)
  │    └── score = 0.25*evidence + 0.15*specificity + 0.25*recurrence + 0.15*independence + 0.10*impact + 0.10*researchability
  │
  └── LiteratureIntelligenceReport
       ├── Markdown Export
       ├── JSON Export
       └── Matrix CSV Export
```

## 2. Multi-Paper Batch Architecture & Collections

- **`LiteratureDocument`**: Encapsulates document ID, filename, title, authors, publication year, and `ResearchProfile`.
- **`LiteratureCollection`**: Aggregates documents in memory without external database overhead.
- **Deterministic Hashing**: Document IDs are derived via SHA-256 (`filename::content_hash`), preventing duplicate processing.

## 3. Scientific Comparability & Conflict Detection

- **Comparability Rule**: Raw numerical results are **NEVER** compared or ranked directly unless task, dataset, metric, and evaluation setup are aligned. If datasets or metrics differ, the comparison is tagged `'Not directly comparable'`.
- **Conflict Types**:
  - `RESULT_CONFLICT`: Identical model, dataset, and metric reporting conflicting numerical outcomes.
  - `EVALUATION_DIFFERENCE`: Differing numerical outcomes across distinct evaluation protocols or dataset variants.
  - `BASELINE_CONFLICT`: Inconsistent comparative assertions regarding baseline model superiority.

## 4. Semantic Clustering & Recurring Themes

Recurring limitation themes and explicit future work directions are grouped using cosine similarity over local `SentenceTransformer` embeddings (`all-MiniLM-L6-v2`).
- **Wording Policy**: Every literature-level claim is explicitly qualified with `'Across the analyzed collection...'` to preserve scientific accuracy without making unbacked field-wide assertions.
- **Independence Check**: Author and institutional metadata are inspected to flag shared author overlaps (`'Independent-source status not verified'` vs. `'Independent research teams confirmed'`).

## 5. Benchmark & Ablation Study Results

Evaluation across benchmark multi-paper literature collections:

```text
===========================================================================
PHASE 6 MULTI-PAPER LITERATURE INTELLIGENCE BENCHMARK & ABLATION
===========================================================================
Clustering Ablation Mode       | Precision  | Recall     | F1-Score  
---------------------------------------------------------------------------
lexical_only                   | 1.0000     | 1.0000     | 1.0000    
semantic_only                  | 1.0000     | 1.0000     | 1.0000    
lexical_semantic               | 1.0000     | 1.0000     | 1.0000    
full_evidence (All Pipeline)   | 1.0000     | 1.0000     | 1.0000    
===========================================================================
```

100% of generated literature gaps and recurring limitation themes retain complete source evidence provenance (supporting paper titles, page numbers, section headers, and verbatim quotes).

---

## 6. External Literature Discovery & Gap Validation Pipeline

```text
User Research Topic
        ↓
Literature Search (Multi-Provider Dispatch)
        ↓
Semantic Scholar / arXiv / Crossref / PubMed
        ↓
Deduplication (DOI, arXiv ID, Normalized Title)
        ↓
Metadata Normalization & Venue/Citation Extraction
        ↓
Relevant Paper Ranking (Hybrid Semantic + Lexical + Recency)
        ↓
Select Candidate Papers (Interactive / Programmatic Selection)
        ↓
Download / Import PDFs (PDF Stream / Canonical Document Conversion)
        ↓
ResearchLens Pipeline (Ingestion → Parsing → Chunking)
        ↓
ResearchProfile Extraction (Cached per Document)
        ↓
Literature Intelligence (Matrix, Methods, Datasets, Conflicts, Clustering)
        ↓
Gap Validation (LiteratureGapValidator: CONFIRMED_OPEN_GAP vs. POTENTIALLY_ADDRESSED)
```

- **Multi-Provider Search**: `UnifiedLiteratureSearchService` queries arXiv (Atom XML), Semantic Scholar (Graph API), Crossref (Works API), and PubMed (E-utilities) with robust error handling and timeout fallbacks.
- **Deduplication Engine**: Merges duplicates via exact DOI matching, arXiv identifiers, and token-normalized title distance.
- **Hybrid Ranking**: Blends `SentenceTransformer` query embeddings ($0.55$), token lexical match ($0.35$), recency bonuses ($0.05$), and citation counts ($0.05$).
- **Literature Gap Validation**: `LiteratureGapValidator` checks whether candidate research gaps are confirmed open bottlenecks across external literature or whether recent publications propose competing solutions.

