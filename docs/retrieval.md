# ResearchLens Scientific Retrieval & Reranking Architecture

## 1. Overview

In scientific literature retrieval, general-purpose vector search often suffers from precision loss when addressing exact technical terms, hyperparameter values, model variations, and acronyms. Conversely, purely lexical matching fails when questions use abstract or paraphrase phrasing.

To resolve this limitation, **ResearchLens Phase 3** implements a **two-stage hybrid retrieval and cross-encoder reranking pipeline**:

1. **Stage 1 (Candidate Generation):**
   - **Dense Retriever (`sentence-transformers/all-MiniLM-L6-v2` + FAISS):** Captures high-level semantic intent and paraphrased questions.
   - **BM25 Lexical Retriever (`rank-bm25` BM25Okapi):** Captures exact academic terminology, model names (`ResNet-50`, `SciTransformer`), datasets (`ImageNet-1K`, `PubMed-200k`), and numerical metrics (`RMSE=0.284`).
2. **Score Normalization & Fusion:**
   - Min-max normalizes disparate dense inner product and BM25 score distributions into $[0.0, 1.0]$.
   - Fuses scores via configurable convex linear combination ($\text{weight}_{\text{dense}} = 0.6, \text{weight}_{\text{bm25}} = 0.4$).
3. **Stage 2 (Candidate Reranking):**
   - **Cross-Encoder Reranker (`cross-encoder/ms-marco-MiniLM-L6-v2`):** Applies full transformer self-attention over concatenated $(query, passage)$ pairs, capturing fine-grained token-level cross-interactions.
   - Re-sorts candidates and extracts the top $K$ evidence excerpts for grounded answer generation.

---

## 2. Why Dense Retrieval Exists

Dense retrieval maps queries and document chunks into a continuous 384-dimensional vector space:
- **Semantic Generalization:** Understands synonyms and abstract phrasing (e.g., *"What problem does the paper address?"* matches text describing *"challenges with domain-specific terminology and multi-modal layouts"*).
- **Fast First-Stage Search:** FAISS `IndexFlatIP` computes dot products across thousands of vectors in sub-millisecond latency.
- **Limitation:** Bi-encoders compress entire chunks into a single vector, losing fine-grained lexical signals such as specific numerical scores, equations, or exact model identifiers.

---

## 3. Why BM25 Exists

Academic papers are characterized by dense technical jargon:
- **Exact Terminology:** Phrases such as `AdamW`, `learning rate 2e-5`, `A100 GPUs`, `RMSE`, and `F1-score` must match exactly.
- **Scientific Tokenization:** Our custom tokenizer `tokenize_for_bm25` preserves hyphenated entity names (e.g., `resnet-50`, `macro-f1`), acronyms, and digits while discarding punctuation noise.
- **Local & Lightweight:** Uses an in-memory `BM25Okapi` inverted index without requiring external server dependencies like Elasticsearch or OpenSearch.
- **Limitation:** BM25 fails completely when the query uses synonyms or conceptual formulations not verbatim in the text.

---

## 4. Why Hybrid Retrieval Is Useful

Dense and lexical retrieval have orthogonal error surfaces. Combining both ensures high candidate recall:
- Queries with explicit entities are captured by BM25.
- Queries with conceptual or paraphrased questions are captured by Dense.

### Score Normalization
Because FAISS inner products (range $\approx [0.2, 0.9]$) and BM25 scores (range $[0.0, \infty)$) operate on incompatible scales, scores are normalized via min-max scaling before linear combination:

$$\text{norm}(s_i) = \frac{s_i - \min(S)}{\max(S) - \min(S)}$$

$$\text{score}_{\text{hybrid}} = w_{\text{dense}} \cdot \text{norm}(s_{\text{dense}}) + w_{\text{bm25}} \cdot \text{norm}(s_{\text{bm25}})$$

---

## 5. Why Cross-Encoder Reranking Is Needed

While bi-encoders generate embeddings for queries and passages independently, a **Cross-Encoder** feeds $(query, passage)$ simultaneously into all attention layers:
- Allows every query token to attend directly to every passage token.
- Produces significantly more accurate relevance discrimination.
- Is computationally too expensive for full-corpus indexing, but ideal as a second-stage reranker over top candidate pools ($N = 10$).

---

## 6. Pipeline Flow

```
Question
   |
   +----------------------+
   |                      |
   v                      v
Dense Retriever       BM25 Retriever
 (Top 10 Chunks)       (Top 10 Chunks)
   |                      |
   +----------+-----------+
              |
              v
       Min-Max Normalization
              |
              v
        Score Fusion (0.6 / 0.4)
              |
              v
     Fused Candidate Pool
              |
              v
       Cross-Encoder Reranker
      (ms-marco-MiniLM-L6-v2)
              |
              v
       Top 5 Evidence Chunks
              |
              v
     LocalQuestionAnswerer
              |
              v
  Grounded Answer + Citations
```

---

## 7. Preserved Provenance Contract

Every returned `SearchResult` retains:
- `chunk_id`: Unique chunk identifier
- `document_id`: Parent document hash
- `filename`: Source PDF filename
- `page_number`: Exact PDF page index
- `section_name`: Canonical academic section header
- `chunk_index`: Zero-based chunk order within section
- `retrieval_score`: First-stage fused hybrid score
- `reranker_score`: Second-stage Cross-Encoder prediction score
- `text`: Verbatim evidence excerpt
