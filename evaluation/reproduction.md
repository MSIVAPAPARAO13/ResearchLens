# ResearchLens Phase 8 Reproduction Guide

This document provides exact, deterministic commands to reproduce all evaluations, regression suites, benchmarks, and applications for **ResearchLens** (Phase 8: Full-System Evaluation, Benchmarking & Production Hardening).

---

## 1. System & Runtime Prerequisites

- **Operating System**: Windows 11 (64-bit) / Linux (Ubuntu 22.04+) / macOS (Sonoma+)
- **Python Version**: `Python 3.13` (Tested and verified on `Python 3.13.13`)
- **Key Pre-trained Models**:
  - Embedding Model: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors)
  - Cross-Encoder: `cross-encoder/ms-marco-MiniLM-L6-v2` (sequence classification reranker)
  - Summarization Model: `facebook/bart-large-cnn` (local BART pipeline with cached singleton)
  - Vector Store: `faiss-cpu >= 1.8.0` (IndexFlatIP cosine similarity)

---

## 2. Environment Setup & Dependency Installation

### Step A: Clone & Create Virtual Environment
```powershell
# In PowerShell (Windows)
git clone https://github.com/MSIVAPAPARAO13/ResearchLens.git
cd ResearchLens

python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step B: Install Dependencies
```powershell
# Upgrade pip and packaging tools
python -m pip install --upgrade pip setuptools wheel

# Install production and development dependencies
pip install -r requirements.txt

# Install package in editable development mode
pip install -e .
```

---

## 3. Static Code Quality & Type Verification

Run code quality, linting, and bytecode compilation checks:

```powershell
# 1. Bytecode compilation across all modules
python -m compileall src app tests scripts evaluation

# 2. Ruff static linter and import checker
python -m ruff check .
```

---

## 4. Full Regression Test Suite Execution

Execute the complete automated test suite across all 18 subsystems (117 test cases):

```powershell
# Run full pytest suite with verbose output
python -m pytest -v

# Run with test coverage report
python -m pytest --cov=researchlens --cov-report=term-missing
```

---

## 5. Master Benchmark & Evaluation Suite

Execute the multi-paper gold-standard benchmark script across the 10-paper multi-domain corpus:

```powershell
# Runs retrieval comparison, RAG citation grounding, information extraction,
# multi-provider literature discovery, deduplication, and model caching audits
python scripts/run_phase8_benchmark.py
```

Results are automatically saved to:
- `evaluation/results/phase8_results.json`

---

## 6. Interactive Streamlit Application Launch

Launch the ResearchLens interactive user interface:

```powershell
# Run Streamlit web UI locally
python -m streamlit run app/streamlit_app.py
```

Access the interface in your browser at:
`http://localhost:8501`

### Offline Mode
To run fully offline without external API calls:
1. Ensure model weights are cached in Hugging Face cache (`~/.cache/huggingface/hub`).
2. Upload PDFs directly in the **Paper Analysis** tab.
3. Dense indexing, BM25 indexing, hybrid retrieval, local Q&A with citations, information extraction, and multi-paper intelligence run 100% locally on CPU/GPU without external network requests.
