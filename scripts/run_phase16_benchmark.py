"""Empirical performance and resource limits benchmark script for Phase 16.

Benchmarks:
1. Document Ingestion and Section Parsing (synthetic multi-page academic papers, invalid upload checks)
2. Section Summarization
3. Document Chunking & BM25 / Scientific Retrieval
4. Batch Literature Analysis & Gap Extraction
5. Workspace Persistence (atomic writes, concurrent entity serialization)
6. Deterministic Experiment Execution
7. Latency and Memory profiling (tracemalloc)
"""

import json
import os
import shutil
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict

# Ensure src is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from researchlens.analysis.research_gaps import extract_research_gaps
from researchlens.execution.executor import ExperimentExecutor
from researchlens.execution.registry import ExperimentRegistryService
from researchlens.ingestion.pdf_loader import Document, Page, extract_text_from_pdf
from researchlens.preprocessing import create_document_chunks
from researchlens.preprocessing.section_parser import split_into_sections
from researchlens.retrieval import BM25Retriever
from researchlens.summarization.summarizer import summarize_section
from researchlens.workspace.schema import (
    ExperimentPlan,
    SavedPaper,
    WorkspaceMember,
)
from researchlens.workspace.store import WorkspaceStore


def run_phase16_benchmarks() -> Dict[str, Any]:
    tracemalloc.start()

    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": 16,
        "environment": {
            "os": os.name,
            "python_version": sys.version.split()[0],
            "platform": sys.platform,
        },
        "benchmarks": {},
        "memory_profile_kb": {},
    }

    academic_text = """Abstract
Modern deep learning architectures require robust validation across diverse distribution shifts.
We present an empirical study investigating generalization bounds in sparse transformer models.

1. Introduction
Recent breakthroughs in foundation models have revolutionized scientific discovery.
However, scaling laws often fail when fine-tuning on domain-specific corpora.

2. Related Work
Prior works explored parameter-efficient fine-tuning (PEFT) and low-rank adaptation (LoRA).
We extend these observations to multimodal reasoning tasks.

3. Methodology
We formulate our optimization framework using gradient alignment.
We evaluate on GLUE, SuperGLUE, and synthetic benchmarks with 10-fold cross validation.

4. Results
Our proposed architecture achieves a 3.4% accuracy improvement over competitive baselines.
Latency overhead remains under 5 milliseconds during inference.

5. Discussion
While results demonstrate significant improvements, memory constraints remain a bottleneck.
Future iterations will explore quantized representations.

6. Conclusion
We presented a rigorous empirical evaluation of foundation model adaptation strategies.
References
[1] Vaswani et al. Attention Is All You Need. NeurIPS 2017.
[2] Brown et al. Language Models are Few-Shot Learners. NeurIPS 2020.
""" * 10

    doc = Document(
        filename="benchmark_paper.pdf",
        full_text=academic_text,
        pages=[Page(page_number=i + 1, text=academic_text[i * 1000 : (i + 1) * 1000]) for i in range(5)],
    )

    # 1. Benchmark: Document Chunking and Section Parsing
    n_runs = 50
    t0 = time.perf_counter()
    for _ in range(n_runs):
        sections = split_into_sections(doc.full_text)
        chunks = create_document_chunks(doc)
    elapsed = time.perf_counter() - t0
    results["benchmarks"]["document_processing_and_section_parsing"] = {
        "runs": n_runs,
        "total_ms": round(elapsed * 1000, 4),
        "per_op_ms": round((elapsed / n_runs) * 1000, 4),
        "ops_per_sec": round(n_runs / elapsed, 2),
        "text_length_chars": len(doc.full_text),
        "sections_extracted": len(sections),
        "chunks_generated": len(chunks),
    }

    # Invalid upload validation benchmark
    t0 = time.perf_counter()
    for _ in range(100):
        try:
            extract_text_from_pdf(b"INVALID_HEADER_NOT_A_PDF")
        except Exception:
            pass
    elapsed_inv = time.perf_counter() - t0
    results["benchmarks"]["invalid_upload_handling"] = {
        "runs": 100,
        "total_ms": round(elapsed_inv * 1000, 4),
        "per_op_ms": round((elapsed_inv / 100) * 1000, 4),
        "ops_per_sec": round(100 / elapsed_inv, 2),
    }

    # 2. Benchmark: Section Summarization
    t0 = time.perf_counter()
    n_sum = 3
    for _ in range(n_sum):
        _ = summarize_section("methodology", academic_text[:800])
    elapsed_sum = time.perf_counter() - t0
    results["benchmarks"]["summarization"] = {
        "runs": n_sum,
        "total_ms": round(elapsed_sum * 1000, 4),
        "per_op_ms": round((elapsed_sum / n_sum) * 1000, 4),
        "ops_per_sec": round(n_sum / elapsed_sum, 2),
    }

    # 3. Benchmark: BM25 Scientific Retrieval
    retriever = BM25Retriever()
    t0 = time.perf_counter()
    retriever.index_chunks(chunks)
    indexing_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(100):
        _ = retriever.retrieve("gradient alignment transformer optimization", top_k=5)
    query_time = time.perf_counter() - t0

    results["benchmarks"]["retrieval"] = {
        "indexing_chunks_count": len(chunks),
        "indexing_total_ms": round(indexing_time * 1000, 4),
        "query_runs": 100,
        "query_total_ms": round(query_time * 1000, 4),
        "query_per_op_ms": round((query_time / 100) * 1000, 4),
        "queries_per_sec": round(100 / query_time, 2),
    }

    # 4. Benchmark: Research Gap Analysis
    text_for_gaps = sections.get("discussion", "") + "\n" + sections.get("conclusion", "")
    t0 = time.perf_counter()
    for _ in range(50):
        _ = extract_research_gaps(text_for_gaps)
    elapsed_gap = time.perf_counter() - t0
    results["benchmarks"]["gap_analysis"] = {
        "runs": 50,
        "total_ms": round(elapsed_gap * 1000, 4),
        "per_op_ms": round((elapsed_gap / 50) * 1000, 4),
        "ops_per_sec": round(50 / elapsed_gap, 2),
    }

    # 5. Benchmark: Workspace Persistence (Atomic File I/O)
    temp_dir = tempfile.mkdtemp(prefix="rl_bench_persist_")
    store = WorkspaceStore(base_dir=temp_dir)
    ws = store.create_workspace("Persistence Benchmark WS")
    ws_id = ws.workspace_id

    records_count = 200
    t0 = time.perf_counter()
    for i in range(records_count):
        paper = SavedPaper(
            paper_id=f"paper_persist_{i}",
            workspace_id=ws_id,
            title=f"Persisted Paper #{i}",
        )
        store.save_paper(paper)
    write_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(20):
        loaded = store.list_papers(ws_id)
    read_time = time.perf_counter() - t0

    results["benchmarks"]["workspace_persistence"] = {
        "write_records": records_count,
        "write_total_ms": round(write_time * 1000, 4),
        "write_per_record_ms": round((write_time / records_count) * 1000, 4),
        "read_iterations": 20,
        "read_total_ms": round(read_time * 1000, 4),
        "read_per_call_ms": round((read_time / 20) * 1000, 4),
        "records_loaded_count": len(loaded),
    }

    # 6. Benchmark: Deterministic Experiment Execution
    store.save_member(WorkspaceMember("m_owner", ws_id, "Alice", "alice@lab.org", "OWNER"))
    store.save_member(WorkspaceMember("m_researcher", ws_id, "Bob", "bob@lab.org", "RESEARCHER"))
    entrypoint = "scripts/experiments/deterministic_demo.py"
    registry = ExperimentRegistryService(store)
    registry.register_experiment(ws_id, "Bench Experiment", entrypoint, owner="m_owner")
    plan = ExperimentPlan("exp_bench_exec", name="Bench Plan", entrypoint=entrypoint)
    store.save_experiment(plan, ws_id)
    executor = ExperimentExecutor(store)

    exec_runs = 10
    t0 = time.perf_counter()
    for _ in range(exec_runs):
        run, _ = executor.execute_experiment(ws_id, plan.experiment_id, actor_id="m_researcher")
        assert run is not None and run.status == "COMPLETED"
    exec_time = time.perf_counter() - t0

    results["benchmarks"]["deterministic_experiment_execution"] = {
        "runs": exec_runs,
        "total_ms": round(exec_time * 1000, 4),
        "per_run_ms": round((exec_time / exec_runs) * 1000, 4),
        "runs_per_sec": round(exec_runs / exec_time, 2),
    }

    # Memory measurement
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results["memory_profile_kb"] = {
        "current_kb": round(current_mem / 1024, 2),
        "peak_kb": round(peak_mem / 1024, 2),
        "peak_mb": round(peak_mem / (1024 * 1024), 2),
    }

    shutil.rmtree(temp_dir, ignore_errors=True)
    return results


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "evaluation" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase16_results.json"

    print("Running Phase 16 Performance & Resource Benchmark Suite...")
    benchmark_data = run_phase16_benchmarks()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)

    print(f"Phase 16 benchmarks complete. Results written to: {out_path}")
    print(json.dumps(benchmark_data, indent=2))
