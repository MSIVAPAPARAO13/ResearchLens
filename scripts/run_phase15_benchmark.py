"""Performance benchmark script for Phase 15 Advanced Research Automation."""

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

# Ensure src is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from researchlens.automation.finding_candidate import FindingCandidateGenerator
from researchlens.automation.proposer import EvidenceGroundedProposer
from researchlens.automation.workflow import ResearchWorkflowOrchestrator
from researchlens.workspace.schema import (
    EvidenceBookmark,
    ExperimentResult,
    ExperimentRun,
    ResearchGap,
    ResearchQuestion,
    SavedPaper,
    WorkspaceMember,
)
from researchlens.workspace.store import WorkspaceStore


def benchmark_phase15():
    temp_dir = tempfile.mkdtemp(prefix="rl_bench_p15_")
    store = WorkspaceStore(base_dir=os.path.join(temp_dir, "workspaces"))
    ws = store.create_workspace("Phase 15 Benchmark Workspace")
    ws_id = ws.workspace_id

    store.save_member(WorkspaceMember("m_owner", ws_id, "Alice", "alice@lab.org", "OWNER"))
    store.save_member(WorkspaceMember("m_researcher", ws_id, "Bob", "bob@lab.org", "RESEARCHER"))
    store.save_member(WorkspaceMember("m_reviewer", ws_id, "Carol", "carol@lab.org", "REVIEWER"))

    paper = SavedPaper(paper_id="p_bench", workspace_id=ws_id, title="Bench Paper", authors=["Author"], year=2024)
    store.save_paper(paper)

    bm = EvidenceBookmark(bookmark_id="bm_bench", workspace_id=ws_id, paper_id="p_bench", source_text="Sample text", page_number=1)
    store.save_evidence(bm)

    gap = ResearchGap(gap_id="gap_bench", workspace_id=ws_id, description="Sample gap", verified=True)
    store.save_gap(gap)

    orchestrator = ResearchWorkflowOrchestrator(store)
    proposer = EvidenceGroundedProposer(ws_id)
    generator = FindingCandidateGenerator(ws_id)

    iterations = [10, 100, 1000]
    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": 15,
        "platform": os.name,
        "benchmarks": {},
    }

    # 1. Benchmark: Workflow Initialization
    results["benchmarks"]["workflow_initialization"] = {}
    init_iterations = [10, 50, 100]
    for n in init_iterations:
        t0 = time.perf_counter()
        for i in range(n):
            _ = orchestrator.initialize_workflow(ws_id, f"Bench WF {n}_{i}", topic="QA")
        elapsed = time.perf_counter() - t0
        per_op_ms = (elapsed / n) * 1000
        results["benchmarks"]["workflow_initialization"][f"{n}_runs"] = {
            "total_ms": round(elapsed * 1000, 4),
            "per_op_ms": round(per_op_ms, 4),
            "ops_per_sec": round(n / elapsed, 2),
        }

    # 2. Benchmark: Evidence-Grounded Hypothesis Proposal
    results["benchmarks"]["hypothesis_proposal"] = {}
    q = ResearchQuestion(question_id="q_b", workspace_id=ws_id, text="How to scale?")
    bookmarks = [bm]
    for n in iterations:
        t0 = time.perf_counter()
        for _ in range(n):
            _ = proposer.propose_hypothesis(q, bookmarks=bookmarks)
        elapsed = time.perf_counter() - t0
        per_op_ms = (elapsed / n) * 1000
        results["benchmarks"]["hypothesis_proposal"][f"{n}_runs"] = {
            "total_ms": round(elapsed * 1000, 4),
            "per_op_ms": round(per_op_ms, 4),
            "ops_per_sec": round(n / elapsed, 2),
        }

    # 3. Benchmark: Approval Gate Validation
    _ = orchestrator.initialize_workflow(ws_id, "Approval Bench WF")
    results["benchmarks"]["approval_gate_validation"] = {}
    for n in iterations:
        t0 = time.perf_counter()
        for _ in range(n):
            _ = store.has_permission(ws_id, "m_researcher", "RESEARCHER")
        elapsed = time.perf_counter() - t0
        per_op_ms = (elapsed / n) * 1000
        results["benchmarks"]["approval_gate_validation"][f"{n}_runs"] = {
            "total_ms": round(elapsed * 1000, 4),
            "per_op_ms": round(per_op_ms, 4),
            "ops_per_sec": round(n / elapsed, 2),
        }

    # 4. Benchmark: Finding Candidate Generation
    run = ExperimentRun(run_id="r_b", experiment_id="exp_b", status="COMPLETED")
    res_list = [ExperimentResult(result_id="res_b", run_id="r_b", model="Hybrid", dataset="PubMedQA", metric="MRR", value=0.85)]
    base_list = [ExperimentResult(result_id="res_base", run_id="r_base", model="BM25", dataset="PubMedQA", metric="MRR", value=0.80)]
    results["benchmarks"]["finding_candidate_generation"] = {}
    for n in iterations:
        t0 = time.perf_counter()
        for _ in range(n):
            _ = generator.generate_candidate(
                experiment_id="exp_b",
                run=run,
                results=res_list,
                baseline_results=base_list,
            )
        elapsed = time.perf_counter() - t0
        per_op_ms = (elapsed / n) * 1000
        results["benchmarks"]["finding_candidate_generation"][f"{n}_runs"] = {
            "total_ms": round(elapsed * 1000, 4),
            "per_op_ms": round(per_op_ms, 4),
            "ops_per_sec": round(n / elapsed, 2),
        }

    # Save results
    os.makedirs("evaluation/results", exist_ok=True)
    out_file = "evaluation/results/phase15_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Phase 15 benchmark completed. Results saved to {out_file}")
    shutil.rmtree(temp_dir, ignore_errors=True)
    return results


if __name__ == "__main__":
    benchmark_phase15()
