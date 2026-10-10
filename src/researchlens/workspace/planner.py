"""Research planner, direction prioritization, risk inference, traceability, and export engine."""

import csv
import io
import json
from typing import Any, Dict, List, Optional, Tuple

from researchlens.utils.logging import get_logger
from researchlens.workspace.schema import (
    EvidenceBookmark,
    ExperimentPlan,
    ResearchDirection,
    ResearchHypothesis,
    ResearchPlan,
    ResearchQuestion,
    ResearchRisk,
    generate_id,
)

logger = get_logger("research_planner")


def calculate_direction_priority_score(
    evidence_strength: float = 0.5,
    research_gap_strength: float = 0.5,
    feasibility: float = 0.5,
    impact: float = 0.5,
    data_availability: float = 0.5,
    evaluation_feasibility: float = 0.5,
) -> Tuple[float, Dict[str, float]]:
    """Calculate transparent heuristic Priority Score for a research direction.

    Formula:
      Priority Score =
        0.25 * evidence_strength +
        0.20 * research_gap_strength +
        0.15 * feasibility +
        0.15 * impact +
        0.15 * data_availability +
        0.10 * evaluation_feasibility
    """
    weights = {
        "evidence_strength": 0.25,
        "research_gap_strength": 0.20,
        "feasibility": 0.15,
        "impact": 0.15,
        "data_availability": 0.15,
        "evaluation_feasibility": 0.10,
    }
    raw_values = {
        "evidence_strength": max(0.0, min(1.0, float(evidence_strength))),
        "research_gap_strength": max(0.0, min(1.0, float(research_gap_strength))),
        "feasibility": max(0.0, min(1.0, float(feasibility))),
        "impact": max(0.0, min(1.0, float(impact))),
        "data_availability": max(0.0, min(1.0, float(data_availability))),
        "evaluation_feasibility": max(0.0, min(1.0, float(evaluation_feasibility))),
    }

    total_score = sum(weights[k] * raw_values[k] for k in weights)
    score_breakdown = {k: round(raw_values[k], 4) for k in raw_values}
    score_breakdown["total_score"] = round(total_score, 4)
    return round(total_score, 4), score_breakdown


def infer_risks_from_limitations(
    workspace_id: str,
    limitations: List[Dict[str, Any]],
) -> List[ResearchRisk]:
    """Inspect literature limitation excerpts and derive evidence-based research risks."""
    risks: List[ResearchRisk] = []
    seen_categories = set()

    for item in limitations:
        text = str(item.get("text", "") or item.get("source_text", "")).lower()
        source = item.get("paper_title", item.get("paper_id", "Analyzed literature"))

        if any(w in text for w in ["single dataset", "limited dataset", "cross-domain", "generalization", "out-of-domain"]):
            cat = "GENERALIZATION"
            if cat not in seen_categories:
                seen_categories.add(cat)
                risks.append(
                    ResearchRisk(
                        risk_id=generate_id("risk"),
                        workspace_id=workspace_id,
                        category=cat,
                        description="Potential generalization risk: related papers frequently evaluate on limited datasets or observe cross-dataset drops.",
                        evidence=[f"Observed in {source}: '{item.get('text', '')[:140]}'"],
                        severity="MEDIUM",
                        mitigation="Evaluate candidate model across at least two domain-diverse public benchmarks.",
                    )
                )

        if any(w in text for w in ["compute", "memory", "gpu", "latency", "scale", "overhead", "runtime"]):
            cat = "COMPUTE"
            if cat not in seen_categories:
                seen_categories.add(cat)
                risks.append(
                    ResearchRisk(
                        risk_id=generate_id("risk"),
                        workspace_id=workspace_id,
                        category=cat,
                        description="Compute and latency overhead: literature notes significant computational requirements during scaling.",
                        evidence=[f"Observed in {source}: '{item.get('text', '')[:140]}'"],
                        severity="MEDIUM",
                        mitigation="Include inference latency and peak memory profiling in evaluation protocol.",
                    )
                )

        if any(w in text for w in ["leakage", "contamination", "overlap"]):
            cat = "DATA_LEAKAGE"
            if cat not in seen_categories:
                seen_categories.add(cat)
                risks.append(
                    ResearchRisk(
                        risk_id=generate_id("risk"),
                        workspace_id=workspace_id,
                        category=cat,
                        description="Data contamination or leakage risk between pretraining and benchmark evaluation splits.",
                        evidence=[f"Observed in {source}: '{item.get('text', '')[:140]}'"],
                        severity="HIGH",
                        mitigation="Enforce strict deduplication and date-based train/test splits.",
                    )
                )

        if any(w in text for w in ["unreleased", "reproducibility", "closed source", "proprietary"]):
            cat = "REPRODUCIBILITY"
            if cat not in seen_categories:
                seen_categories.add(cat)
                risks.append(
                    ResearchRisk(
                        risk_id=generate_id("risk"),
                        workspace_id=workspace_id,
                        category=cat,
                        description="Reproducibility risk: baseline weights, datasets, or code may lack open availability.",
                        evidence=[f"Observed in {source}: '{item.get('text', '')[:140]}'"],
                        severity="HIGH",
                        mitigation="Prioritize verified open-source baselines with publicly available checkpoints.",
                    )
                )

    return risks


class ResearchPlanner:
    """Orchestrates draft research plan synthesis, traceability tracking, and multi-format exports."""

    @staticmethod
    def generate_draft_plan(
        workspace_id: str,
        title: str,
        problem_statement: str,
        questions: List[ResearchQuestion],
        hypotheses: List[ResearchHypothesis],
        directions: List[ResearchDirection],
        bookmarks: List[EvidenceBookmark],
        candidate_models: Optional[List[Dict[str, Any]]] = None,
        candidate_datasets: Optional[List[Dict[str, Any]]] = None,
        candidate_baselines: Optional[List[Dict[str, Any]]] = None,
        candidate_metrics: Optional[List[str]] = None,
        inferred_risks: Optional[List[ResearchRisk]] = None,
    ) -> ResearchPlan:
        """Construct a structured, evidence-grounded draft research plan."""
        traceability: List[Dict[str, Any]] = []

        # Trace questions
        q_texts = []
        for q in questions:
            q_texts.append(q.text)
            if q.supporting_evidence_ids:
                traceability.append({
                    "component": f"Research Question: {q.text}",
                    "type": "QUESTION",
                    "evidence_ids": q.supporting_evidence_ids,
                    "gap_ids": q.supporting_gap_ids,
                })

        # Trace hypotheses
        h_texts = []
        for h in hypotheses:
            h_texts.append(h.statement)
            if h.supporting_evidence:
                traceability.append({
                    "component": f"Hypothesis: {h.statement}",
                    "type": "HYPOTHESIS",
                    "evidence": h.supporting_evidence,
                    "question_id": h.question_id,
                })

        # Process datasets with traceability
        datasets = candidate_datasets or []
        for d in datasets:
            traceability.append({
                "component": f"Dataset: {d.get('name', 'Unknown')}",
                "type": "DATASET",
                "source_paper": d.get("source_paper", "Monitored literature"),
                "provenance": d.get("provenance", ""),
            })

        # Process baselines with traceability
        baselines = candidate_baselines or []
        for b in baselines:
            traceability.append({
                "component": f"Baseline: {b.get('name', 'Unknown')}",
                "type": "BASELINE",
                "source_paper": b.get("source_paper", "Monitored literature"),
                "provenance": b.get("provenance", ""),
            })

        # Process metrics
        metrics = candidate_metrics or ["Accuracy", "F1", "Latency"]
        for m in metrics:
            traceability.append({
                "component": f"Metric: {m}",
                "type": "METRIC",
                "source": "Commonly reported in analyzed literature",
            })

        # Evidence bookmarks traceability
        for bm in bookmarks:
            traceability.append({
                "component": f"Evidence Bookmark: {bm.label or bm.source_text[:40]}",
                "type": "EVIDENCE",
                "paper_id": bm.paper_id,
                "page": bm.page_number,
                "section": bm.section_name,
                "chunk_id": bm.chunk_id,
                "source_text": bm.source_text,
            })

        methodology = {
            "candidate_models": candidate_models or [{"name": "Literature-derived model", "source": "Analyzed papers"}],
            "training_setup": "Domain-specific fine-tuning with fixed random seeds and learning rate schedules.",
            "validation_protocol": "Stratified k-fold cross-validation or out-of-domain evaluation.",
        }

        milestones = [
            "Literature Synthesis & Baseline Verification",
            "Dataset Ingestion & Quality Preprocessing",
            "Baseline Implementation & Reproduction",
            "Proposed Model Architecture Implementation",
            "Empirical Benchmark Evaluation & Ablations",
            "Research Paper Manuscript Drafting",
        ]

        risks = [r.description for r in (inferred_risks or [])]

        return ResearchPlan(
            plan_id=generate_id("rp"),
            workspace_id=workspace_id,
            title=title,
            problem_statement=problem_statement,
            research_questions=q_texts,
            hypotheses=h_texts,
            methodology=methodology,
            datasets=datasets,
            baselines=baselines,
            metrics=metrics,
            experiments=[],
            risks=risks,
            milestones=milestones,
            traceability=traceability,
        )

    @staticmethod
    def validate_plan(plan: ResearchPlan) -> Dict[str, Any]:
        """Validate research plan readiness and return completeness audit."""
        missing = []
        warnings = []

        if not plan.problem_statement.strip():
            missing.append("Problem statement is missing.")
        if not plan.research_questions:
            missing.append("At least one research question is required.")
        if not plan.hypotheses:
            warnings.append("No explicit hypotheses linked yet.")
        if not plan.datasets:
            missing.append("No evaluation datasets identified.")
        if not plan.baselines:
            warnings.append("No competitive baseline models specified.")
        if not plan.metrics:
            missing.append("No evaluation metrics selected.")
        if not plan.traceability:
            warnings.append("No evidence traceability citations attached.")

        is_complete = len(missing) == 0
        return {
            "status": "COMPLETE" if is_complete else "INCOMPLETE",
            "missing_components": missing,
            "warnings": warnings,
            "traceability_count": len(plan.traceability),
        }

    @staticmethod
    def export_plan_to_markdown(plan: ResearchPlan) -> str:
        """Export research plan to structured Markdown format."""
        lines = [
            f"# Research Plan: {plan.title}",
            "",
            "> **Scientific Integrity Notice:** This research plan is an evidence-grounded planning artifact generated by ResearchLens. Candidate components are literature-derived suggestions and do not constitute a peer-reviewed experimental protocol until verified and confirmed by the researcher.",
            "",
            "## 1. Problem Statement",
            plan.problem_statement or "_No problem statement specified._",
            "",
            "## 2. Research Questions",
        ]
        if plan.research_questions:
            for i, q in enumerate(plan.research_questions, 1):
                lines.append(f"{i}. {q}")
        else:
            lines.append("_No research questions specified._")

        lines.extend(["", "## 3. Hypotheses"])
        if plan.hypotheses:
            for i, h in enumerate(plan.hypotheses, 1):
                lines.append(f"{i}. {h}")
        else:
            lines.append("_No hypotheses formulated yet._")

        lines.extend(["", "## 4. Candidate Datasets"])
        if plan.datasets:
            for d in plan.datasets:
                lines.append(f"- **{d.get('name', 'Dataset')}**: {d.get('description', '')} (Source: {d.get('source_paper', 'Literature')})")
        else:
            lines.append("_No datasets assigned._")

        lines.extend(["", "## 5. Baselines"])
        if plan.baselines:
            for b in plan.baselines:
                lines.append(f"- **{b.get('name', 'Baseline')}**: {b.get('description', '')} (Source: {b.get('source_paper', 'Literature')})")
        else:
            lines.append("_No baselines defined._")

        lines.extend(["", "## 6. Evaluation Metrics"])
        if plan.metrics:
            lines.append(", ".join(plan.metrics))
        else:
            lines.append("_No evaluation metrics selected._")

        lines.extend(["", "## 7. Known Research Risks & Mitigations"])
        if plan.risks:
            for r in plan.risks:
                lines.append(f"- ⚠️ {r}")
        else:
            lines.append("_No open risks recorded._")

        lines.extend(["", "## 8. Project Milestones"])
        if plan.milestones:
            for i, m in enumerate(plan.milestones, 1):
                lines.append(f"{i}. [ ] {m}")
        else:
            lines.append("_No timeline milestones set._")

        lines.extend(["", "## 9. Literature Traceability & Provenance"])
        if plan.traceability:
            lines.append("| Component | Type | Source / Provenance |")
            lines.append("|---|---|---|")
            for t in plan.traceability:
                comp = t.get("component", "Item")
                t_type = t.get("type", "GENERAL")
                src = t.get("source_paper") or t.get("source") or t.get("paper_id") or "Workspace"
                lines.append(f"| {comp} | {t_type} | {src} |")
        else:
            lines.append("_No explicit provenance records attached._")

        return "\n".join(lines)

    @staticmethod
    def export_plan_to_json(plan: ResearchPlan) -> str:
        """Export research plan to JSON string."""
        return json.dumps(plan.to_dict(), indent=2, ensure_ascii=False)

    @staticmethod
    def export_experiments_to_csv(experiments: List[ExperimentPlan]) -> str:
        """Export experiment plans to CSV formatted table."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Experiment ID", "Name", "Hypothesis ID", "Model", "Baseline", "Dataset", "Metrics", "Status"])
        for exp in experiments:
            writer.writerow([
                exp.experiment_id,
                exp.name,
                exp.hypothesis_id or "N/A",
                exp.model,
                exp.baseline,
                exp.dataset,
                "; ".join(exp.metrics),
                exp.status,
            ])
        return output.getvalue()

    @staticmethod
    def create_plan_version(
        base_plan: ResearchPlan,
        changed_fields: Dict[str, Any],
        author_id: str = "",
    ) -> ResearchPlan:
        """Create a new version of a research plan without overwriting the previous version."""
        d = base_plan.to_dict()
        d["parent_plan_id"] = base_plan.plan_id
        d["plan_id"] = generate_id("rp")
        d["version"] = base_plan.version + 1
        d["author_id"] = author_id

        for k, v in changed_fields.items():
            if k in d:
                d[k] = v

        return ResearchPlan.from_dict(d)

    @staticmethod
    def generate_experiment_report(
        experiment: ExperimentPlan,
        run: Optional[Any] = None,
        results: Optional[List[Any]] = None,
        comparisons: Optional[List[Dict[str, Any]]] = None,
        reproducibility: Optional[Any] = None,
        findings: Optional[List[Any]] = None,
        review: Optional[Any] = None,
    ) -> str:
        """Generate structured Experiment Report covering all 13 required sections."""
        lines = [
            f"# Experiment Report: {experiment.name} (v{experiment.version})",
            "",
            "> **Scientific Integrity Notice:** This experiment report synthesizes imported empirical results and researcher reviews. Imported values reflect external evaluation logs and do not constitute universal peer-reviewed claims.",
            "",
            "## 1. Objective",
            f"Evaluate candidate architecture `{experiment.model}` against `{experiment.baseline}` on `{experiment.dataset}`.",
            "",
            "## 2. Tested Hypothesis",
            f"Hypothesis ID: `{experiment.hypothesis_id or 'None'}`",
            "",
            "## 3. Dataset Specification",
            f"Dataset: **{experiment.dataset}**",
            "",
            "## 4. Model Specification",
            f"Proposed Model: **{experiment.model}**",
            "",
            "## 5. Baseline Model",
            f"Baseline Model: **{experiment.baseline}**",
            "",
            "## 6. Experimental Configuration",
        ]
        if run:
            lines.append(f"- Run ID: `{run.run_id}` | Seed: `{run.seed}` | Environment: `{run.environment or 'N/A'}`")
            lines.append(f"- Hardware: `{run.hardware or 'N/A'}`")
        else:
            lines.append("_No execution run recorded._")

        lines.extend(["", "## 7. Evaluated Metrics", ", ".join(experiment.metrics) if experiment.metrics else "None"])

        lines.extend(["", "## 8. Empirical Results", "| Model | Dataset | Metric | Value | Split | Seed |", "|---|---|---|---|---|---|"])
        for r in (results or []):
            lines.append(f"| {r.model} | {r.dataset} | {r.metric} | {r.value:.4f} | {r.split} | {r.seed or 'N/A'} |")
        if not results:
            lines.append("| _No results recorded_ | | | | | |")

        lines.extend(["", "## 9. Baseline Comparison"])
        if comparisons:
            lines.append("| Metric | Proposed | Baseline | Absolute Diff | Relative Impr. | Comparability |")
            lines.append("|---|---|---|---|---|---|")
            for c in comparisons:
                b_val_str = f"{c['baseline_value']:.4f}" if c.get("baseline_value") is not None else "N/A"
                diff_str = f"{c['absolute_diff']:+.4f}" if c.get("absolute_diff") is not None else "N/A"
                pct_str = f"{c['relative_improvement_pct']:+.2f}%" if c.get("relative_improvement_pct") is not None else "N/A"
                lines.append(f"| {c['metric']} | {c['proposed_value']:.4f} | {b_val_str} | {diff_str} | {pct_str} | `{c['comparability']}` |")
        else:
            lines.append("_No comparative metrics calculated._")

        lines.extend(["", "## 10. Reproducibility Assessment"])
        if reproducibility:
            lines.append(f"- Status: **{reproducibility.completeness_status}**")
            if reproducibility.missing_fields:
                lines.append(f"- Missing Parameters: `{', '.join(reproducibility.missing_fields)}`")
        else:
            lines.append("_Reproducibility audit not performed._")

        lines.extend(["", "## 11. Limitations", "- Single-benchmark evaluation scope.", "- Potential domain bias in benchmark split."])

        lines.extend(["", "## 12. Research Findings"])
        if findings:
            for f in findings:
                lines.append(f"- **{f.title}** ({f.finding_type}): {f.statement} [Confidence: `{f.confidence_level}`]")
        else:
            lines.append("_No findings formalized yet._")

        lines.extend(["", "## 13. Reviewer Decision"])
        if review:
            lines.append(f"- Decision: **{review.decision}** by `{review.reviewer_id}`")
            if review.comments:
                lines.append(f"- Comments: *{review.comments}*")
        else:
            lines.append("_Review decision pending._")

        return "\n".join(lines)

    @staticmethod
    def generate_collaborative_research_report(
        workspace_name: str,
        problem_statement: str,
        plans: List[ResearchPlan],
        questions: List[Any],
        hypotheses: List[Any],
        results: List[Any],
        findings: List[Any],
        members: List[Any],
    ) -> str:
        """Generate comprehensive collaborative workspace report."""
        lines = [
            f"# Collaborative Research Report: {workspace_name}",
            "",
            "> **Scientific Integrity Notice:** Synthesized report representing collaborative findings and literature provenance. Literature-derived components preserve citations; imported experiment results are tagged for researcher validation.",
            "",
            "## 1. Problem Statement",
            problem_statement or "_No problem statement provided._",
            "",
            "## 2. Research Team",
        ]
        if members:
            for m in members:
                lines.append(f"- **{m.display_name}** (`{m.role}`) — {m.email}")
        else:
            lines.append("_Single investigator workspace._")

        lines.extend(["", "## 3. Research Questions & Hypotheses"])
        for q in questions:
            lines.append(f"### Q: {q.text} (`{q.type}`)")
            linked_hyps = [h for h in hypotheses if getattr(h, "question_id", None) == q.question_id]
            for h in linked_hyps:
                lines.append(f"- Hypothesis: {h.statement} [Status: `{h.status}`]")

        lines.extend(["", "## 4. Empirical Findings & Baseline Comparisons"])
        if findings:
            for f in findings:
                lines.append(f"- **{f.title}** ({f.finding_type}): {f.statement} (Status: `{f.status}`)")
        else:
            lines.append("_No empirical findings recorded._")

        lines.extend(["", "## 5. Traceability & Literature Provenance"])
        if plans and plans[0].traceability:
            lines.append("| Component | Type | Source / Provenance |")
            lines.append("|---|---|---|")
            for t in plans[0].traceability:
                lines.append(f"| {t.get('component', 'Item')} | {t.get('type', 'GENERAL')} | {t.get('source_paper') or t.get('source') or 'Monitored Corpus'} |")

        return "\n".join(lines)
