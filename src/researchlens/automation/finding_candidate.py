"""Finding Candidate Generator (Phase 15).

Generates evidence-grounded research finding candidates from validated experiment results
and supporting literature evidence. Enforces that all candidates remain in
PENDING_RESEARCHER_REVIEW status until formally confirmed by a human researcher.
"""

from typing import Any, Dict, List, Optional

from researchlens.utils.logging import get_logger
from researchlens.workspace.schema import (
    ExperimentResult,
    ExperimentRun,
    ResearchFinding,
    ResearchHypothesis,
    ResearchQuestion,
    generate_id,
)

logger = get_logger("finding_candidate_generator")

INTEGRITY_DISCLAIMER = (
    "SCIENTIFIC_INTEGRITY_DISCLAIMER: This candidate finding is an automatically generated synthesis "
    "of empirical run observations and literature context. It does NOT constitute a validated "
    "scientific conclusion. A higher metric does not imply statistical significance unless formal "
    "hypothesis testing has been conducted. Human researcher review is strictly required."
)


class FindingCandidateGenerator:
    """Synthesizes empirical run outputs and literature evidence into candidate research findings."""

    def __init__(self, workspace_id: str):
        self.workspace_id = workspace_id

    def generate_candidate(
        self,
        experiment_id: str,
        run: ExperimentRun,
        results: List[ExperimentResult],
        question: Optional[ResearchQuestion] = None,
        hypothesis: Optional[ResearchHypothesis] = None,
        baseline_results: Optional[List[ExperimentResult]] = None,
        literature_evidence: Optional[List[Dict[str, Any]]] = None,
        reproducibility_status: str = "NOT_ASSESSED",
        dataset_version: str = "1.0",
        model_version: str = "1.0",
    ) -> ResearchFinding:
        """Create a candidate ResearchFinding with complete provenance and review-pending status."""
        observed_metrics: Dict[str, Any] = {}
        result_ids: List[str] = []

        for r in results:
            observed_metrics[r.metric] = r.value
            if r.result_id:
                result_ids.append(r.result_id)

        # Calculate comparison against baseline if provided
        baseline_comparison: Dict[str, Any] = {}
        baseline_metrics: Dict[str, float] = {}
        if baseline_results:
            for b in baseline_results:
                baseline_metrics[b.metric] = b.value

            for metric_name, val in observed_metrics.items():
                if metric_name in baseline_metrics:
                    base_val = baseline_metrics[metric_name]
                    delta = round(float(val) - float(base_val), 4)
                    pct_delta = round((delta / (float(base_val) + 1e-9)) * 100, 2)
                    baseline_comparison[metric_name] = {
                        "proposed_value": val,
                        "baseline_value": base_val,
                        "delta": delta,
                        "percent_delta": pct_delta,
                        "status": "IMPROVED" if delta > 0 else ("DEGRADED" if delta < 0 else "IDENTICAL"),
                    }

        # Build concise statement
        model_name = results[0].model if results else "Proposed Model"
        dataset_name = results[0].dataset if results else "Benchmark Dataset"
        metric_str = ", ".join([f"{k}={v}" for k, v in list(observed_metrics.items())[:3]])

        statement = (
            f"Empirical evaluation of {model_name} on {dataset_name} yielded {metric_str}."
        )
        if baseline_comparison:
            improvements = [k for k, v in baseline_comparison.items() if v["delta"] > 0]
            if improvements:
                statement += f" Observed improvement over baseline in {', '.join(improvements)}."

        # Limitations & uncertainty
        limitations = [
            f"Observed on single benchmark dataset ({dataset_name}); cross-domain validity unverified.",
            "Statistical significance is unconfirmed without multiple randomized seed trials.",
            "Evaluation assumes standardized ground-truth labels and evaluation metrics.",
        ]

        title = f"Candidate Finding: {model_name} on {dataset_name}"

        return ResearchFinding(
            finding_id=generate_id("fnd_cand"),
            workspace_id=self.workspace_id,
            title=title,
            statement=statement,
            finding_type="RESULT",
            evidence_ids=[ev.get("paper_id", "") for ev in (literature_evidence or []) if ev.get("paper_id")],
            experiment_result_ids=result_ids,
            related_hypothesis_id=hypothesis.hypothesis_id if hypothesis else None,
            confidence_level="MEDIUM" if (results and literature_evidence) else "LOW",
            status="PENDING_RESEARCHER_REVIEW",
            question_id=question.question_id if question else None,
            dataset_version=dataset_version,
            model_version=model_version,
            experiment_id=experiment_id,
            run_id=run.run_id,
            mlflow_run_id=run.mlflow_run_id,
            observed_metrics=observed_metrics,
            baseline_comparison=baseline_comparison,
            reproducibility_status=reproducibility_status,
            supporting_evidence=literature_evidence or [],
            limitations=limitations,
            uncertainty_notes=INTEGRITY_DISCLAIMER,
        )
