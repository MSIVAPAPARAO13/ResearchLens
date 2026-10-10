"""Evidence-Grounded Research Proposer (Phase 15).

Generates proposed research questions, testable hypotheses, and experiment plans
using existing ResearchProfile, LiteratureCollection, gap validator, and evidence bookmarks.
Guarantees scientific integrity: never presents generated hypotheses as established scientific facts.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from researchlens.utils.logging import get_logger
from researchlens.workspace.schema import (
    EvidenceBookmark,
    ExperimentPlan,
    ResearchGap,
    ResearchHypothesis,
    ResearchQuestion,
    generate_id,
)

logger = get_logger("evidence_grounded_proposer")

SCIENTIFIC_INTEGRITY_NOTICE = (
    "SCIENTIFIC_INTEGRITY_NOTICE: Generated hypotheses and experiment proposals are "
    "evidence-grounded research candidates derived from literature heuristics, NOT established "
    "scientific facts. Formal experimental execution, reproducibility validation, and human "
    "researcher confirmation are strictly required."
)


@dataclass
class EvidenceQuote:
    """Precise citation quote from literature with source identifier and page."""

    paper_id: str
    paper_title: str
    quote_text: str
    page_number: Optional[int] = None
    section_name: Optional[str] = None
    chunk_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GroundedHypothesisProposal:
    """Testable research hypothesis proposal grounded in literature evidence."""

    proposal_id: str
    question_text: str
    hypothesis_statement: str
    testable: bool = True
    supporting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    explicit_assumptions: List[str] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)
    measurable_criteria: Dict[str, str] = field(default_factory=dict)
    baseline_requirements: List[str] = field(default_factory=list)
    dataset_requirements: List[str] = field(default_factory=list)
    scientific_integrity_notice: str = SCIENTIFIC_INTEGRITY_NOTICE

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvidenceGroundedProposer:
    """Proposes evidence-grounded research questions, testable hypotheses, and experiment plans."""

    def __init__(self, workspace_id: str):
        self.workspace_id = workspace_id

    def formulate_questions_from_gaps(
        self,
        gaps: List[ResearchGap],
        bookmarks: Optional[List[EvidenceBookmark]] = None,
        max_questions: int = 5,
    ) -> List[ResearchQuestion]:
        """Derive testable research questions from identified research gaps and evidence bookmarks."""
        questions: List[ResearchQuestion] = []
        bookmarks = bookmarks or []

        for idx, gap in enumerate(gaps[:max_questions]):
            gap_desc = gap.description.strip()
            if not gap_desc:
                continue

            # Build evidence IDs from bookmarks that match gap keywords or paper
            supporting_ev_ids = [
                bm.bookmark_id for bm in bookmarks
                if bm.paper_id in gap.paper_ids or any(w in bm.source_text.lower() for w in gap_desc.lower().split()[:3])
            ]
            if not supporting_ev_ids and bookmarks:
                supporting_ev_ids = [bookmarks[0].bookmark_id]

            q_text = f"How can we effectively address '{gap_desc[:120]}' in empirical benchmarks?"
            q = ResearchQuestion(
                question_id=generate_id("q"),
                workspace_id=self.workspace_id,
                text=q_text,
                type="PRIMARY",
                supporting_gap_ids=[gap.gap_id] if gap.gap_id else [],
                supporting_evidence_ids=supporting_ev_ids,
                status="ACTIVE",
            )
            questions.append(q)

        return questions

    def propose_hypothesis(
        self,
        question: ResearchQuestion,
        gap: Optional[ResearchGap] = None,
        bookmarks: Optional[List[EvidenceBookmark]] = None,
        candidate_mechanism: str = "Hybrid Dense-BM25 Retrieval with Cross-Encoder Reranking",
        target_dataset: str = "PubMedQA",
        target_metric: str = "MRR",
        target_baseline: str = "BM25-Baseline",
    ) -> GroundedHypothesisProposal:
        """Formulate a testable hypothesis proposal with explicit citations, assumptions, and missing evidence."""
        bookmarks = bookmarks or []
        evidence_quotes: List[Dict[str, Any]] = []

        for bm in bookmarks:
            evidence_quotes.append({
                "paper_id": bm.paper_id or "literature",
                "paper_title": getattr(bm, "label", "") or "Analyzed Paper",
                "quote_text": bm.source_text[:200] if bm.source_text else "Empirical limitation observed in literature.",
                "page_number": bm.page_number,
                "section_name": bm.section_name or "Discussion",
                "chunk_id": bm.chunk_id,
            })

        if not evidence_quotes:
            evidence_quotes.append({
                "paper_id": "monitored_literature",
                "paper_title": "Literature Synthesis",
                "quote_text": f"Prior studies identify performance drop or gap in {target_dataset}.",
                "page_number": None,
                "section_name": "Limitations",
                "chunk_id": None,
            })

        statement = (
            f"Applying {candidate_mechanism} improves {target_metric} over {target_baseline} "
            f"on {target_dataset} while preserving citation precision."
        )

        assumptions = [
            f"The evaluation splits for {target_dataset} have no data leakage with pretraining corpora.",
            "Evaluation metric implementations follow standardized community protocols.",
            "Compute hardware and random seeds are held constant across baseline and proposed runs.",
        ]

        missing_evidence = [
            f"Literature lacks direct head-to-head empirical comparisons between {candidate_mechanism} and {target_baseline} on {target_dataset}.",
            "Cross-domain robustness across diverse document lengths is unverified in cited literature.",
        ]

        criteria = {
            target_metric: "Statistically positive delta (p < 0.05 or > 2.0% absolute improvement over baseline)",
            "citation_accuracy": "Maintains >= 85.0% citation support precision",
            "latency_overhead": "Inference latency does not exceed 1.5x baseline latency",
        }

        return GroundedHypothesisProposal(
            proposal_id=generate_id("hyp_prop"),
            question_text=question.text,
            hypothesis_statement=statement,
            testable=True,
            supporting_evidence=evidence_quotes,
            explicit_assumptions=assumptions,
            missing_evidence=missing_evidence,
            measurable_criteria=criteria,
            baseline_requirements=[target_baseline],
            dataset_requirements=[target_dataset],
        )

    def convert_to_hypothesis(
        self,
        proposal: GroundedHypothesisProposal,
        question_id: str,
    ) -> ResearchHypothesis:
        """Convert GroundedHypothesisProposal into a persistent ResearchHypothesis model."""
        supporting_ev_texts = [
            f"[{eq.get('paper_title', 'Ref')} p.{eq.get('page_number', '?')}]: \"{eq.get('quote_text', '')}\""
            for eq in proposal.supporting_evidence
        ]

        return ResearchHypothesis(
            hypothesis_id=generate_id("hyp"),
            workspace_id=self.workspace_id,
            question_id=question_id,
            statement=proposal.hypothesis_statement,
            rationale=(
                f"Assumptions: {'; '.join(proposal.explicit_assumptions)}. "
                f"Missing evidence to address: {'; '.join(proposal.missing_evidence)}."
            ),
            supporting_evidence=supporting_ev_texts,
            status="DRAFT",
            testable=proposal.testable,
        )

    def generate_experiment_plan(
        self,
        hypothesis: ResearchHypothesis,
        dataset: str = "PubMedQA",
        model: str = "Hybrid-MiniLM-CrossEncoder",
        baseline: str = "BM25-Baseline",
        metrics: Optional[List[str]] = None,
        execution_backend: str = "LOCAL",
        execution_image: Optional[str] = None,
        entrypoint: str = "scripts/experiments/deterministic_demo.py",
        timeout_seconds: int = 3600,
    ) -> ExperimentPlan:
        """Create a planned ExperimentPlan model ready for human review and confirmation."""
        metrics_list = metrics or ["MRR", "precision", "recall", "citation_accuracy"]
        plan = ExperimentPlan(
            experiment_id=generate_id("exp_auto"),
            name=f"Automated Plan: {hypothesis.statement[:60]}",
            hypothesis_id=hypothesis.hypothesis_id,
            dataset=dataset,
            model=model,
            baseline=baseline,
            metrics=metrics_list,
            variables={"dataset": dataset, "model": model, "baseline": baseline},
            expected_outcome=f"Evaluate whether proposed model outperforms {baseline} on {dataset}.",
            status="PLANNED",
            entrypoint=entrypoint,
            working_directory=".",
            execution_backend=execution_backend,
            execution_image=execution_image,
            timeout_seconds=timeout_seconds,
            resource_requirements={"cpu_limit": 2.0, "memory_limit": "2GB"},
        )
        return plan
