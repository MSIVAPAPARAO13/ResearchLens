"""Research Gap and Hypothesis schema definitions with provenance tracking."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from researchlens.extraction.schema import ExtractedItem


@dataclass
class ResearchGap:
    """Dataclass representing an evidence-grounded research gap."""

    gap_id: str
    title: str
    category: str  # DATASET, GENERALIZATION, METHODOLOGY, EVALUATION, BASELINE, REPRODUCIBILITY, ROBUSTNESS, EFFICIENCY, FAIRNESS, DOMAIN_COVERAGE, TEMPORAL_COVERAGE, COMPARISON, THEORETICAL, APPLICATION, OTHER
    gap_type: str  # EXPLICIT_LIMITATION, EXPLICIT_FUTURE_WORK, MISSING_EVALUATION, MISSING_COMPARISON, CONFLICTING_RESULT, UNDEREXPLORED_DATASET, GENERALIZATION_GAP, REPRODUCIBILITY_GAP
    statement: str
    supporting_evidence: List[ExtractedItem] = field(default_factory=list)
    source_pages: List[int] = field(default_factory=list)
    source_sections: List[str] = field(default_factory=list)
    source_chunks: List[str] = field(default_factory=list)
    explicit_or_inferred: str = "INFERRED"  # "EXPLICIT" or "INFERRED"
    confidence_level: str = "Medium"  # "High", "Medium", "Low"
    confidence_score: float = 0.5
    rationale: str = ""
    potential_direction: str = ""
    affected_method: str = ""
    affected_dataset: str = ""
    affected_metric: str = ""
    severity: str = "Medium"  # "High", "Medium", "Low"
    # Multi-paper preparation fields (Phase 6 readiness)
    supporting_documents: List[str] = field(default_factory=list)
    cross_paper_frequency: int = 1
    cross_paper_evidence: List[Dict[str, Any]] = field(default_factory=list)
    conflicting_papers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ResearchGap to a JSON-compatible dictionary."""
        return {
            "gap_id": self.gap_id,
            "title": self.title,
            "category": self.category,
            "gap_type": self.gap_type,
            "statement": self.statement,
            "supporting_evidence": [
                ev.to_dict() if hasattr(ev, "to_dict") else str(ev) for ev in self.supporting_evidence
            ],
            "source_pages": self.source_pages,
            "source_sections": self.source_sections,
            "source_chunks": self.source_chunks,
            "explicit_or_inferred": self.explicit_or_inferred,
            "confidence_level": self.confidence_level,
            "confidence_score": round(self.confidence_score, 4),
            "rationale": self.rationale,
            "potential_direction": self.potential_direction,
            "affected_method": self.affected_method,
            "affected_dataset": self.affected_dataset,
            "affected_metric": self.affected_metric,
            "severity": self.severity,
            "supporting_documents": self.supporting_documents,
            "cross_paper_frequency": self.cross_paper_frequency,
            "cross_paper_evidence": self.cross_paper_evidence,
            "conflicting_papers": self.conflicting_papers,
        }


@dataclass
class ResearchHypothesis:
    """Dataclass representing an evidence-derived potential research hypothesis."""

    hypothesis_id: str
    title: str
    statement: str
    motivation: str
    supporting_evidence: List[ExtractedItem] = field(default_factory=list)
    variables: Dict[str, str] = field(default_factory=dict)
    suggested_evaluation: str = ""
    confidence_level: str = "Medium"
    label: str = "Potential Hypothesis"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ResearchHypothesis to a JSON-compatible dictionary."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "title": self.title,
            "statement": self.statement,
            "motivation": self.motivation,
            "supporting_evidence": [
                ev.to_dict() if hasattr(ev, "to_dict") else str(ev) for ev in self.supporting_evidence
            ],
            "variables": self.variables,
            "suggested_evaluation": self.suggested_evaluation,
            "confidence_level": self.confidence_level,
            "label": self.label,
        }
