"""Structured scientific research profile and provenance schema definitions."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExtractedItem:
    """An individual extracted scientific fact with evidence provenance metadata."""
    value: str
    source_text: str = ""
    page_number: Optional[int] = None
    section_name: str = "unknown"
    chunk_id: Optional[str] = None
    confidence: float = 1.0
    confidence_level: str = "Medium"  # "High", "Medium", "Low"
    extraction_method: str = "pattern"  # e.g., "pattern", "ner", "heading", "tfidf", "heuristic"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "source_text": self.source_text,
            "page_number": self.page_number,
            "section_name": self.section_name,
            "chunk_id": self.chunk_id,
            "confidence": round(self.confidence, 4),
            "confidence_level": self.confidence_level,
            "extraction_method": self.extraction_method,
        }


@dataclass
class QuantitativeResultItem:
    """Extracted quantitative result with metric name, numerical value, model, and dataset."""
    metric: str
    value: str
    unit: Optional[str] = None
    model: Optional[str] = None
    dataset: Optional[str] = None
    source_text: str = ""
    page_number: Optional[int] = None
    section_name: str = "unknown"
    chunk_id: Optional[str] = None
    confidence: float = 1.0
    extraction_method: str = "pattern"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "model": self.model,
            "dataset": self.dataset,
            "source_text": self.source_text,
            "page_number": self.page_number,
            "section_name": self.section_name,
            "chunk_id": self.chunk_id,
            "confidence": round(self.confidence, 4),
            "extraction_method": self.extraction_method,
        }


@dataclass
class BaselineComparison:
    """Comparative baseline performance finding."""
    baseline_model: str
    baseline_metric: Optional[str] = None
    baseline_value: Optional[str] = None
    proposed_model: Optional[str] = None
    proposed_value: Optional[str] = None
    difference: Optional[str] = None
    source_text: str = ""
    page_number: Optional[int] = None
    section_name: str = "unknown"
    chunk_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_model": self.baseline_model,
            "baseline_metric": self.baseline_metric,
            "baseline_value": self.baseline_value,
            "proposed_model": self.proposed_model,
            "proposed_value": self.proposed_value,
            "difference": self.difference,
            "source_text": self.source_text,
            "page_number": self.page_number,
            "section_name": self.section_name,
            "chunk_id": self.chunk_id,
        }


@dataclass
class ResearchProfile:
    """Structured knowledge representation of a scientific research paper."""
    document_id: str = ""
    title: str = "Not detected"
    authors: List[ExtractedItem] = field(default_factory=list)
    organizations: List[ExtractedItem] = field(default_factory=list)
    year: Optional[str] = None
    keywords: List[ExtractedItem] = field(default_factory=list)

    problem: List[ExtractedItem] = field(default_factory=list)
    objective: List[ExtractedItem] = field(default_factory=list)
    research_questions: List[ExtractedItem] = field(default_factory=list)
    hypotheses: List[ExtractedItem] = field(default_factory=list)

    methods: List[ExtractedItem] = field(default_factory=list)
    models: List[ExtractedItem] = field(default_factory=list)
    algorithms: List[ExtractedItem] = field(default_factory=list)

    datasets: List[ExtractedItem] = field(default_factory=list)
    experimental_setup: List[ExtractedItem] = field(default_factory=list)

    metrics: List[ExtractedItem] = field(default_factory=list)
    results: List[QuantitativeResultItem] = field(default_factory=list)
    baselines: List[BaselineComparison] = field(default_factory=list)

    contributions: List[ExtractedItem] = field(default_factory=list)
    limitations: List[ExtractedItem] = field(default_factory=list)
    future_work: List[ExtractedItem] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ResearchProfile into a structured dictionary."""
        return {
            "document_id": self.document_id,
            "title": self.title,
            "authors": [a.to_dict() for a in self.authors],
            "organizations": [o.to_dict() for o in self.organizations],
            "year": self.year or "Not detected",
            "keywords": [k.to_dict() for k in self.keywords],
            "problem": [p.to_dict() for p in self.problem],
            "objective": [o.to_dict() for o in self.objective],
            "research_questions": [rq.to_dict() for rq in self.research_questions],
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "methods": [m.to_dict() for m in self.methods],
            "models": [m.to_dict() for m in self.models],
            "algorithms": [alg.to_dict() for alg in self.algorithms],
            "datasets": [d.to_dict() for d in self.datasets],
            "experimental_setup": [e.to_dict() for e in self.experimental_setup],
            "metrics": [m.to_dict() for m in self.metrics],
            "results": [r.to_dict() for r in self.results],
            "baselines": [b.to_dict() for b in self.baselines],
            "contributions": [c.to_dict() for c in self.contributions],
            "limitations": [lim.to_dict() for lim in self.limitations],
            "future_work": [f.to_dict() for f in self.future_work],
        }
