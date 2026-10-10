"""Cross-paper conflict detection module for identifying empirical inconsistency."""

from dataclasses import dataclass
from typing import Any, Dict, List

from researchlens.analysis.comparability import check_scientific_comparability
from researchlens.analysis.literature_collection import LiteratureCollection


@dataclass
class CrossPaperConflict:
    """Dataclass representing a detected empirical contradiction or evaluation discrepancy."""

    conflict_id: str
    conflict_type: str  # RESULT_CONFLICT, BASELINE_CONFLICT, METHOD_CONFLICT, DATASET_DIFFERENCE, EVALUATION_DIFFERENCE
    model: str
    dataset: str
    metric: str
    paper_a_title: str
    value_a: str
    evidence_a: str
    paper_b_title: str
    value_b: str
    evidence_b: str
    comparison_reason: str
    comparability_status: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize CrossPaperConflict to a JSON-compatible dictionary."""
        return {
            "conflict_id": self.conflict_id,
            "conflict_type": self.conflict_type,
            "model": self.model,
            "dataset": self.dataset,
            "metric": self.metric,
            "paper_a_title": self.paper_a_title,
            "value_a": self.value_a,
            "evidence_a": self.evidence_a,
            "paper_b_title": self.paper_b_title,
            "value_b": self.value_b,
            "evidence_b": self.evidence_b,
            "comparison_reason": self.comparison_reason,
            "comparability_status": self.comparability_status,
        }


def detect_cross_paper_conflicts(collection: LiteratureCollection) -> List[CrossPaperConflict]:
    """Detect potential conflicting quantitative results or baseline assertions across papers."""
    conflicts: List[CrossPaperConflict] = []
    idx = 1

    docs = collection.documents
    for i in range(len(docs)):
        doc_a = docs[i]
        if not doc_a.profile:
            continue

        for res_a in doc_a.profile.results:
            for j in range(i + 1, len(docs)):
                doc_b = docs[j]
                if not doc_b.profile:
                    continue

                for res_b in doc_b.profile.results:
                    # Check model similarity
                    mod_a = (res_a.model or "model").lower()
                    mod_b = (res_b.model or "model").lower()

                    if mod_a == mod_b or mod_a in mod_b or mod_b in mod_a:
                        is_comp, reason = check_scientific_comparability(
                            metric_a=res_a.metric,
                            dataset_a=res_a.dataset or "Unknown",
                            metric_b=res_b.metric,
                            dataset_b=res_b.dataset or "Unknown",
                        )

                        if is_comp and res_a.value != res_b.value:
                            # Flag potential conflicting result
                            c = CrossPaperConflict(
                                conflict_id=f"conflict_{idx}",
                                conflict_type="RESULT_CONFLICT",
                                model=res_a.model or mod_a.title(),
                                dataset=res_a.dataset or "Shared Dataset",
                                metric=res_a.metric,
                                paper_a_title=doc_a.title,
                                value_a=res_a.value,
                                evidence_a=res_a.source_text or f"{res_a.metric}={res_a.value}",
                                paper_b_title=doc_b.title,
                                value_b=res_b.value,
                                evidence_b=res_b.source_text or f"{res_b.metric}={res_b.value}",
                                comparison_reason=f"Both studies report metric '{res_a.metric}' on '{res_a.dataset}' for '{res_a.model}', but report differing values ({res_a.value} vs. {res_b.value}).",
                                comparability_status="Comparable",
                            )
                            conflicts.append(c)
                            idx += 1
                        elif not is_comp and res_a.metric.lower() == res_b.metric.lower() and res_a.value != res_b.value:
                            # Flag evaluation difference
                            c = CrossPaperConflict(
                                conflict_id=f"conflict_{idx}",
                                conflict_type="EVALUATION_DIFFERENCE",
                                model=res_a.model or mod_a.title(),
                                dataset=f"{res_a.dataset} vs {res_b.dataset}",
                                metric=res_a.metric,
                                paper_a_title=doc_a.title,
                                value_a=res_a.value,
                                evidence_a=res_a.source_text or f"{res_a.metric}={res_a.value}",
                                paper_b_title=doc_b.title,
                                value_b=res_b.value,
                                evidence_b=res_b.source_text or f"{res_b.metric}={res_b.value}",
                                comparison_reason=f"Results differ ({res_a.value} vs. {res_b.value}), but evaluation setups or datasets differ ({res_a.dataset} vs {res_b.dataset}). Not directly comparable.",
                                comparability_status="Not directly comparable",
                            )
                            conflicts.append(c)
                            idx += 1

    return conflicts
