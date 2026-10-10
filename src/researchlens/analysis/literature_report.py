"""Literature Intelligence Report generator and multi-format exporter."""

import csv
import io
from dataclasses import dataclass, field
from typing import Any, Dict, List

from researchlens.analysis.conflict_detector import CrossPaperConflict, detect_cross_paper_conflicts
from researchlens.analysis.cross_paper_gap_engine import (
    CrossPaperResearchGapEngine,
    LiteratureResearchGap,
)
from researchlens.analysis.literature_collection import LiteratureCollection
from researchlens.analysis.paper_matrix import (
    PaperComparisonMatrix,
    generate_dataset_comparison_table,
    generate_method_comparison_table,
    generate_paper_matrix,
    generate_result_comparison_table,
)
from researchlens.analysis.semantic_clustering import (
    LimitationCluster,
    cluster_limitations_and_future_work,
)


@dataclass
class LiteratureIntelligenceReport:
    """Dataclass aggregating multi-paper literature intelligence findings."""

    collection_name: str
    paper_count: int
    executive_summary: str
    paper_matrix: PaperComparisonMatrix = field(default_factory=PaperComparisonMatrix)
    method_table: List[Dict[str, Any]] = field(default_factory=list)
    dataset_table: List[Dict[str, Any]] = field(default_factory=list)
    result_table: List[Dict[str, Any]] = field(default_factory=list)
    recurring_limitations: List[LimitationCluster] = field(default_factory=list)
    recurring_future_work: List[LimitationCluster] = field(default_factory=list)
    conflicts: List[CrossPaperConflict] = field(default_factory=list)
    literature_gaps: List[LiteratureResearchGap] = field(default_factory=list)
    ablation_mode: str = "full_evidence"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize report to a JSON-compatible dictionary."""
        return {
            "collection_name": self.collection_name,
            "paper_count": self.paper_count,
            "executive_summary": self.executive_summary,
            "paper_matrix": self.paper_matrix.to_table_dict(),
            "method_table": self.method_table,
            "dataset_table": self.dataset_table,
            "result_table": self.result_table,
            "recurring_limitations": [c.to_dict() for c in self.recurring_limitations],
            "recurring_future_work": [c.to_dict() for c in self.recurring_future_work],
            "conflicts": [conf.to_dict() for conf in self.conflicts],
            "literature_gaps": [g.to_dict() for g in self.literature_gaps],
            "ablation_mode": self.ablation_mode,
        }

    def to_matrix_csv(self) -> str:
        """Export PaperComparisonMatrix as CSV string."""
        matrix_dict = self.paper_matrix.to_table_dict()
        if not matrix_dict:
            return ""

        output = io.StringIO()
        headers = list(matrix_dict.keys())
        writer = csv.writer(output)
        writer.writerow(headers)

        num_rows = len(next(iter(matrix_dict.values())))
        for i in range(num_rows):
            row = [matrix_dict[h][i] for h in headers]
            writer.writerow(row)

        return output.getvalue()

    def to_markdown(self) -> str:
        """Render LiteratureIntelligenceReport into GitHub Flavored Markdown."""
        lines = [
            f"# 📚 Literature Intelligence Report: {self.collection_name}",
            "",
            f"**Total Papers Analyzed:** {self.paper_count}",
            "",
            "## 📌 Executive Summary",
            self.executive_summary,
            "",
            "---",
            "",
            "## 🎯 Literature-Level Research Gaps",
        ]

        if not self.literature_gaps or self.literature_gaps[0].gap_id == "lit_gap_refusal":
            lines.append("*Insufficient evidence to identify reliable literature-level research gaps.*")
        else:
            for idx, g in enumerate(self.literature_gaps[:5], 1):
                lines.extend([
                    f"### Gap #{idx}: {g.title}",
                    f"- **Category:** `{g.category}`",
                    f"- **Confidence:** `{g.confidence_level}` ({g.confidence_score:.2f})",
                    f"- **Recurrence:** Supported by {g.recurrence_count} papers ({', '.join(g.supporting_papers)})",
                    f"- **Statement:** {g.statement}",
                    f"- **Rationale:** {g.rationale}",
                    f"- **Potential Direction:** {g.potential_direction}",
                    f"- **Independence Status:** {g.independence_status}",
                    "",
                ])

        lines.extend(["---", "", "## 🔄 Recurring Limitation Themes"])
        if not self.recurring_limitations:
            lines.append("*No recurring limitation themes detected.*")
        else:
            for c in self.recurring_limitations[:5]:
                lines.extend([
                    f"### Theme: {c.theme_name}",
                    f"- **Category:** `{c.category}` | **Supporting Papers ({c.recurrence_count}):** {', '.join(c.supporting_papers)}",
                    "",
                ])

        lines.extend(["---", "", "## ⚡ Conflicting Findings & Discrepancies"])
        if not self.conflicts:
            lines.append("*No empirical result conflicts detected across analyzed papers.*")
        else:
            for conf in self.conflicts[:5]:
                lines.extend([
                    f"### Potential Conflict: {conf.model} ({conf.metric})",
                    f"- **Paper A ({conf.paper_a_title}):** `{conf.value_a}`",
                    f"- **Paper B ({conf.paper_b_title}):** `{conf.value_b}`",
                    f"- **Comparability:** `{conf.comparability_status}`",
                    f"- **Reason:** {conf.comparison_reason}",
                    "",
                ])

        return "\n".join(lines)


def generate_literature_report(
    collection: LiteratureCollection, ablation_mode: str = "full_evidence"
) -> LiteratureIntelligenceReport:
    """Generate a comprehensive LiteratureIntelligenceReport from a LiteratureCollection."""
    matrix = generate_paper_matrix(collection)
    method_table = generate_method_comparison_table(collection)
    dataset_table = generate_dataset_comparison_table(collection)
    result_table = generate_result_comparison_table(collection)

    lim_clusters, fw_clusters = cluster_limitations_and_future_work(
        collection=collection, ablation_mode=ablation_mode
    )

    conflicts = detect_cross_paper_conflicts(collection)

    gap_engine = CrossPaperResearchGapEngine(ablation_mode=ablation_mode)
    lit_gaps = gap_engine.analyze_collection(collection)

    # Executive Summary Construction
    valid_gaps = [g for g in lit_gaps if g.gap_id != "lit_gap_refusal"]
    num_lim_themes = len(lim_clusters)
    num_conflicts = len(conflicts)

    summary = (
        f"Literature intelligence analysis of {collection.paper_count} research papers identified "
        f"{len(method_table)} unique methods, {len(dataset_table)} benchmark datasets, {num_lim_themes} recurring limitation themes, "
        f"{num_conflicts} potential empirical conflicts, and {len(valid_gaps)} literature-level research gaps. "
    )
    if valid_gaps:
        summary += f"Primary cross-paper priority is '{valid_gaps[0].title}' supported across {valid_gaps[0].recurrence_count} independent papers."

    return LiteratureIntelligenceReport(
        collection_name=collection.collection_name,
        paper_count=collection.paper_count,
        executive_summary=summary,
        paper_matrix=matrix,
        method_table=method_table,
        dataset_table=dataset_table,
        result_table=result_table,
        recurring_limitations=lim_clusters,
        recurring_future_work=fw_clusters,
        conflicts=conflicts,
        literature_gaps=lit_gaps,
        ablation_mode=ablation_mode,
    )
