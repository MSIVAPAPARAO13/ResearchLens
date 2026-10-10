"""Research Gap Intelligence Report aggregator."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from researchlens.analysis.gap_engine import ResearchGapEngine
from researchlens.analysis.research_gap_schema import ResearchGap, ResearchHypothesis
from researchlens.extraction.schema import ExtractedItem, ResearchProfile


@dataclass
class ResearchGapReport:
    """Dataclass holding structured gap intelligence report findings."""

    paper_title: str
    executive_summary: str
    top_gaps: List[ResearchGap] = field(default_factory=list)
    all_gaps: List[ResearchGap] = field(default_factory=list)
    explicit_limitations: List[ExtractedItem] = field(default_factory=list)
    explicit_future_work: List[ExtractedItem] = field(default_factory=list)
    potential_hypotheses: List[ResearchHypothesis] = field(default_factory=list)
    ablation_mode: str = "full"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ResearchGapReport to a JSON-compatible dictionary."""
        return {
            "paper_title": self.paper_title,
            "executive_summary": self.executive_summary,
            "top_gaps": [g.to_dict() for g in self.top_gaps],
            "all_gaps": [g.to_dict() for g in self.all_gaps],
            "explicit_limitations": [
                lim.to_dict() if hasattr(lim, "to_dict") else str(lim) for lim in self.explicit_limitations
            ],
            "explicit_future_work": [
                fw.to_dict() if hasattr(fw, "to_dict") else str(fw) for fw in self.explicit_future_work
            ],
            "potential_hypotheses": [h.to_dict() for h in self.potential_hypotheses],
            "ablation_mode": self.ablation_mode,
        }

    def to_markdown(self) -> str:
        """Render ResearchGapReport into clean GitHub Flavored Markdown."""
        lines = [
            f"# 💡 Research Gap Intelligence Report: {self.paper_title}",
            "",
            "## 📌 Executive Summary",
            self.executive_summary,
            "",
            "---",
            "",
            "## 🎯 Top Identified Research Gaps",
        ]

        if not self.top_gaps or self.top_gaps[0].gap_type == "REFRUSAL":
            lines.append("*Insufficient evidence to identify reliable research gaps.*")
        else:
            for idx, g in enumerate(self.top_gaps, 1):
                lines.extend([
                    f"### Gap #{idx}: {g.title}",
                    f"- **Category:** `{g.category}`",
                    f"- **Type:** `{g.gap_type}` ({g.explicit_or_inferred})",
                    f"- **Confidence:** `{g.confidence_level}` ({g.confidence_score:.2f})",
                    f"- **Statement:** {g.statement}",
                    f"- **Rationale:** {g.rationale}",
                    f"- **Potential Direction:** {g.potential_direction}",
                    "",
                ])

        lines.extend(["---", "", "## 🧪 Potential Hypotheses"])
        if not self.potential_hypotheses:
            lines.append("*No hypotheses generated based on available evidence.*")
        else:
            for h in self.potential_hypotheses:
                lines.extend([
                    f"### ⚠️ {h.label}: {h.title}",
                    f"- **Statement:** {h.statement}",
                    f"- **Motivation:** {h.motivation}",
                    f"- **Suggested Evaluation:** {h.suggested_evaluation}",
                    "",
                ])

        return "\n".join(lines)


def generate_research_gap_report(
    profile: ResearchProfile, ablation_mode: str = "full"
) -> ResearchGapReport:
    """Generate a complete ResearchGapReport from a ResearchProfile."""
    engine = ResearchGapEngine(ablation_mode=ablation_mode)
    gaps, hypotheses = engine.analyze_profile(profile)

    # Build executive summary
    valid_gaps = [g for g in gaps if g.gap_type != "REFRUSAL"]
    if not valid_gaps:
        summary = "Insufficient explicit or inferred evidence was detected in the paper to formulate high-confidence research gaps."
    else:
        categories = list(set(g.category for g in valid_gaps))
        summary = (
            f"Analysis of '{profile.title}' identified {len(valid_gaps)} potential research gaps across "
            f"the following categories: {', '.join(categories)}. "
            f"Top priority items focus on {valid_gaps[0].category.lower()} and {valid_gaps[0].potential_direction.lower()}."
        )

    return ResearchGapReport(
        paper_title=profile.title or "Untitled Research Paper",
        executive_summary=summary,
        top_gaps=valid_gaps[:5],
        all_gaps=gaps,
        explicit_limitations=profile.limitations,
        explicit_future_work=profile.future_work,
        potential_hypotheses=hypotheses,
        ablation_mode=ablation_mode,
    )
