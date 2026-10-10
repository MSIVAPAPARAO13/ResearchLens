"""Structured Monitoring Report generator and multi-format exporters (Markdown, JSON, CSV)."""

import csv
import io
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from researchlens.analysis.external_provider import ExternalPaper
from researchlens.analysis.gap_evolution import ResearchGapEvolutionRecord
from researchlens.analysis.research_topic import MonitoringRun, ResearchTopic
from researchlens.analysis.trend_analysis import EmergingEntity
from researchlens.utils.logging import get_logger

logger = get_logger("monitoring_report")


@dataclass
class MonitoringReport:
    """Comprehensive literature monitoring intelligence report over tracked topics."""

    topic_name: str
    topic_query: str
    providers: List[str]
    monitoring_period: str
    total_papers_tracked: int
    new_papers: List[ExternalPaper] = field(default_factory=list)
    new_methods: List[EmergingEntity] = field(default_factory=list)
    new_datasets: List[EmergingEntity] = field(default_factory=list)
    new_metrics: List[EmergingEntity] = field(default_factory=list)
    persistent_themes: List[str] = field(default_factory=list)
    gap_evolution: List[ResearchGapEvolutionRecord] = field(default_factory=list)
    potentially_addressed_gaps: List[ResearchGapEvolutionRecord] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_name": self.topic_name,
            "topic_query": self.topic_query,
            "providers": self.providers,
            "monitoring_period": self.monitoring_period,
            "total_papers_tracked": self.total_papers_tracked,
            "new_papers": [p.to_dict() if hasattr(p, "to_dict") else p for p in self.new_papers],
            "new_methods": [m.to_dict() for m in self.new_methods],
            "new_datasets": [d.to_dict() for d in self.new_datasets],
            "new_metrics": [m.to_dict() for m in self.new_metrics],
            "persistent_themes": self.persistent_themes,
            "gap_evolution": [g.to_dict() for g in self.gap_evolution],
            "potentially_addressed_gaps": [g.to_dict() for g in self.potentially_addressed_gaps],
            "generated_at": self.generated_at,
        }

    def to_json(self) -> str:
        """Export report as formatted JSON."""
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        """Render publication-grade scientific monitoring report."""
        lines = [
            f"# Literature Monitoring Report: {self.topic_name}",
            "",
            f"**Generated**: {self.generated_at} | **Query**: `{self.topic_query}`",
            f"**Providers**: {', '.join(self.providers)} | **Tracked Papers**: {self.total_papers_tracked}",
            "",
            "> *Scientific Notice: Findings in this report represent observations derived strictly "
            "from literature retrieved and analyzed by ResearchLens across monitored scholarly providers. "
            "They do not constitute exhaustive claims regarding the entire scientific community.*",
            "",
            "---",
            "## 1. Executive Summary & New Papers",
            f"Among the literature retrieved by ResearchLens within the monitored corpus, **{len(self.new_papers)} new publications** were identified in the latest monitoring cycle.",
            "",
        ]

        if self.new_papers:
            lines.append("| Title | Year | Authors | Provider | Citations |")
            lines.append("| :--- | :---: | :--- | :---: | :---: |")
            for p in self.new_papers[:10]:
                auth = ", ".join(p.authors[:2]) + (" et al." if len(p.authors) > 2 else "")
                lines.append(f"| {p.title} | {p.year or 'N/A'} | {auth} | {p.provider} | {p.citation_count} |")
            lines.append("")
        else:
            lines.append("_No new publications observed in this cycle._\n")

        lines.extend([
            "## 2. Emerging Scientific Methodology & Benchmarks",
            "### Newly Observed Models & Architectures",
        ])
        if self.new_methods:
            for m in self.new_methods:
                lines.append(f"- **{m.name}**: First observed on {m.first_observed[:10]} in *{m.supporting_papers[0] if m.supporting_papers else 'literature'}*")
        else:
            lines.append("_No novel model variants detected._")

        lines.append("\n### Newly Observed Benchmark Datasets")
        if self.new_datasets:
            for d in self.new_datasets:
                lines.append(f"- **{d.name}**: First cited in *{d.supporting_papers[0] if d.supporting_papers else 'literature'}*")
        else:
            lines.append("_No newly introduced datasets detected._")

        lines.extend([
            "",
            "## 3. Research Gap Evolution & Validation Trajectory",
            "Longitudinal analysis of open limitations and external resolution signals:",
            "",
        ])

        if self.gap_evolution:
            lines.append("| Gap Description | Status | Evidence Count | Recent Supporting Literature |")
            lines.append("| :--- | :---: | :---: | :--- |")
            for g in self.gap_evolution[:8]:
                sup = ", ".join(g.supporting_papers[:2]) if g.supporting_papers else "Initial run"
                lines.append(f"| {g.gap_statement} | **{g.status}** | {g.evidence_count} | {sup} |")
            lines.append("")
        else:
            lines.append("_No persistent gaps tracked._\n")

        lines.extend([
            "## 4. Potentially Addressed Gaps",
        ])
        if self.potentially_addressed_gaps:
            for ag in self.potentially_addressed_gaps:
                lines.append(f"- **{ag.gap_statement}**")
                lines.append(f"  - *Validation*: {ag.validation_status} ({ag.validation_rationale or 'Candidate solutions identified.'})")
                if ag.addressed_by_papers:
                    lines.append(f"  - *Candidate Solutions in*: {', '.join(ag.addressed_by_papers)}")
        else:
            lines.append("_No verified gap resolutions observed in retrieved literature._")

        lines.extend([
            "",
            "## 5. Limitations & Caveats",
            "- Monitoring coverage is constrained to query parameters and participating open-access API indices.",
            "- Citation dynamics reflect provider ingestion latencies and do not imply definitive field consensus.",
        ])

        return "\n".join(lines)

    def to_csv(self) -> str:
        """Export tracked papers list as CSV string."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["paper_id", "title", "authors", "year", "provider", "doi", "arxiv_id", "citation_count", "url"])

        for p in self.new_papers:
            writer.writerow([
                p.paper_id,
                p.title,
                "; ".join(p.authors),
                p.year or "",
                p.provider,
                p.doi or "",
                p.arxiv_id or "",
                p.citation_count,
                getattr(p, "url", getattr(p, "pdf_url", "")) or "",
            ])

        return output.getvalue()


def generate_monitoring_report(
    topic: ResearchTopic,
    latest_run: Optional[MonitoringRun] = None,
    gap_history: Optional[Dict[str, ResearchGapEvolutionRecord]] = None,
    emerging_methods: Optional[List[EmergingEntity]] = None,
    emerging_datasets: Optional[List[EmergingEntity]] = None,
) -> MonitoringReport:
    """Synthesize a complete MonitoringReport instance from topic state and run results."""
    run = latest_run or (topic.runs[-1] if topic.runs else None)
    run_papers = run.papers if run else []
    new_paper_ids = set(run.new_paper_ids) if run else set()

    new_papers = [p for p in run_papers if p.paper_id in new_paper_ids]

    gaps = list(gap_history.values()) if gap_history else []
    addressed = [g for g in gaps if g.status == "POTENTIALLY_ADDRESSED"]

    period = "All Historical Runs"
    if topic.runs:
        start = topic.runs[0].started_at[:10]
        end = topic.runs[-1].started_at[:10]
        period = f"{start} to {end}"

    return MonitoringReport(
        topic_name=topic.name,
        topic_query=topic.query,
        providers=topic.providers,
        monitoring_period=period,
        total_papers_tracked=len(topic.history),
        new_papers=new_papers,
        new_methods=emerging_methods or [],
        new_datasets=emerging_datasets or [],
        gap_evolution=gaps,
        potentially_addressed_gaps=addressed,
    )
