"""Research monitoring service, paper change detection, alert dispatching, and run execution."""

import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from researchlens.analysis.external_provider import (
    ExternalPaper,
    UnifiedLiteratureSearchService,
    convert_external_paper_to_document,
    deduplicate_papers,
    normalize_title,
    rank_papers_by_relevance,
)
from researchlens.analysis.gap_evolution import GapEvolutionEngine, ResearchGapEvolutionRecord
from researchlens.analysis.research_gap_schema import ResearchGap
from researchlens.analysis.research_topic import (
    MonitoringRun,
    PaperHistoryEntry,
    ResearchTopic,
    TopicStore,
)
from researchlens.analysis.trend_analysis import EmergingEntity, TrendAnalyzer
from researchlens.extraction.pipeline import extract_research_profile
from researchlens.extraction.schema import ResearchProfile
from researchlens.utils.logging import get_logger

logger = get_logger("monitoring_service")


@dataclass
class ResearchAlert:
    """Actionable alert generated when significant literature changes are detected."""

    alert_id: str
    topic_id: str
    type: str  # NEW_PAPER, NEW_METHOD, NEW_DATASET, NEW_RESEARCH_GAP, GAP_STATUS_CHANGED, POTENTIALLY_ADDRESSED_GAP, SIGNIFICANT_LITERATURE_CHANGE
    severity: str  # INFO, IMPORTANT, HIGH
    title: str
    message: str
    evidence: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    read: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchAlert":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class NotificationProvider(ABC):
    """Abstract interface for dispatching scientific monitoring notifications."""

    @abstractmethod
    def send_alert(self, alert: ResearchAlert) -> bool:
        """Deliver alert to user or destination channel."""
        pass


class ConsoleNotificationProvider(NotificationProvider):
    """Default logger-based notification provider."""

    def send_alert(self, alert: ResearchAlert) -> bool:
        logger.info(f"[{alert.severity}] ResearchAlert: {alert.title} - {alert.message}")
        return True


class MonitoringScheduler(ABC):
    """Foundation interface for automated periodic monitoring schedules."""

    @abstractmethod
    def trigger_manual_run(self, topic: ResearchTopic) -> MonitoringRun:
        """Execute a manual immediate monitoring run."""
        pass


def _resolve_paper_stable_key(paper: ExternalPaper) -> str:
    """Return the strongest unique identifier for a paper (DOI > arXiv > provider_id > normalized_title)."""
    if paper.doi:
        return f"doi:{paper.doi.lower().strip()}"
    if paper.arxiv_id:
        return f"arxiv:{paper.arxiv_id.lower().strip()}"
    if paper.paper_id and paper.provider and paper.provider != "Unknown":
        return f"{paper.provider}:{paper.paper_id.strip()}"
    return f"title:{normalize_title(paper.title)}"


def detect_paper_changes(
    current_papers: List[ExternalPaper],
    existing_history: Dict[str, PaperHistoryEntry],
) -> Tuple[List[ExternalPaper], List[Tuple[ExternalPaper, Dict[str, Any]]], List[ExternalPaper]]:
    """Compare freshly retrieved papers against historical records.

    Returns:
        (new_papers, updated_papers_with_diffs, existing_unmodified_papers)
    """
    new_papers: List[ExternalPaper] = []
    updated_papers: List[Tuple[ExternalPaper, Dict[str, Any]]] = []
    existing_unmodified: List[ExternalPaper] = []

    for paper in current_papers:
        key = _resolve_paper_stable_key(paper)

        # Check if present in history by stable key or title fallback
        matched_history = existing_history.get(key)
        if not matched_history:
            # Fallback title match check if key didn't hit
            norm_t = normalize_title(paper.title)
            for h in existing_history.values():
                if normalize_title(h.title) == norm_t:
                    matched_history = h
                    break

        if matched_history is None:
            new_papers.append(paper)
        else:
            diffs: Dict[str, Any] = {}
            # Check for substantive changes
            if paper.citation_count != matched_history.citation_count:
                diffs["citation_count"] = {
                    "old": matched_history.citation_count,
                    "new": paper.citation_count,
                }
            if paper.abstract and paper.abstract != matched_history.abstract:
                diffs["abstract"] = "Updated abstract available"
            if paper.pdf_url and paper.pdf_url != matched_history.pdf_url:
                diffs["pdf_url"] = {"old": matched_history.pdf_url, "new": paper.pdf_url}

            if diffs:
                updated_papers.append((paper, diffs))
            else:
                existing_unmodified.append(paper)

    return new_papers, updated_papers, existing_unmodified


class ResearchMonitoringService:
    """Unified service for orchestrating research topic monitoring, change detection, and alert generation."""

    def __init__(
        self,
        search_service: Optional[UnifiedLiteratureSearchService] = None,
        topic_store: Optional[TopicStore] = None,
        notification_provider: Optional[NotificationProvider] = None,
    ):
        self.search_service = search_service or UnifiedLiteratureSearchService()
        self.topic_store = topic_store or TopicStore()
        self.notification_provider = notification_provider or ConsoleNotificationProvider()
        self.trend_analyzer = TrendAnalyzer()
        self.gap_engine = GapEvolutionEngine()

    def execute_monitoring_run(
        self,
        topic: ResearchTopic,
        gap_history: Optional[Dict[str, ResearchGapEvolutionRecord]] = None,
    ) -> Tuple[MonitoringRun, List[ResearchAlert], Dict[str, ResearchGapEvolutionRecord]]:
        """Perform a single end-to-end monitoring run for a research topic."""
        run_id = f"run_{str(uuid.uuid4())[:8]}"
        now_iso = datetime.now(timezone.utc).isoformat()
        logger.info(f"Starting MonitoringRun {run_id} for topic '{topic.name}'")

        run = MonitoringRun(
            run_id=run_id,
            topic_id=topic.topic_id,
            started_at=now_iso,
            query=topic.config.search_query,
            providers=topic.config.providers,
        )

        alerts: List[ResearchAlert] = []
        current_gap_history = dict(gap_history or {})

        try:
            # 1. Search literature across configured providers
            raw_papers, errors = self.search_service.search_all(
                query=topic.config.search_query,
                providers=topic.config.providers,
                limit_per_provider=max(5, topic.config.max_results // len(topic.config.providers or [1])),
                year_from=topic.year_from,
                year_to=topic.year_to,
            )
            run.errors = errors

            # 2. Deduplicate
            deduped = deduplicate_papers(raw_papers)
            run.duplicate_count = len(raw_papers) - len(deduped)

            # 3. Relevance ranking
            ranked = rank_papers_by_relevance(
                query=topic.config.search_query,
                papers=deduped,
            )
            final_papers = ranked[:topic.config.max_results]
            run.papers = final_papers
            run.results_count = len(final_papers)

            # 4. Compare with topic history
            new_papers, updated_papers, _ = detect_paper_changes(final_papers, topic.history)
            run.new_papers_count = len(new_papers)
            run.updated_papers_count = len(updated_papers)
            run.new_paper_ids = [p.paper_id for p in new_papers]
            run.updated_paper_ids = [p.paper_id for p, _ in updated_papers]

            # 5. Update topic history structures
            for p in final_papers:
                key = _resolve_paper_stable_key(p)
                if key in topic.history:
                    entry = topic.history[key]
                    entry.last_seen = now_iso
                    entry.citation_count = p.citation_count
                    entry.current_metadata = p.to_dict()
                else:
                    topic.history[key] = PaperHistoryEntry(
                        paper_id=p.paper_id,
                        first_seen=now_iso,
                        last_seen=now_iso,
                        providers=[p.provider],
                        title=p.title,
                        authors=p.authors,
                        year=p.year,
                        doi=p.doi,
                        arxiv_id=p.arxiv_id,
                        citation_count=p.citation_count,
                        current_metadata=p.to_dict(),
                        abstract=p.abstract,
                        url=getattr(p, "url", getattr(p, "pdf_url", None)),
                        pdf_url=p.pdf_url,
                    )

            # 6. Extract profiles and detect emerging entities for newly discovered papers
            historical_profiles: List[ResearchProfile] = []
            new_methods_detected: List[EmergingEntity] = []
            new_datasets_detected: List[EmergingEntity] = []
            extracted_gaps: List[ResearchGap] = []

            for p in new_papers:
                doc = convert_external_paper_to_document(p)
                profile = extract_research_profile(doc)

                emerging = self.trend_analyzer.detect_emerging_entities(
                    historical_profiles, profile, p.title, detected_at=now_iso
                )
                new_methods_detected.extend(emerging["methods"])
                new_datasets_detected.extend(emerging["datasets"])
                historical_profiles.append(profile)

                # Extract explicit gaps from profile limitations
                for lim in profile.limitations:
                    lim_text = lim.value if hasattr(lim, "value") else str(lim)
                    extracted_gaps.append(
                        ResearchGap(
                            gap_id=f"gap_{len(extracted_gaps) + 1}",
                            title="Empirical Limitation",
                            category="Methodological",
                            gap_type="EXPLICIT_LIMITATION",
                            statement=lim_text,
                            confidence_score=0.75,
                            rationale=f"Identified in '{p.title}': {lim_text}",
                        )
                    )

            # 7. Update research gap evolution
            if extracted_gaps or new_papers:
                current_gap_history = self.gap_engine.track_gap_history(
                    current_gap_history,
                    extracted_gaps,
                    final_papers,
                    paper_title=f"Monitoring Run {run_id}",
                )

            # 8. Alert generation
            if new_papers:
                alerts.append(
                    ResearchAlert(
                        alert_id=f"alt_{str(uuid.uuid4())[:8]}",
                        topic_id=topic.topic_id,
                        type="NEW_PAPER",
                        severity="INFO",
                        title=f"{len(new_papers)} New Papers Discovered",
                        message=f"Discovered {len(new_papers)} newly published or indexed papers matching '{topic.name}'.",
                        evidence=f"Top new paper: '{new_papers[0].title}' ({new_papers[0].provider})",
                    )
                )

            for m in new_methods_detected[:3]:
                alerts.append(
                    ResearchAlert(
                        alert_id=f"alt_{str(uuid.uuid4())[:8]}",
                        topic_id=topic.topic_id,
                        type="NEW_METHOD",
                        severity="IMPORTANT",
                        title=f"New Method Detected: {m.name}",
                        message=f"Observed model/method '{m.name}' first cited in {m.supporting_papers[0] if m.supporting_papers else 'literature'}.",
                        evidence=m.evidence_snippet,
                    )
                )

            for d in new_datasets_detected[:3]:
                alerts.append(
                    ResearchAlert(
                        alert_id=f"alt_{str(uuid.uuid4())[:8]}",
                        topic_id=topic.topic_id,
                        type="NEW_DATASET",
                        severity="IMPORTANT",
                        title=f"New Benchmark Dataset: {d.name}",
                        message=f"Observed evaluation dataset '{d.name}' introduced in recent literature.",
                        evidence=d.evidence_snippet,
                    )
                )

            for gap_rec in current_gap_history.values():
                if gap_rec.status == "POTENTIALLY_ADDRESSED" and gap_rec.addressed_by_papers:
                    alerts.append(
                        ResearchAlert(
                            alert_id=f"alt_{str(uuid.uuid4())[:8]}",
                            topic_id=topic.topic_id,
                            type="POTENTIALLY_ADDRESSED_GAP",
                            severity="HIGH",
                            title=f"Potentially Addressed Gap: {gap_rec.gap_statement[:50]}...",
                            message=f"External literature suggests potential resolution for limitation '{gap_rec.gap_statement[:80]}'.",
                            evidence=f"Addressed by: {', '.join(gap_rec.addressed_by_papers[:2])}",
                        )
                    )

            # Dispatch alerts
            for a in alerts:
                self.notification_provider.send_alert(a)

            # Complete run status
            run.completed_at = datetime.now(timezone.utc).isoformat()
            if errors and not final_papers:
                run.status = "FAILED"
            elif errors:
                run.status = "PARTIAL"
            else:
                run.status = "COMPLETED"

        except Exception as e:
            logger.error(f"Error during MonitoringRun {run_id}: {e}", exc_info=True)
            run.completed_at = datetime.now(timezone.utc).isoformat()
            run.status = "FAILED"
            run.errors.append(str(e))

        # Record run and save topic
        topic.last_checked_at = run.completed_at
        topic.runs.append(run)
        self.topic_store.save_topic(topic)

        return run, alerts, current_gap_history
