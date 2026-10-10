"""Cross-paper Literature-Level Research Gap Engine."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from researchlens.analysis.literature_collection import LiteratureCollection
from researchlens.analysis.semantic_clustering import cluster_limitations_and_future_work
from researchlens.extraction.schema import ExtractedItem

logger = logging.getLogger("researchlens.cross_paper_gap_engine")


@dataclass
class LiteratureResearchGap:
    """Dataclass representing a literature-level research gap derived across multiple papers."""

    gap_id: str
    title: str
    category: str
    statement: str
    supporting_papers: List[str] = field(default_factory=list)
    supporting_evidence: List[ExtractedItem] = field(default_factory=list)
    recurrence_count: int = 1
    confidence_level: str = "Medium"  # High, Medium, Low
    confidence_score: float = 0.5
    rationale: str = ""
    potential_direction: str = ""
    comparability_notes: str = "Evaluated across collection papers."
    independence_status: str = "Independent-source status not verified"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize LiteratureResearchGap to a JSON-compatible dictionary."""
        return {
            "gap_id": self.gap_id,
            "title": self.title,
            "category": self.category,
            "statement": self.statement,
            "supporting_papers": self.supporting_papers,
            "supporting_evidence": [
                ev.to_dict() if hasattr(ev, "to_dict") else str(ev) for ev in self.supporting_evidence
            ],
            "recurrence_count": self.recurrence_count,
            "confidence_level": self.confidence_level,
            "confidence_score": round(self.confidence_score, 4),
            "rationale": self.rationale,
            "potential_direction": self.potential_direction,
            "comparability_notes": self.comparability_notes,
            "independence_status": self.independence_status,
        }


class CrossPaperResearchGapEngine:
    """Engine analyzing LiteratureCollections to formulate evidence-backed literature-level research gaps."""

    def __init__(self, ablation_mode: str = "full_evidence"):
        self.ablation_mode = ablation_mode

    def analyze_collection(
        self, collection: LiteratureCollection
    ) -> List[LiteratureResearchGap]:
        """Analyze a LiteratureCollection and extract cross-paper literature research gaps."""
        logger.info(f"Executing CrossPaperResearchGapEngine on collection '{collection.collection_name}' ({collection.paper_count} papers)")

        if collection.paper_count == 0:
            return [
                LiteratureResearchGap(
                    gap_id="gap_lit_refusal",
                    title="Empty Literature Collection",
                    category="OTHER",
                    statement="No papers available in the collection for literature gap analysis.",
                    confidence_level="Low",
                    confidence_score=0.0,
                    rationale="The collection contains zero parsed research papers.",
                    potential_direction="Upload multiple research paper PDFs to generate literature intelligence.",
                )
            ]

        lim_clusters, fw_clusters = cluster_limitations_and_future_work(
            collection=collection, ablation_mode=self.ablation_mode
        )

        gaps: List[LiteratureResearchGap] = []
        gap_idx = 1

        # 1. Process Recurring Limitation Clusters
        for cluster in lim_clusters:
            score, level = self._calculate_cross_paper_score(cluster, is_explicit=True)

            stmt = f"Across the analyzed collection ({len(cluster.supporting_papers)} papers), a recurring constraint was identified regarding {cluster.theme_name.lower()}."
            rat = f"Supported by evidence from {len(cluster.supporting_papers)} independent documents ({', '.join(cluster.supporting_papers)})."

            gaps.append(
                LiteratureResearchGap(
                    gap_id=f"lit_gap_{gap_idx}",
                    title=f"Recurring Limitation: {cluster.theme_name[:50]}",
                    category=cluster.category,
                    statement=stmt,
                    supporting_papers=cluster.supporting_papers,
                    supporting_evidence=cluster.supporting_evidence,
                    recurrence_count=cluster.recurrence_count,
                    confidence_level=level,
                    confidence_score=score,
                    rationale=rat,
                    potential_direction=f"Develop literature-wide frameworks addressing {cluster.theme_name.lower()}.",
                    independence_status=self._check_author_independence(cluster.supporting_papers, collection),
                )
            )
            gap_idx += 1

        # 2. Process Cross-Paper Dataset Generalization Gaps
        dataset_gaps = self._detect_cross_paper_dataset_gaps(collection, gap_idx)
        gaps.extend(dataset_gaps)

        # Refusal fallback if no valid gaps detected
        if not gaps:
            return [
                LiteratureResearchGap(
                    gap_id="lit_gap_refusal",
                    title="Insufficient Evidence",
                    category="OTHER",
                    statement="Across the analyzed papers, insufficient recurring evidence was found to identify high-confidence literature gaps.",
                    confidence_level="Low",
                    confidence_score=0.0,
                    rationale="No recurring limitations or unaddressed methodology gaps met the evidence threshold.",
                    potential_direction="Expand the literature collection by uploading additional related research papers.",
                )
            ]

        # Sort gaps by confidence score and recurrence
        return sorted(gaps, key=lambda g: (g.confidence_score, g.recurrence_count), reverse=True)

    def _calculate_cross_paper_score(
        self, cluster: Any, is_explicit: bool
    ) -> Tuple[float, str]:
        # Transparent scoring formula:
        # score = 0.25*evidence_strength + 0.15*specificity + 0.25*paper_recurrence + 0.15*independence + 0.10*impact + 0.10*researchability
        ev_strength = 0.90 if is_explicit else 0.70
        specificity = 0.85 if cluster.category != "OTHER" else 0.50
        recurrence = min(1.0, 0.50 + 0.25 * (cluster.recurrence_count - 1))
        independence = 0.70  # Default unverified status
        impact = 0.80 if cluster.category in ("GENERALIZATION", "METHODOLOGY", "EFFICIENCY") else 0.65
        researchability = 0.85

        score = (
            0.25 * ev_strength
            + 0.15 * specificity
            + 0.25 * recurrence
            + 0.15 * independence
            + 0.10 * impact
            + 0.10 * researchability
        )
        score = round(min(max(score, 0.0), 1.0), 4)

        if score >= 0.80 and cluster.recurrence_count >= 2:
            level = "High"
        elif score >= 0.55:
            level = "Medium"
        else:
            level = "Low"

        return score, level

    def _detect_cross_paper_dataset_gaps(
        self, collection: LiteratureCollection, start_idx: int
    ) -> List[LiteratureResearchGap]:
        dataset_usage: Dict[str, List[str]] = {}
        for doc in collection.documents:
            if not doc.profile:
                continue
            for ds in doc.profile.datasets:
                v = ds.value.strip()
                if v and v not in ("Not detected", "Not explicitly stated"):
                    if v not in dataset_usage:
                        dataset_usage[v] = []
                    if doc.title not in dataset_usage[v]:
                        dataset_usage[v].append(doc.title)

        gaps = []
        if dataset_usage and len(collection.documents) >= 2:
            # If all papers evaluate on distinct single datasets
            papers_with_single_ds = [
                paper_titles[0]
                for ds_name, paper_titles in dataset_usage.items()
                if len(paper_titles) == 1
            ]
            if len(papers_with_single_ds) >= 2:
                gaps.append(
                    LiteratureResearchGap(
                        gap_id=f"lit_gap_{start_idx}",
                        title="Cross-Dataset Benchmark Fragmenting",
                        category="GENERALIZATION",
                        statement="Across the analyzed collection, distinct papers rely on fragmented single datasets, leaving cross-benchmark model evaluation incomplete.",
                        supporting_papers=papers_with_single_ds[:5],
                        recurrence_count=len(papers_with_single_ds),
                        confidence_level="High" if len(papers_with_single_ds) >= 3 else "Medium",
                        confidence_score=0.82,
                        rationale=f"Identified across {len(papers_with_single_ds)} papers using isolated datasets without unified cross-dataset validation.",
                        potential_direction="Conduct a unified cross-dataset benchmark evaluating all proposed architectures on a shared test harness.",
                        independence_status=self._check_author_independence(papers_with_single_ds, collection),
                    )
                )

        return gaps

    def _check_author_independence(
        self, paper_titles: List[str], collection: LiteratureCollection
    ) -> str:
        authors_seen = set()
        overlap_found = False

        for title in paper_titles:
            for doc in collection.documents:
                if doc.title == title and doc.profile:
                    p_authors = [a.value.lower() for a in doc.profile.authors if a.value != "Not detected"]
                    for a in p_authors:
                        if a in authors_seen:
                            overlap_found = True
                        authors_seen.add(a)

        if overlap_found:
            return "Shared author/institution overlaps detected across supporting papers"
        elif len(authors_seen) >= 2:
            return "Independent research teams confirmed via author metadata"
        return "Independent-source status not verified"
