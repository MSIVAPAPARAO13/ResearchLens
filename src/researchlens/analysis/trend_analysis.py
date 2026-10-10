"""Literature trend analysis, emerging methods/datasets/metrics detection, and theme evolution."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from researchlens.analysis.external_provider import ExternalPaper
from researchlens.analysis.semantic_clustering import cluster_limitations_and_future_work
from researchlens.extraction.schema import ResearchProfile
from researchlens.utils.logging import get_logger

logger = get_logger("trend_analysis")


@dataclass
class EmergingEntity:
    """An empirical entity (model, dataset, metric) first observed in a monitoring run."""

    entity_type: str  # method, dataset, metric
    name: str
    first_observed: str
    supporting_papers: List[str]  # Paper titles or IDs
    evidence_snippet: str
    paper_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmergingEntity":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ThemeEvolutionRecord:
    """Cluster of research limitations or methodology themes tracked over time."""

    theme_label: str
    status: str  # NEW_THEME, PERSISTENT_THEME, DECLINING_THEME, EMERGING_THEME
    first_detected: str
    last_detected: str
    item_count: int
    items: List[str] = field(default_factory=list)
    supporting_papers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LiteratureTrendReport:
    """Consolidated summary of empirical trends detected across monitoring runs."""

    topic_name: str
    analyzed_runs_count: int
    new_methods: List[EmergingEntity] = field(default_factory=list)
    new_datasets: List[EmergingEntity] = field(default_factory=list)
    new_metrics: List[EmergingEntity] = field(default_factory=list)
    theme_evolutions: List[ThemeEvolutionRecord] = field(default_factory=list)
    top_recurring_limitations: List[Dict[str, Any]] = field(default_factory=list)
    method_frequency: Dict[str, int] = field(default_factory=dict)
    dataset_frequency: Dict[str, int] = field(default_factory=dict)
    metric_frequency: Dict[str, int] = field(default_factory=dict)
    summary_narrative: str = ""

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["new_methods"] = [m.to_dict() for m in self.new_methods]
        res["new_datasets"] = [d.to_dict() for d in self.new_datasets]
        res["new_metrics"] = [m.to_dict() for m in self.new_metrics]
        res["theme_evolutions"] = [t.to_dict() for t in self.theme_evolutions]
        return res


def _item_str(item: Any) -> str:
    if hasattr(item, "value"):
        return str(item.value)
    return str(item)


class TrendAnalyzer:
    """Analyzes longitudinal shifts in scientific methods, datasets, metrics, and themes."""

    def __init__(self):
        pass

    def detect_emerging_entities(
        self,
        historical_profiles: List[ResearchProfile],
        current_profile: ResearchProfile,
        paper_title: str,
        detected_at: Optional[str] = None,
    ) -> Dict[str, List[EmergingEntity]]:
        """Identify methods, datasets, and metrics in current profile absent in previous history."""
        timestamp = detected_at or datetime.now(timezone.utc).isoformat()

        # Aggregate historical sets
        hist_methods: Set[str] = set()
        hist_datasets: Set[str] = set()
        hist_metrics: Set[str] = set()

        for p in historical_profiles:
            hist_methods.update(_item_str(m).lower().strip() for m in p.models)
            hist_datasets.update(_item_str(d).lower().strip() for d in p.datasets)
            hist_metrics.update(_item_str(m).lower().strip() for m in p.metrics)

        new_methods: List[EmergingEntity] = []
        for m in current_profile.models:
            clean = _item_str(m).strip()
            if clean.lower() not in hist_methods and len(clean) > 2:
                new_methods.append(
                    EmergingEntity(
                        entity_type="method",
                        name=clean,
                        first_observed=timestamp,
                        supporting_papers=[paper_title],
                        evidence_snippet=f"Observed model/method '{clean}' in {paper_title}",
                    )
                )

        new_datasets: List[EmergingEntity] = []
        for d in current_profile.datasets:
            clean = _item_str(d).strip()
            if clean.lower() not in hist_datasets and len(clean) > 2:
                new_datasets.append(
                    EmergingEntity(
                        entity_type="dataset",
                        name=clean,
                        first_observed=timestamp,
                        supporting_papers=[paper_title],
                        evidence_snippet=f"Observed benchmark dataset '{clean}' in {paper_title}",
                    )
                )

        new_metrics: List[EmergingEntity] = []
        for met in current_profile.metrics:
            clean = _item_str(met).strip()
            if clean.lower() not in hist_metrics and len(clean) > 1:
                new_metrics.append(
                    EmergingEntity(
                        entity_type="metric",
                        name=clean,
                        first_observed=timestamp,
                        supporting_papers=[paper_title],
                        evidence_snippet=f"Observed evaluation metric '{clean}' in {paper_title}",
                    )
                )

        return {
            "methods": new_methods,
            "datasets": new_datasets,
            "metrics": new_metrics,
        }

    def analyze_themes_evolution(
        self,
        previous_limitation_texts: List[str],
        current_limitation_texts: List[str],
    ) -> List[ThemeEvolutionRecord]:
        """Cluster limitations and track theme persistence or emergence across runs."""
        now_iso = datetime.now(timezone.utc).isoformat()
        records: List[ThemeEvolutionRecord] = []

        if not current_limitation_texts and not previous_limitation_texts:
            return records

        # Cluster current texts
        current_clusters = cluster_limitations_and_future_work(current_limitation_texts)
        prev_clusters = cluster_limitations_and_future_work(previous_limitation_texts) if previous_limitation_texts else []

        prev_labels = {c.theme_label.lower(): c for c in prev_clusters}

        for c in current_clusters:
            label_lower = c.theme_label.lower()
            if label_lower not in prev_labels:
                status = "NEW_THEME" if len(prev_clusters) > 0 else "PERSISTENT_THEME"
            else:
                prev_c = prev_labels[label_lower]
                if len(c.items) > len(prev_c.items):
                    status = "EMERGING_THEME"
                elif len(c.items) < len(prev_c.items):
                    status = "DECLINING_THEME"
                else:
                    status = "PERSISTENT_THEME"

            records.append(
                ThemeEvolutionRecord(
                    theme_label=c.theme_label,
                    status=status,
                    first_detected=now_iso,
                    last_detected=now_iso,
                    item_count=len(c.items),
                    items=c.items,
                )
            )

        return records

    def build_trend_report(
        self,
        topic_name: str,
        historical_runs_papers: List[List[ExternalPaper]],
        all_profiles: List[ResearchProfile],
    ) -> LiteratureTrendReport:
        """Synthesize high-level trend report across multiple historical runs."""
        method_counts: Dict[str, int] = {}
        dataset_counts: Dict[str, int] = {}
        metric_counts: Dict[str, int] = {}

        for prof in all_profiles:
            for m in prof.models:
                m_c = _item_str(m).strip()
                if m_c:
                    method_counts[m_c] = method_counts.get(m_c, 0) + 1
            for d in prof.datasets:
                d_c = _item_str(d).strip()
                if d_c:
                    dataset_counts[d_c] = dataset_counts.get(d_c, 0) + 1
            for met in prof.metrics:
                met_c = _item_str(met).strip()
                if met_c:
                    metric_counts[met_c] = metric_counts.get(met_c, 0) + 1

        narrative = (
            f"Among the literature retrieved by ResearchLens for topic '{topic_name}' "
            f"across {len(historical_runs_papers)} monitoring runs, the most frequently observed "
            f"architectures include: {', '.join(sorted(method_counts.keys(), key=lambda k: method_counts[k], reverse=True)[:3]) or 'None'}. "
            f"Evaluations predominantly utilized: {', '.join(sorted(dataset_counts.keys(), key=lambda k: dataset_counts[k], reverse=True)[:3]) or 'None'}."
        )

        return LiteratureTrendReport(
            topic_name=topic_name,
            analyzed_runs_count=len(historical_runs_papers),
            method_frequency=method_counts,
            dataset_frequency=dataset_counts,
            metric_frequency=metric_counts,
            summary_narrative=narrative,
        )
