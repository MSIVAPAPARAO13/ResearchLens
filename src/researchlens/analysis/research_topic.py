"""Research topic schema, monitoring configurations, runs, and paper history models."""

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from researchlens.analysis.external_provider import ExternalPaper
from researchlens.utils.logging import get_logger

logger = get_logger("research_topic")


@dataclass
class MonitoringConfig:
    """Search configuration for automated research literature monitoring."""

    search_query: str
    providers: List[str] = field(default_factory=lambda: ["arXiv", "Semantic Scholar", "Crossref", "PubMed"])
    max_results: int = 20
    date_filter: Optional[str] = None
    open_access_only: bool = False
    similarity_threshold: float = 0.25
    ranking_config: Dict[str, float] = field(
        default_factory=lambda: {
            "relevance_weight": 0.5,
            "recency_weight": 0.3,
            "citation_weight": 0.2,
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MonitoringConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PaperHistoryEntry:
    """Historical tracking structure for a monitored paper across search iterations."""

    paper_id: str
    first_seen: str
    last_seen: str
    providers: List[str]
    title: str
    authors: List[str]
    year: Optional[int]
    doi: Optional[str]
    arxiv_id: Optional[str]
    citation_count: int
    previous_metadata: Dict[str, Any] = field(default_factory=dict)
    current_metadata: Dict[str, Any] = field(default_factory=dict)
    abstract: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PaperHistoryEntry":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MonitoringRun:
    """Record of an executed literature monitoring iteration."""

    run_id: str
    topic_id: str
    started_at: str
    completed_at: Optional[str] = None
    status: str = "RUNNING"  # RUNNING, COMPLETED, PARTIAL, FAILED
    query: str = ""
    providers: List[str] = field(default_factory=list)
    results_count: int = 0
    new_papers_count: int = 0
    updated_papers_count: int = 0
    duplicate_count: int = 0
    errors: List[str] = field(default_factory=list)
    papers: List[ExternalPaper] = field(default_factory=list)
    new_paper_ids: List[str] = field(default_factory=list)
    updated_paper_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        # ExternalPaper serializes cleanly
        res["papers"] = [p.to_dict() if hasattr(p, "to_dict") else p for p in self.papers]
        return res

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MonitoringRun":
        clean_data = dict(data)
        if "papers" in clean_data:
            clean_data["papers"] = [
                ExternalPaper.from_dict(p) if isinstance(p, dict) else p
                for p in clean_data["papers"]
            ]
        return cls(**{k: v for k, v in clean_data.items() if k in cls.__dataclass_fields__})


@dataclass
class ResearchTopic:
    """Persistent scientific research topic specification tracked over time."""

    topic_id: str
    name: str
    query: str
    description: str = ""
    providers: List[str] = field(default_factory=lambda: ["arXiv", "Semantic Scholar", "Crossref", "PubMed"])
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    max_results: int = 25
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_checked_at: Optional[str] = None
    active: bool = True
    config: MonitoringConfig = field(default_factory=lambda: MonitoringConfig(search_query=""))
    history: Dict[str, PaperHistoryEntry] = field(default_factory=dict)
    runs: List[MonitoringRun] = field(default_factory=list)

    def __post_init__(self):
        if not self.config.search_query:
            self.config.search_query = self.query
            self.config.providers = self.providers
            self.config.max_results = self.max_results

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "name": self.name,
            "query": self.query,
            "description": self.description,
            "providers": self.providers,
            "year_from": self.year_from,
            "year_to": self.year_to,
            "max_results": self.max_results,
            "created_at": self.created_at,
            "last_checked_at": self.last_checked_at,
            "active": self.active,
            "config": self.config.to_dict(),
            "history": {k: v.to_dict() for k, v in self.history.items()},
            "runs": [r.to_dict() for r in self.runs],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchTopic":
        config_data = data.get("config", {})
        config = MonitoringConfig.from_dict(config_data) if config_data else MonitoringConfig(data.get("query", ""))

        history_data = data.get("history", {})
        history = {k: PaperHistoryEntry.from_dict(v) for k, v in history_data.items()}

        runs_data = data.get("runs", [])
        runs = [MonitoringRun.from_dict(r) for r in runs_data]

        return cls(
            topic_id=data.get("topic_id", str(uuid.uuid4())[:8]),
            name=data.get("name", "Untitled Topic"),
            query=data.get("query", ""),
            description=data.get("description", ""),
            providers=data.get("providers", ["arXiv", "Semantic Scholar", "Crossref", "PubMed"]),
            year_from=data.get("year_from"),
            year_to=data.get("year_to"),
            max_results=data.get("max_results", 25),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            last_checked_at=data.get("last_checked_at"),
            active=data.get("active", True),
            config=config,
            history=history,
            runs=runs,
        )


class TopicStore:
    """Local JSON-backed persistent store for ResearchTopics."""

    def __init__(self, storage_dir: str = "data/topics"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def _get_path(self, topic_id: str) -> str:
        return os.path.join(self.storage_dir, f"{topic_id}.json")

    def save_topic(self, topic: ResearchTopic) -> None:
        """Persist a topic to disk."""
        path = self._get_path(topic.topic_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(topic.to_dict(), f, indent=2)
        logger.info(f"Saved ResearchTopic '{topic.name}' ({topic.topic_id}) to {path}")

    def load_topic(self, topic_id: str) -> Optional[ResearchTopic]:
        """Load a single topic from disk."""
        path = self._get_path(topic_id)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ResearchTopic.from_dict(data)

    def list_topics(self) -> List[ResearchTopic]:
        """List all stored research topics."""
        topics = []
        if not os.path.exists(self.storage_dir):
            return topics
        for filename in os.listdir(self.storage_dir):
            if filename.endswith(".json"):
                path = os.path.join(self.storage_dir, filename)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    topics.append(ResearchTopic.from_dict(data))
                except Exception as e:
                    logger.error(f"Error loading topic file {filename}: {e}")
        return sorted(topics, key=lambda t: t.created_at, reverse=True)

    def delete_topic(self, topic_id: str) -> bool:
        """Remove a topic from disk."""
        path = self._get_path(topic_id)
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"Deleted ResearchTopic {topic_id}")
            return True
        return False
