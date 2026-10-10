"""Data models and schemas for ResearchLens Research Workspaces (Phase 10)."""

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def generate_id(prefix: str = "") -> str:
    """Generate a deterministic or unique ID with an optional prefix."""
    u = uuid.uuid4().hex[:12]
    return f"{prefix}_{u}" if prefix else u


def current_iso_time() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ResearchWorkspace:
    """Top-level research project workspace."""

    workspace_id: str
    name: str
    description: str = ""
    topic_id: Optional[str] = None
    created_at: str = field(default_factory=current_iso_time)
    updated_at: str = field(default_factory=current_iso_time)
    status: str = "ACTIVE"  # ACTIVE, ARCHIVED

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchWorkspace":
        return cls(
            workspace_id=data.get("workspace_id", generate_id("ws")),
            name=data.get("name", "Untitled Workspace"),
            description=data.get("description", ""),
            topic_id=data.get("topic_id"),
            created_at=data.get("created_at", current_iso_time()),
            updated_at=data.get("updated_at", current_iso_time()),
            status=data.get("status", "ACTIVE"),
        )


@dataclass
class PaperCollection:
    """Named user collection or folder of papers within a workspace."""

    collection_id: str
    workspace_id: str
    name: str
    description: str = ""
    paper_ids: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=current_iso_time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PaperCollection":
        return cls(
            collection_id=data.get("collection_id", generate_id("col")),
            workspace_id=data.get("workspace_id", ""),
            name=data.get("name", "Untitled Collection"),
            description=data.get("description", ""),
            paper_ids=data.get("paper_ids", []),
            created_at=data.get("created_at", current_iso_time()),
        )


@dataclass
class SavedPaper:
    """Paper entry saved into a workspace with researcher tags, notes, and reading status."""

    paper_id: str
    workspace_id: str
    saved_at: str = field(default_factory=current_iso_time)
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    priority: str = "MEDIUM"  # LOW, MEDIUM, HIGH
    status: str = "UNREAD"  # UNREAD, READ, REVIEWED
    title: str = ""
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SavedPaper":
        return cls(
            paper_id=data.get("paper_id", ""),
            workspace_id=data.get("workspace_id", ""),
            saved_at=data.get("saved_at", current_iso_time()),
            tags=data.get("tags", []),
            notes=data.get("notes", ""),
            priority=data.get("priority", "MEDIUM"),
            status=data.get("status", "UNREAD"),
            title=data.get("title", ""),
            authors=data.get("authors", []),
            year=data.get("year"),
            doi=data.get("doi"),
            arxiv_id=data.get("arxiv_id"),
        )


@dataclass
class EvidenceBookmark:
    """Precise evidence bookmark referencing a paper section, page, or chunk."""

    bookmark_id: str
    workspace_id: str
    paper_id: str
    page_number: Optional[int] = None
    section_name: Optional[str] = None
    chunk_id: Optional[str] = None
    source_text: str = ""
    label: str = ""
    note: str = ""
    created_by: str = ""
    visibility: str = "WORKSPACE"  # WORKSPACE, PRIVATE
    comments_count: int = 0
    created_at: str = field(default_factory=current_iso_time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceBookmark":
        return cls(
            bookmark_id=data.get("bookmark_id", generate_id("bm")),
            workspace_id=data.get("workspace_id", ""),
            paper_id=data.get("paper_id", ""),
            page_number=data.get("page_number"),
            section_name=data.get("section_name"),
            chunk_id=data.get("chunk_id"),
            source_text=data.get("source_text", ""),
            label=data.get("label", ""),
            note=data.get("note", ""),
            created_by=data.get("created_by", ""),
            visibility=data.get("visibility", "WORKSPACE"),
            comments_count=int(data.get("comments_count", 0)),
            created_at=data.get("created_at", current_iso_time()),
        )


@dataclass
class ResearchAnnotation:
    """Researcher annotation attached to a specific passage of evidence."""

    annotation_id: str
    workspace_id: str
    paper_id: str
    page_number: Optional[int] = None
    section_name: Optional[str] = None
    chunk_id: Optional[str] = None
    source_text: str = ""
    annotation_type: str = "IMPORTANT"  # IMPORTANT, QUESTION, LIMITATION, METHOD, RESULT, IDEA, FOLLOW_UP
    content: str = ""
    created_at: str = field(default_factory=current_iso_time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchAnnotation":
        return cls(
            annotation_id=data.get("annotation_id", generate_id("ann")),
            workspace_id=data.get("workspace_id", ""),
            paper_id=data.get("paper_id", ""),
            page_number=data.get("page_number"),
            section_name=data.get("section_name"),
            chunk_id=data.get("chunk_id"),
            source_text=data.get("source_text", ""),
            annotation_type=data.get("annotation_type", "IMPORTANT"),
            content=data.get("content", ""),
            created_at=data.get("created_at", current_iso_time()),
        )


@dataclass
class ResearchQuestion:
    """Formal research question grounded in gaps and evidence."""

    question_id: str
    workspace_id: str
    text: str
    type: str = "PRIMARY"  # PRIMARY, SECONDARY, EXPLORATORY
    status: str = "ACTIVE"  # DRAFT, ACTIVE, ANSWERED, ARCHIVED
    priority: str = "MEDIUM"  # LOW, MEDIUM, HIGH
    supporting_gap_ids: List[str] = field(default_factory=list)
    supporting_evidence_ids: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=current_iso_time)
    updated_at: str = field(default_factory=current_iso_time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchQuestion":
        return cls(
            question_id=data.get("question_id", generate_id("rq")),
            workspace_id=data.get("workspace_id", ""),
            text=data.get("text", ""),
            type=data.get("type", "PRIMARY"),
            status=data.get("status", "ACTIVE"),
            priority=data.get("priority", "MEDIUM"),
            supporting_gap_ids=data.get("supporting_gap_ids", []),
            supporting_evidence_ids=data.get("supporting_evidence_ids", []),
            created_at=data.get("created_at", current_iso_time()),
            updated_at=data.get("updated_at", current_iso_time()),
        )


@dataclass
class ResearchGap:
    """Evidence-grounded research gap identifying limitations or missing evaluations."""

    gap_id: str
    workspace_id: str = ""
    category: str = "METHODOLOGY"
    description: str = ""
    statement: str = ""
    paper_ids: List[str] = field(default_factory=list)
    supporting_evidence: List[str] = field(default_factory=list)
    verified: bool = False
    status: str = "CONFIRMED_OPEN_GAP"
    created_at: str = field(default_factory=current_iso_time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchGap":
        return cls(
            gap_id=data.get("gap_id", generate_id("gap")),
            workspace_id=data.get("workspace_id", ""),
            category=data.get("category", "METHODOLOGY"),
            description=data.get("description", data.get("statement", "")),
            statement=data.get("statement", data.get("description", "")),
            paper_ids=data.get("paper_ids", []),
            supporting_evidence=data.get("supporting_evidence", []),
            verified=data.get("verified", False),
            status=data.get("status", "CONFIRMED_OPEN_GAP"),
            created_at=data.get("created_at", current_iso_time()),
        )


@dataclass
class ResearchHypothesis:
    """Scientific hypothesis articulating expected variables and relationships."""

    hypothesis_id: str
    workspace_id: str
    question_id: Optional[str] = None
    statement: str = ""
    independent_variables: List[str] = field(default_factory=list)
    dependent_variables: List[str] = field(default_factory=list)
    expected_relationship: str = ""
    supporting_evidence: List[str] = field(default_factory=list)
    rationale: str = ""
    testable: bool = True
    status: str = "PROPOSED"  # DRAFT, PROPOSED, TESTED, SUPPORTED, NOT_SUPPORTED, INCONCLUSIVE
    created_at: str = field(default_factory=current_iso_time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchHypothesis":
        return cls(
            hypothesis_id=data.get("hypothesis_id", generate_id("hyp")),
            workspace_id=data.get("workspace_id", ""),
            question_id=data.get("question_id"),
            statement=data.get("statement", ""),
            independent_variables=data.get("independent_variables", []),
            dependent_variables=data.get("dependent_variables", []),
            expected_relationship=data.get("expected_relationship", ""),
            supporting_evidence=data.get("supporting_evidence", []),
            rationale=data.get("rationale", ""),
            testable=data.get("testable", True),
            status=data.get("status", "PROPOSED"),
            created_at=data.get("created_at", current_iso_time()),
        )


@dataclass
class ResearchDirection:
    """Potential research direction derived from gap clusters with transparent prioritization."""

    direction_id: str
    workspace_id: str
    title: str
    description: str = ""
    source_gap_ids: List[str] = field(default_factory=list)
    supporting_papers: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    priority: str = "MEDIUM"  # LOW, MEDIUM, HIGH
    status: str = "IDEA"  # IDEA, SHORTLISTED, PLANNED, IN_PROGRESS, COMPLETED, REJECTED
    priority_score: float = 0.0
    score_breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchDirection":
        return cls(
            direction_id=data.get("direction_id", generate_id("rd")),
            workspace_id=data.get("workspace_id", ""),
            title=data.get("title", ""),
            description=data.get("description", ""),
            source_gap_ids=data.get("source_gap_ids", []),
            supporting_papers=data.get("supporting_papers", []),
            evidence=data.get("evidence", []),
            priority=data.get("priority", "MEDIUM"),
            status=data.get("status", "IDEA"),
            priority_score=float(data.get("priority_score", 0.0)),
            score_breakdown=data.get("score_breakdown", {}),
        )


@dataclass
class ExperimentPlan:
    """Planned empirical evaluation or benchmark test."""

    experiment_id: str
    research_plan_id: Optional[str] = None
    name: str = ""
    hypothesis_id: Optional[str] = None
    dataset: str = ""
    model: str = ""
    baseline: str = ""
    metrics: List[str] = field(default_factory=list)
    variables: Dict[str, str] = field(default_factory=dict)
    expected_outcome: str = ""
    status: str = "PLANNED"  # PLANNED, RUNNING, COMPLETED, BLOCKED
    version: int = 1
    parent_experiment_id: Optional[str] = None
    created_by: str = ""
    entrypoint: str = ""
    command_template: str = ""
    working_directory: str = ""
    environment_id: Optional[str] = None
    dataset_id: Optional[str] = None
    model_id: Optional[str] = None
    configuration: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 3600
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    execution_backend: str = "LOCAL"  # LOCAL, DOCKER, REMOTE
    execution_image: Optional[str] = None
    created_at: str = field(default_factory=current_iso_time)
    updated_at: str = field(default_factory=current_iso_time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentPlan":
        return cls(
            experiment_id=data.get("experiment_id", generate_id("exp")),
            research_plan_id=data.get("research_plan_id"),
            name=data.get("name", "Untitled Experiment"),
            hypothesis_id=data.get("hypothesis_id"),
            dataset=data.get("dataset", ""),
            model=data.get("model", ""),
            baseline=data.get("baseline", ""),
            metrics=data.get("metrics", []),
            variables=data.get("variables", {}),
            expected_outcome=data.get("expected_outcome", ""),
            status=data.get("status", "PLANNED"),
            version=int(data.get("version", 1)),
            parent_experiment_id=data.get("parent_experiment_id"),
            created_by=data.get("created_by", ""),
            entrypoint=data.get("entrypoint", ""),
            command_template=data.get("command_template", ""),
            working_directory=data.get("working_directory", ""),
            environment_id=data.get("environment_id"),
            dataset_id=data.get("dataset_id"),
            model_id=data.get("model_id"),
            configuration=data.get("configuration", {}),
            timeout_seconds=int(data.get("timeout_seconds", 3600)),
            resource_requirements=data.get("resource_requirements", {}),
            execution_backend=data.get("execution_backend", "LOCAL"),
            execution_image=data.get("execution_image"),
            created_at=data.get("created_at", current_iso_time()),
            updated_at=data.get("updated_at", current_iso_time()),
        )


@dataclass
class ResearchRisk:
    """Identified research risk grounded in literature limitations."""

    risk_id: str
    workspace_id: str
    category: str = "GENERALIZATION"  # DATA, MODEL, COMPUTE, EVALUATION, GENERALIZATION, REPRODUCIBILITY, DATA_LEAKAGE, BIAS, EXTERNAL_VALIDITY
    description: str = ""
    evidence: List[str] = field(default_factory=list)
    severity: str = "MEDIUM"  # LOW, MEDIUM, HIGH
    mitigation: str = ""
    status: str = "OPEN"  # OPEN, MITIGATED, ACCEPTED

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchRisk":
        return cls(
            risk_id=data.get("risk_id", generate_id("risk")),
            workspace_id=data.get("workspace_id", ""),
            category=data.get("category", "GENERALIZATION"),
            description=data.get("description", ""),
            evidence=data.get("evidence", []),
            severity=data.get("severity", "MEDIUM"),
            mitigation=data.get("mitigation", ""),
            status=data.get("status", "OPEN"),
        )


@dataclass
class ResearchMilestone:
    """Project timeline milestone."""

    milestone_id: str
    workspace_id: str
    title: str
    description: str = ""
    target_date: Optional[str] = None
    status: str = "NOT_STARTED"  # NOT_STARTED, IN_PROGRESS, COMPLETED, BLOCKED

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchMilestone":
        return cls(
            milestone_id=data.get("milestone_id", generate_id("ms")),
            workspace_id=data.get("workspace_id", ""),
            title=data.get("title", ""),
            description=data.get("description", ""),
            target_date=data.get("target_date"),
            status=data.get("status", "NOT_STARTED"),
        )


@dataclass
class ResearchPlan:
    """Structured, evidence-grounded research project plan."""

    plan_id: str
    workspace_id: str
    title: str
    problem_statement: str = ""
    research_questions: List[str] = field(default_factory=list)
    hypotheses: List[str] = field(default_factory=list)
    methodology: Dict[str, Any] = field(default_factory=dict)
    datasets: List[Dict[str, Any]] = field(default_factory=list)
    baselines: List[Dict[str, Any]] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    experiments: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    milestones: List[str] = field(default_factory=list)
    traceability: List[Dict[str, Any]] = field(default_factory=list)
    version: int = 1
    parent_plan_id: Optional[str] = None
    author_id: str = ""
    created_at: str = field(default_factory=current_iso_time)
    updated_at: str = field(default_factory=current_iso_time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchPlan":
        return cls(
            plan_id=data.get("plan_id", generate_id("rp")),
            workspace_id=data.get("workspace_id", ""),
            title=data.get("title", "Research Plan"),
            problem_statement=data.get("problem_statement", ""),
            research_questions=data.get("research_questions", []),
            hypotheses=data.get("hypotheses", []),
            methodology=data.get("methodology", {}),
            datasets=data.get("datasets", []),
            baselines=data.get("baselines", []),
            metrics=data.get("metrics", []),
            experiments=data.get("experiments", []),
            risks=data.get("risks", []),
            milestones=data.get("milestones", []),
            traceability=data.get("traceability", []),
            version=int(data.get("version", 1)),
            parent_plan_id=data.get("parent_plan_id"),
            author_id=data.get("author_id", ""),
            created_at=data.get("created_at", current_iso_time()),
            updated_at=data.get("updated_at", current_iso_time()),
        )


# =============================================================================
# Phase 11: Collaboration & Experiment Intelligence Models
# =============================================================================


@dataclass
class WorkspaceMember:
    """Researcher participating in a collaborative research workspace."""

    member_id: str
    workspace_id: str
    display_name: str
    email: str
    role: str = "RESEARCHER"  # OWNER, RESEARCHER, REVIEWER, VIEWER
    status: str = "ACTIVE"  # ACTIVE, INVITED, REMOVED
    joined_at: str = field(default_factory=current_iso_time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkspaceMember":
        return cls(
            member_id=data.get("member_id", generate_id("mem")),
            workspace_id=data.get("workspace_id", ""),
            display_name=data.get("display_name", "Researcher"),
            email=data.get("email", ""),
            role=data.get("role", "RESEARCHER"),
            status=data.get("status", "ACTIVE"),
            joined_at=data.get("joined_at", current_iso_time()),
        )


@dataclass
class ResearchComment:
    """Comment on a paper, evidence bookmark, question, hypothesis, experiment, or result."""

    comment_id: str
    workspace_id: str
    author_id: str
    target_type: str  # PAPER, EVIDENCE, QUESTION, HYPOTHESIS, EXPERIMENT, RESULT, RESEARCH_PLAN
    target_id: str
    content: str
    created_at: str = field(default_factory=current_iso_time)
    updated_at: str = field(default_factory=current_iso_time)
    status: str = "ACTIVE"  # ACTIVE, RESOLVED, DELETED

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchComment":
        return cls(
            comment_id=data.get("comment_id", generate_id("cmt")),
            workspace_id=data.get("workspace_id", ""),
            author_id=data.get("author_id", ""),
            target_type=data.get("target_type", "PAPER"),
            target_id=data.get("target_id", ""),
            content=data.get("content", ""),
            created_at=data.get("created_at", current_iso_time()),
            updated_at=data.get("updated_at", current_iso_time()),
            status=data.get("status", "ACTIVE"),
        )


@dataclass
class DiscussionMessage:
    """Individual message inside a discussion thread."""

    message_id: str
    thread_id: str
    author_id: str
    content: str
    created_at: str = field(default_factory=current_iso_time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiscussionMessage":
        return cls(
            message_id=data.get("message_id", generate_id("msg")),
            thread_id=data.get("thread_id", ""),
            author_id=data.get("author_id", ""),
            content=data.get("content", ""),
            created_at=data.get("created_at", current_iso_time()),
        )


@dataclass
class DiscussionThread:
    """Collaborative research discussion thread centered on a specific entity."""

    thread_id: str
    workspace_id: str
    target_type: str  # PAPER, EVIDENCE, QUESTION, HYPOTHESIS, EXPERIMENT, RESULT
    target_id: str
    title: str
    status: str = "OPEN"  # OPEN, RESOLVED, ARCHIVED
    messages: List[DiscussionMessage] = field(default_factory=list)
    created_at: str = field(default_factory=current_iso_time)
    updated_at: str = field(default_factory=current_iso_time)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["messages"] = [m.to_dict() if isinstance(m, DiscussionMessage) else m for m in self.messages]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiscussionThread":
        raw_msgs = data.get("messages", [])
        msgs = [DiscussionMessage.from_dict(m) if isinstance(m, dict) else m for m in raw_msgs]
        return cls(
            thread_id=data.get("thread_id", generate_id("thd")),
            workspace_id=data.get("workspace_id", ""),
            target_type=data.get("target_type", "PAPER"),
            target_id=data.get("target_id", ""),
            title=data.get("title", "Discussion"),
            status=data.get("status", "OPEN"),
            messages=msgs,
            created_at=data.get("created_at", current_iso_time()),
            updated_at=data.get("updated_at", current_iso_time()),
        )


@dataclass
class ExperimentRun:
    """Historical execution record of an experiment (metadata, environment, and execution tracking)."""

    run_id: str
    experiment_id: str
    run_number: int = 1
    started_at: str = field(default_factory=current_iso_time)
    completed_at: Optional[str] = None
    status: str = "COMPLETED"  # QUEUED, PREPARING, RUNNING, COMPLETED, FAILED, TIMEOUT, CANCELLED, BLOCKED
    environment: str = ""
    hardware: str = ""
    software_versions: Dict[str, str] = field(default_factory=dict)
    seed: Optional[int] = None
    notes: str = ""
    git_commit_sha: str = "NOT_AVAILABLE"
    entrypoint: str = ""
    command_executed: str = ""
    duration_seconds: float = 0.0
    exit_code: Optional[int] = None
    failure_category: Optional[str] = None  # CONFIGURATION_ERROR, DEPENDENCY_ERROR, DATA_ERROR, MODEL_ERROR, RESOURCE_ERROR, TIMEOUT, UNKNOWN
    cancelled_by: Optional[str] = None
    cancelled_at: Optional[str] = None
    cancellation_reason: Optional[str] = None
    artifact_ids: List[str] = field(default_factory=list)
    mlflow_run_id: Optional[str] = None
    mlflow_experiment_id: Optional[str] = None
    mlflow_sync_status: Optional[str] = None  # SYNCED, PENDING, FAILED, DISABLED
    mlflow_sync_error: Optional[str] = None
    execution_backend: str = "LOCAL"  # LOCAL, DOCKER, REMOTE
    execution_image: Optional[str] = None
    image_digest: Optional[str] = None
    job_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentRun":
        return cls(
            run_id=data.get("run_id", generate_id("run")),
            experiment_id=data.get("experiment_id", ""),
            run_number=int(data.get("run_number", 1)),
            started_at=data.get("started_at", current_iso_time()),
            completed_at=data.get("completed_at"),
            status=data.get("status", "COMPLETED"),
            environment=data.get("environment", ""),
            hardware=data.get("hardware", ""),
            software_versions=data.get("software_versions", {}),
            seed=data.get("seed"),
            notes=data.get("notes", ""),
            git_commit_sha=data.get("git_commit_sha", "NOT_AVAILABLE"),
            entrypoint=data.get("entrypoint", ""),
            command_executed=data.get("command_executed", ""),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            exit_code=data.get("exit_code"),
            failure_category=data.get("failure_category"),
            cancelled_by=data.get("cancelled_by"),
            cancelled_at=data.get("cancelled_at"),
            cancellation_reason=data.get("cancellation_reason"),
            artifact_ids=data.get("artifact_ids", []),
            mlflow_run_id=data.get("mlflow_run_id"),
            mlflow_experiment_id=data.get("mlflow_experiment_id"),
            mlflow_sync_status=data.get("mlflow_sync_status"),
            mlflow_sync_error=data.get("mlflow_sync_error"),
            execution_backend=data.get("execution_backend", "LOCAL"),
            execution_image=data.get("execution_image"),
            image_digest=data.get("image_digest"),
            job_id=data.get("job_id"),
        )


@dataclass
class ExperimentResult:
    """Normalized empirical metric observed from an ingested experiment run."""

    result_id: str
    run_id: str
    model: str
    dataset: str
    metric: str
    value: float
    unit: str = ""
    split: str = "test"
    seed: Optional[int] = None
    source: str = "IMPORTED_RESULT"  # IMPORTED_RESULT, MANUAL_ENTRY, EXTERNAL_FILE
    status: str = "UNVERIFIED_IMPORT"  # UNVERIFIED_IMPORT, VERIFIED
    created_at: str = field(default_factory=current_iso_time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentResult":
        return cls(
            result_id=data.get("result_id", generate_id("res")),
            run_id=data.get("run_id", ""),
            model=data.get("model", ""),
            dataset=data.get("dataset", ""),
            metric=data.get("metric", ""),
            value=float(data.get("value", 0.0)),
            unit=data.get("unit", ""),
            split=data.get("split", "test"),
            seed=data.get("seed"),
            source=data.get("source", "IMPORTED_RESULT"),
            status=data.get("status", "UNVERIFIED_IMPORT"),
            created_at=data.get("created_at", current_iso_time()),
        )


@dataclass
class DatasetReference:
    """Documented reference to an empirical dataset with versioning and license."""

    dataset_id: str
    name: str
    version: str = "1.0"
    source_url: str = ""
    local_path: str = ""
    hash: str = ""
    license: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetReference":
        return cls(
            dataset_id=data.get("dataset_id", generate_id("ds")),
            name=data.get("name", ""),
            version=data.get("version", "1.0"),
            source_url=data.get("source_url", ""),
            local_path=data.get("local_path", ""),
            hash=data.get("hash", ""),
            license=data.get("license", ""),
            notes=data.get("notes", ""),
        )


@dataclass
class ModelReference:
    """Documented reference to a model architecture and checkpoints."""

    model_id: str
    name: str
    version: str = "1.0"
    source: str = ""
    commit: str = ""
    configuration: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelReference":
        return cls(
            model_id=data.get("model_id", generate_id("mod")),
            name=data.get("name", ""),
            version=data.get("version", "1.0"),
            source=data.get("source", ""),
            commit=data.get("commit", ""),
            configuration=data.get("configuration", {}),
            notes=data.get("notes", ""),
        )


@dataclass
class ResearchFinding:
    """Researcher-reviewed finding connecting empirical results to literature hypotheses."""

    finding_id: str
    workspace_id: str
    title: str
    statement: str
    finding_type: str = "RESULT"  # OBSERVATION, COMPARISON, RESULT, LIMITATION, NEGATIVE_RESULT, REPRODUCIBILITY_ISSUE
    evidence_ids: List[str] = field(default_factory=list)
    experiment_result_ids: List[str] = field(default_factory=list)
    related_hypothesis_id: Optional[str] = None
    confidence_level: str = "MEDIUM"  # HIGH, MEDIUM, LOW (Evidence organization label, not calibrated probability)
    status: str = "RESEARCHER_REVIEW_REQUIRED"  # RESEARCHER_REVIEW_REQUIRED, PENDING_RESEARCHER_REVIEW, CONFIRMED, REJECTED
    question_id: Optional[str] = None
    dataset_version: Optional[str] = None
    model_version: Optional[str] = None
    experiment_id: Optional[str] = None
    run_id: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    observed_metrics: Dict[str, Any] = field(default_factory=dict)
    baseline_comparison: Dict[str, Any] = field(default_factory=dict)
    reproducibility_status: str = "NOT_ASSESSED"
    supporting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    uncertainty_notes: str = ""
    created_at: str = field(default_factory=current_iso_time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchFinding":
        return cls(
            finding_id=data.get("finding_id", generate_id("fnd")),
            workspace_id=data.get("workspace_id", ""),
            title=data.get("title", ""),
            statement=data.get("statement", ""),
            finding_type=data.get("finding_type", "RESULT"),
            evidence_ids=data.get("evidence_ids", []),
            experiment_result_ids=data.get("experiment_result_ids", []),
            related_hypothesis_id=data.get("related_hypothesis_id"),
            confidence_level=data.get("confidence_level", "MEDIUM"),
            status=data.get("status", "RESEARCHER_REVIEW_REQUIRED"),
            question_id=data.get("question_id"),
            dataset_version=data.get("dataset_version"),
            model_version=data.get("model_version"),
            experiment_id=data.get("experiment_id"),
            run_id=data.get("run_id"),
            mlflow_run_id=data.get("mlflow_run_id"),
            observed_metrics=data.get("observed_metrics", {}),
            baseline_comparison=data.get("baseline_comparison", {}),
            reproducibility_status=data.get("reproducibility_status", "NOT_ASSESSED"),
            supporting_evidence=data.get("supporting_evidence", []),
            limitations=data.get("limitations", []),
            uncertainty_notes=data.get("uncertainty_notes", ""),
            created_at=data.get("created_at", current_iso_time()),
        )


@dataclass
class ReproducibilityReport:
    """Assessment of experiment reproducibility metadata completeness."""

    report_id: str
    run_id: str
    completeness_status: str = "INSUFFICIENT"  # COMPLETE, PARTIAL, INSUFFICIENT
    missing_fields: List[str] = field(default_factory=list)
    metadata_summary: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=current_iso_time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReproducibilityReport":
        return cls(
            report_id=data.get("report_id", generate_id("rep")),
            run_id=data.get("run_id", ""),
            completeness_status=data.get("completeness_status", "INSUFFICIENT"),
            missing_fields=data.get("missing_fields", []),
            metadata_summary=data.get("metadata_summary", {}),
            created_at=data.get("created_at", current_iso_time()),
        )


@dataclass
class ResearchReview:
    """Formal researcher review decision on an experiment, finding, hypothesis, or plan."""

    review_id: str
    target_type: str  # EXPERIMENT, FINDING, HYPOTHESIS, RESEARCH_PLAN
    target_id: str
    reviewer_id: str
    decision: str = "REQUIRES_REVISION"  # APPROVED, REQUIRES_REVISION, REJECTED
    comments: str = ""
    created_at: str = field(default_factory=current_iso_time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchReview":
        return cls(
            review_id=data.get("review_id", generate_id("rev")),
            target_type=data.get("target_type", ""),
            target_id=data.get("target_id", ""),
            reviewer_id=data.get("reviewer_id", ""),
            decision=data.get("decision", "REQUIRES_REVISION"),
            comments=data.get("comments", ""),
            created_at=data.get("created_at", current_iso_time()),
        )


@dataclass
class ActivityEvent:
    """Timeline entry tracking team events in a workspace."""

    event_id: str
    workspace_id: str
    actor_id: str
    action: str  # PAPER_ADDED, EVIDENCE_BOOKMARKED, QUESTION_CREATED, HYPOTHESIS_CREATED, EXPERIMENT_ADDED, RESULT_IMPORTED, FINDING_CREATED, REVIEW_COMPLETED
    target_type: str
    target_id: str
    summary: str
    timestamp: str = field(default_factory=current_iso_time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActivityEvent":
        return cls(
            event_id=data.get("event_id", generate_id("act")),
            workspace_id=data.get("workspace_id", ""),
            actor_id=data.get("actor_id", "system"),
            action=data.get("action", ""),
            target_type=data.get("target_type", ""),
            target_id=data.get("target_id", ""),
            summary=data.get("summary", ""),
            timestamp=data.get("timestamp", current_iso_time()),
        )


@dataclass
class AuditEvent:
    """Immutable audit trail event for workspace actions."""

    event_id: str
    workspace_id: str
    actor_id: str
    action: str  # CREATE, UPDATE, DELETE, IMPORT, REVIEW, EXPORT
    target_type: str
    target_id: str
    timestamp: str = field(default_factory=current_iso_time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditEvent":
        return cls(
            event_id=data.get("event_id", generate_id("aud")),
            workspace_id=data.get("workspace_id", ""),
            actor_id=data.get("actor_id", "system"),
            action=data.get("action", ""),
            target_type=data.get("target_type", ""),
            target_id=data.get("target_id", ""),
            timestamp=data.get("timestamp", current_iso_time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ExecutionProfile:
    """Hardware and environment profile for controlled experiment execution."""

    profile_id: str
    name: str
    python_version: str = "3.11"
    dependencies: Dict[str, str] = field(default_factory=dict)
    hardware: str = "Standard CPU"
    cpu: str = "4 Cores"
    memory: str = "16GB"
    gpu: str = "None"
    cuda_version: str = "N/A"
    environment_type: str = "LOCAL"  # LOCAL, VIRTUALENV, DOCKER
    created_at: str = field(default_factory=current_iso_time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionProfile":
        return cls(
            profile_id=data.get("profile_id", generate_id("prof")),
            name=data.get("name", "Default Profile"),
            python_version=data.get("python_version", "3.11"),
            dependencies=data.get("dependencies", {}),
            hardware=data.get("hardware", "Standard CPU"),
            cpu=data.get("cpu", "4 Cores"),
            memory=data.get("memory", "16GB"),
            gpu=data.get("gpu", "None"),
            cuda_version=data.get("cuda_version", "N/A"),
            environment_type=data.get("environment_type", "LOCAL"),
            created_at=data.get("created_at", current_iso_time()),
        )


@dataclass
class ExperimentRegistryEntry:
    """Explicitly registered experiment entrypoint allowed to execute."""

    experiment_id: str
    name: str
    description: str = ""
    entrypoint: str = ""
    version: str = "1.0"
    owner: str = ""
    allowed: bool = True
    command_template: str = ""
    default_timeout_seconds: int = 3600
    run_count: int = 0
    created_at: str = field(default_factory=current_iso_time)
    updated_at: str = field(default_factory=current_iso_time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentRegistryEntry":
        return cls(
            experiment_id=data.get("experiment_id", generate_id("reg")),
            name=data.get("name", "Untitled Experiment"),
            description=data.get("description", ""),
            entrypoint=data.get("entrypoint", ""),
            version=data.get("version", "1.0"),
            owner=data.get("owner", ""),
            allowed=bool(data.get("allowed", True)),
            command_template=data.get("command_template", ""),
            default_timeout_seconds=int(data.get("default_timeout_seconds", 3600)),
            run_count=int(data.get("run_count", 0)),
            created_at=data.get("created_at", current_iso_time()),
            updated_at=data.get("updated_at", current_iso_time()),
        )


@dataclass
class ExperimentConfiguration:
    """Versioned parameter snapshot used for a specific experiment execution run."""

    configuration_id: str
    experiment_id: str
    version: int = 1
    parameters: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=current_iso_time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfiguration":
        return cls(
            configuration_id=data.get("configuration_id", generate_id("cfg")),
            experiment_id=data.get("experiment_id", ""),
            version=int(data.get("version", 1)),
            parameters=data.get("parameters", {}),
            created_at=data.get("created_at", current_iso_time()),
        )


@dataclass
class ExperimentArtifact:
    """Registered output artifact generated by an experiment run with integrity hash."""

    artifact_id: str
    run_id: str
    name: str
    path: str
    type: str = "METRICS"  # MODEL, METRICS, PLOT, TABLE, LOG, CONFIG, PREDICTION, REPORT
    size_bytes: int = 0
    sha256: str = ""
    created_at: str = field(default_factory=current_iso_time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentArtifact":
        return cls(
            artifact_id=data.get("artifact_id", generate_id("art")),
            run_id=data.get("run_id", ""),
            name=data.get("name", ""),
            path=data.get("path", ""),
            type=data.get("type", "METRICS"),
            size_bytes=int(data.get("size_bytes", 0)),
            sha256=data.get("sha256", ""),
            created_at=data.get("created_at", current_iso_time()),
        )


@dataclass
class ExecutionQueueItem:
    """Execution queue record for scheduled or running experiment jobs."""

    queue_id: str
    run_id: str
    priority: int = 1
    status: str = "QUEUED"  # QUEUED, PREPARING, RUNNING, COMPLETED, FAILED, TIMEOUT, CANCELLED, BLOCKED
    job_id: Optional[str] = None
    experiment_id: Optional[str] = None
    backend: str = "LOCAL"
    created_at: str = field(default_factory=current_iso_time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionQueueItem":
        return cls(
            queue_id=data.get("queue_id", generate_id("q")),
            run_id=data.get("run_id", ""),
            priority=int(data.get("priority", 1)),
            status=data.get("status", "QUEUED"),
            job_id=data.get("job_id"),
            experiment_id=data.get("experiment_id"),
            backend=data.get("backend", "LOCAL"),
            created_at=data.get("created_at", current_iso_time()),
        )


@dataclass
class ResearchModelRegistration:
    """Registered research model with version, lineage, stage, and artifact provenance."""

    model_id: str
    run_id: str
    name: str
    version: str = "1.0.0"
    stage: str = "NONE"  # NONE, CANDIDATE, VALIDATED, PRODUCTION, ARCHIVED
    artifact_uri: str = ""
    dataset: str = ""
    metric: str = ""
    metric_value: float = 0.0
    artifact_hash: str = ""
    code_commit: str = "NOT_AVAILABLE"
    reviewed_by: Optional[str] = None
    notes: str = ""
    mlflow_model_name: Optional[str] = None
    mlflow_model_version: Optional[str] = None
    execution_image: Optional[str] = None
    image_digest: Optional[str] = None
    created_at: str = field(default_factory=current_iso_time)
    updated_at: str = field(default_factory=current_iso_time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchModelRegistration":
        return cls(
            model_id=data.get("model_id", generate_id("reg_mod")),
            run_id=data.get("run_id", ""),
            name=data.get("name", ""),
            version=data.get("version", "1.0.0"),
            stage=data.get("stage", "NONE"),
            artifact_uri=data.get("artifact_uri", ""),
            dataset=data.get("dataset", ""),
            metric=data.get("metric", ""),
            metric_value=float(data.get("metric_value", 0.0)),
            artifact_hash=data.get("artifact_hash", ""),
            code_commit=data.get("code_commit", "NOT_AVAILABLE"),
            reviewed_by=data.get("reviewed_by"),
            notes=data.get("notes", ""),
            mlflow_model_name=data.get("mlflow_model_name"),
            mlflow_model_version=data.get("mlflow_model_version"),
            execution_image=data.get("execution_image"),
            image_digest=data.get("image_digest"),
            created_at=data.get("created_at", current_iso_time()),
            updated_at=data.get("updated_at", current_iso_time()),
        )


# ==============================================================================
# Phase 14: Remote MLOps, Container Isolation & Reproducibility Schemas
# ==============================================================================


@dataclass
class ExecutionJob:
    """Formal specification of an executable experiment job (execution_job.json)."""

    job_id: str
    workspace_id: str
    experiment_id: str
    run_id: str
    backend: str = "LOCAL"  # LOCAL, DOCKER, REMOTE
    image: Optional[str] = None
    command: List[str] = field(default_factory=list)
    resources: Dict[str, Any] = field(default_factory=dict)  # cpu_limit, memory_limit, timeout
    environment: Dict[str, str] = field(default_factory=dict)
    network_policy: str = "DISABLED"  # DISABLED, ENABLED
    timeout: int = 3600
    status: str = "CREATED"  # CREATED, VALIDATING, QUEUED, PREPARING, RUNNING, UPLOADING, COMPLETED, FAILED, TIMEOUT, CANCELLED, BLOCKED
    created_at: str = field(default_factory=current_iso_time)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    transitions: List[Dict[str, str]] = field(default_factory=list)
    error_message: Optional[str] = None

    def transition_to(self, new_status: str, reason: str = "") -> None:
        """Persist state transitions in lifecycle trajectory."""
        now = current_iso_time()
        self.status = new_status
        self.transitions.append({
            "from_status": self.status,
            "to_status": new_status,
            "timestamp": now,
            "reason": reason,
        })
        if new_status == "RUNNING" and not self.started_at:
            self.started_at = now
        elif new_status in ["COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "BLOCKED"]:
            self.completed_at = now
            if reason and not self.error_message:
                self.error_message = reason

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionJob":
        return cls(
            job_id=data.get("job_id", generate_id("job")),
            workspace_id=data.get("workspace_id", ""),
            experiment_id=data.get("experiment_id", ""),
            run_id=data.get("run_id", ""),
            backend=data.get("backend", "LOCAL"),
            image=data.get("image"),
            command=data.get("command", []),
            resources=data.get("resources", {}),
            environment=data.get("environment", {}),
            network_policy=data.get("network_policy", "DISABLED"),
            timeout=int(data.get("timeout", 3600)),
            status=data.get("status", "CREATED"),
            created_at=data.get("created_at", current_iso_time()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            transitions=data.get("transitions", []),
            error_message=data.get("error_message"),
        )


@dataclass
class ExecutionImage:
    """Docker image specification for containerized experiment execution."""

    image_id: str
    name: str
    tag: str = "latest"
    digest: str = "NOT_PROVIDED"
    python_version: str = "3.11"
    dependencies: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=current_iso_time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionImage":
        return cls(
            image_id=data.get("image_id", generate_id("img")),
            name=data.get("name", ""),
            tag=data.get("tag", "latest"),
            digest=data.get("digest", "NOT_PROVIDED"),
            python_version=data.get("python_version", "3.11"),
            dependencies=data.get("dependencies", []),
            created_at=data.get("created_at", current_iso_time()),
        )


@dataclass
class ExecutionImageRegistryEntry:
    """Security allowlist entry for approved container images."""

    image: str
    digest: str = "NOT_PROVIDED"
    allowed: bool = True
    owner: str = "admin"
    description: str = ""
    created_at: str = field(default_factory=current_iso_time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionImageRegistryEntry":
        return cls(
            image=data.get("image", ""),
            digest=data.get("digest", "NOT_PROVIDED"),
            allowed=data.get("allowed", True),
            owner=data.get("owner", "admin"),
            description=data.get("description", ""),
            created_at=data.get("created_at", current_iso_time()),
        )


@dataclass
class DatasetMount:
    """Read-only container dataset mount definition."""

    dataset_id: str
    mount_path: str
    read_only: bool = True
    hash: str = "NOT_PROVIDED"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetMount":
        return cls(
            dataset_id=data.get("dataset_id", ""),
            mount_path=data.get("mount_path", ""),
            read_only=data.get("read_only", True),
            hash=data.get("hash", "NOT_PROVIDED"),
        )


@dataclass
class SourceSnapshot:
    """Git commit provenance and source snapshot."""

    repository: str = "NOT_AVAILABLE"
    branch: str = "NOT_AVAILABLE"
    commit: str = "NOT_AVAILABLE"
    dirty: bool = False
    snapshot_hash: str = "NOT_AVAILABLE"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceSnapshot":
        return cls(
            repository=data.get("repository", "NOT_AVAILABLE"),
            branch=data.get("branch", "NOT_AVAILABLE"),
            commit=data.get("commit", "NOT_AVAILABLE"),
            dirty=data.get("dirty", False),
            snapshot_hash=data.get("snapshot_hash", "NOT_AVAILABLE"),
        )


@dataclass
class EvaluationContract:
    """Standard evaluation contract (evaluation.json)."""

    metrics: Dict[str, Any] = field(default_factory=dict)
    baseline: Dict[str, Any] = field(default_factory=dict)
    comparability: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationContract":
        return cls(
            metrics=data.get("metrics", {}),
            baseline=data.get("baseline", {}),
            comparability=data.get("comparability", {}),
            warnings=data.get("warnings", []),
        )


@dataclass
class ResearchWorkflowState:
    """End-to-end evidence-grounded research automation workflow state (Phase 15)."""

    workflow_id: str
    workspace_id: str
    title: str
    topic: str = ""
    current_stage: str = "LITERATURE_DISCOVERY"
    status: str = "IN_PROGRESS"  # IN_PROGRESS, AWAITING_APPROVAL, APPROVED, REJECTED, COMPLETED, FAILED, CANCELLED
    approval_status: str = "PENDING"  # PENDING, APPROVED, REJECTED
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    approval_notes: str = ""
    stage_data: Dict[str, Any] = field(default_factory=dict)
    stage_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=current_iso_time)
    updated_at: str = field(default_factory=current_iso_time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchWorkflowState":
        return cls(
            workflow_id=data.get("workflow_id", generate_id("wf")),
            workspace_id=data.get("workspace_id", ""),
            title=data.get("title", "Untitled Workflow"),
            topic=data.get("topic", ""),
            current_stage=data.get("current_stage", "LITERATURE_DISCOVERY"),
            status=data.get("status", "IN_PROGRESS"),
            approval_status=data.get("approval_status", "PENDING"),
            approved_by=data.get("approved_by"),
            approved_at=data.get("approved_at"),
            approval_notes=data.get("approval_notes", ""),
            stage_data=data.get("stage_data", {}),
            stage_history=data.get("stage_history", []),
            created_at=data.get("created_at", current_iso_time()),
            updated_at=data.get("updated_at", current_iso_time()),
        )

