"""Atomic file-based persistence and search store for ResearchLens Workspaces (Phase 10)."""

import json
import os
import re
import tempfile
from typing import Any, Dict, List, Optional

from researchlens.utils.logging import get_logger
from researchlens.workspace.schema import (
    ActivityEvent,
    AuditEvent,
    DatasetReference,
    DiscussionMessage,
    DiscussionThread,
    EvidenceBookmark,
    ExecutionImageRegistryEntry,
    ExecutionJob,
    ExecutionProfile,
    ExecutionQueueItem,
    ExperimentArtifact,
    ExperimentConfiguration,
    ExperimentPlan,
    ExperimentRegistryEntry,
    ExperimentResult,
    ExperimentRun,
    ModelReference,
    PaperCollection,
    ReproducibilityReport,
    ResearchAnnotation,
    ResearchComment,
    ResearchDirection,
    ResearchFinding,
    ResearchGap,
    ResearchHypothesis,
    ResearchMilestone,
    ResearchModelRegistration,
    ResearchPlan,
    ResearchQuestion,
    ResearchReview,
    ResearchRisk,
    ResearchWorkflowState,
    ResearchWorkspace,
    SavedPaper,
    WorkspaceMember,
    current_iso_time,
    generate_id,
)

logger = get_logger("workspace_store")


def _sanitize_id(identifier: str) -> str:
    """Validate and sanitize identifiers against path traversal attacks."""
    if not identifier or not re.match(r"^[a-zA-Z0-9_\-]+$", identifier):
        raise ValueError(f"Invalid identifier for workspace storage: '{identifier}'")
    return identifier


def _atomic_write_json(file_path: str, data: Any) -> None:
    """Atomically write JSON data to disk using a temporary file and replace."""
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    temp_fd, temp_path = tempfile.mkstemp(dir=directory, prefix=".tmp_ws_")
    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        # On Windows, replace will overwrite atomically if destination exists
        os.replace(temp_path, file_path)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error(f"Failed atomic write to {file_path}: {e}")
        raise


def _read_json(file_path: str, default: Any = None) -> Any:
    """Read JSON from file safely, returning default on missing or empty file."""
    if not os.path.exists(file_path):
        return default
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Error reading JSON from {file_path}: {e}")
        return default


class WorkspaceStore:
    """File-backed local persistence engine for workspaces."""

    def __init__(self, base_dir: str = "data/workspaces") -> None:
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def _workspace_dir(self, workspace_id: str) -> str:
        safe_id = _sanitize_id(workspace_id)
        path = os.path.join(self.base_dir, safe_id)
        os.makedirs(path, exist_ok=True)
        return path

    # --- Workspace CRUD ---

    def list_workspaces(self) -> List[ResearchWorkspace]:
        """List all workspaces stored on disk."""
        workspaces: List[ResearchWorkspace] = []
        if not os.path.exists(self.base_dir):
            return workspaces

        for entry in os.listdir(self.base_dir):
            dir_path = os.path.join(self.base_dir, entry)
            if os.path.isdir(dir_path):
                ws_file = os.path.join(dir_path, "workspace.json")
                data = _read_json(ws_file)
                if data and isinstance(data, dict):
                    workspaces.append(ResearchWorkspace.from_dict(data))
        return sorted(workspaces, key=lambda w: w.created_at, reverse=True)

    def create_workspace(
        self,
        name: str,
        description: str = "",
        topic_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> ResearchWorkspace:
        """Create a new workspace."""
        ws_id = _sanitize_id(workspace_id) if workspace_id else generate_id("ws")
        ws = ResearchWorkspace(
            workspace_id=ws_id,
            name=name,
            description=description,
            topic_id=topic_id,
        )
        self.save_workspace(ws)
        return ws

    def get_workspace(self, workspace_id: str) -> Optional[ResearchWorkspace]:
        """Retrieve a workspace by ID."""
        try:
            ws_dir = self._workspace_dir(workspace_id)
            ws_file = os.path.join(ws_dir, "workspace.json")
            data = _read_json(ws_file)
            return ResearchWorkspace.from_dict(data) if data else None
        except Exception as e:
            logger.error(f"Error getting workspace {workspace_id}: {e}")
            return None

    def save_workspace(self, workspace: ResearchWorkspace) -> None:
        """Save or update workspace metadata."""
        workspace.updated_at = current_iso_time()
        ws_dir = self._workspace_dir(workspace.workspace_id)
        ws_file = os.path.join(ws_dir, "workspace.json")
        _atomic_write_json(ws_file, workspace.to_dict())

    def delete_workspace(self, workspace_id: str) -> bool:
        """Delete an entire workspace folder and all its contents."""
        try:
            ws_dir = self._workspace_dir(workspace_id)
            if os.path.exists(ws_dir):
                import shutil
                shutil.rmtree(ws_dir)
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting workspace {workspace_id}: {e}")
            return False

    # --- Generic Entity Helpers ---

    def _load_list(self, workspace_id: str, filename: str) -> List[Dict[str, Any]]:
        ws_dir = self._workspace_dir(workspace_id)
        file_path = os.path.join(ws_dir, filename)
        return _read_json(file_path, default=[])

    def _save_list(self, workspace_id: str, filename: str, data: List[Dict[str, Any]]) -> None:
        ws_dir = self._workspace_dir(workspace_id)
        file_path = os.path.join(ws_dir, filename)
        _atomic_write_json(file_path, data)

    # --- Collections ---

    def list_collections(self, workspace_id: str) -> List[PaperCollection]:
        raw = self._load_list(workspace_id, "collections.json")
        return [PaperCollection.from_dict(item) for item in raw]

    def save_collection(self, collection: PaperCollection) -> None:
        collections = self.list_collections(collection.workspace_id)
        idx = next((i for i, c in enumerate(collections) if c.collection_id == collection.collection_id), None)
        if idx is not None:
            collections[idx] = collection
        else:
            collections.append(collection)
        self._save_list(collection.workspace_id, "collections.json", [c.to_dict() for c in collections])

    def get_collection(self, workspace_id: str, collection_id: str) -> Optional[PaperCollection]:
        collections = self.list_collections(workspace_id)
        return next((c for c in collections if c.collection_id == collection_id), None)

    def delete_collection(self, workspace_id: str, collection_id: str) -> bool:
        collections = self.list_collections(workspace_id)
        filtered = [c for c in collections if c.collection_id != collection_id]
        if len(filtered) != len(collections):
            self._save_list(workspace_id, "collections.json", [c.to_dict() for c in filtered])
            return True
        return False

    def add_paper_to_collection(self, workspace_id: str, collection_id: str, paper_id: str) -> bool:
        collections = self.list_collections(workspace_id)
        target = next((c for c in collections if c.collection_id == collection_id), None)
        if target and paper_id not in target.paper_ids:
            target.paper_ids.append(paper_id)
            self._save_list(workspace_id, "collections.json", [c.to_dict() for c in collections])
            return True
        return False

    def remove_paper_from_collection(self, workspace_id: str, collection_id: str, paper_id: str) -> bool:
        collections = self.list_collections(workspace_id)
        target = next((c for c in collections if c.collection_id == collection_id), None)
        if target and paper_id in target.paper_ids:
            target.paper_ids.remove(paper_id)
            self._save_list(workspace_id, "collections.json", [c.to_dict() for c in collections])
            return True
        return False

    # --- Saved Papers ---

    def list_papers(self, workspace_id: str) -> List[SavedPaper]:
        raw = self._load_list(workspace_id, "papers.json")
        return [SavedPaper.from_dict(item) for item in raw]

    def get_paper(self, workspace_id: str, paper_id: str) -> Optional[SavedPaper]:
        papers = self.list_papers(workspace_id)
        return next((p for p in papers if p.paper_id == paper_id), None)

    def save_paper(self, paper: SavedPaper) -> None:
        papers = self.list_papers(paper.workspace_id)
        idx = next((i for i, p in enumerate(papers) if p.paper_id == paper.paper_id), None)
        if idx is not None:
            papers[idx] = paper
        else:
            papers.append(paper)
        self._save_list(paper.workspace_id, "papers.json", [p.to_dict() for p in papers])

    def delete_paper(self, workspace_id: str, paper_id: str) -> bool:

        papers = self.list_papers(workspace_id)
        filtered = [p for p in papers if p.paper_id != paper_id]
        if len(filtered) != len(papers):
            self._save_list(workspace_id, "papers.json", [p.to_dict() for p in filtered])
            return True
        return False

    # --- Evidence Bookmarks ---

    def list_evidence(self, workspace_id: str) -> List[EvidenceBookmark]:
        raw = self._load_list(workspace_id, "evidence.json")
        return [EvidenceBookmark.from_dict(item) for item in raw]

    def save_evidence(self, bookmark: EvidenceBookmark) -> None:
        bookmarks = self.list_evidence(bookmark.workspace_id)
        idx = next((i for i, b in enumerate(bookmarks) if b.bookmark_id == bookmark.bookmark_id), None)
        if idx is not None:
            bookmarks[idx] = bookmark
        else:
            bookmarks.append(bookmark)
        self._save_list(bookmark.workspace_id, "evidence.json", [b.to_dict() for b in bookmarks])

    def delete_evidence(self, workspace_id: str, bookmark_id: str) -> bool:
        bookmarks = self.list_evidence(workspace_id)
        filtered = [b for b in bookmarks if b.bookmark_id != bookmark_id]
        if len(filtered) != len(bookmarks):
            self._save_list(workspace_id, "evidence.json", [b.to_dict() for b in filtered])
            return True
        return False

    # --- Research Annotations ---

    def list_annotations(self, workspace_id: str) -> List[ResearchAnnotation]:
        raw = self._load_list(workspace_id, "annotations.json")
        return [ResearchAnnotation.from_dict(item) for item in raw]

    def save_annotation(self, annotation: ResearchAnnotation) -> None:
        annotations = self.list_annotations(annotation.workspace_id)
        idx = next((i for i, a in enumerate(annotations) if a.annotation_id == annotation.annotation_id), None)
        if idx is not None:
            annotations[idx] = annotation
        else:
            annotations.append(annotation)
        self._save_list(annotation.workspace_id, "annotations.json", [a.to_dict() for a in annotations])

    def delete_annotation(self, workspace_id: str, annotation_id: str) -> bool:
        annotations = self.list_annotations(workspace_id)
        filtered = [a for a in annotations if a.annotation_id != annotation_id]
        if len(filtered) != len(annotations):
            self._save_list(workspace_id, "annotations.json", [a.to_dict() for a in filtered])
            return True
        return False

    # --- Research Gaps ---

    def list_gaps(self, workspace_id: str) -> List[ResearchGap]:
        raw = self._load_list(workspace_id, "gaps.json")
        return [ResearchGap.from_dict(item) for item in raw]

    def save_gap(self, gap: ResearchGap, workspace_id: Optional[str] = None) -> None:
        ws_id = workspace_id or getattr(gap, "workspace_id", "") or "default"
        gaps = self.list_gaps(ws_id)
        idx = next((i for i, g in enumerate(gaps) if g.gap_id == gap.gap_id), None)
        if idx is not None:
            gaps[idx] = gap
        else:
            gaps.append(gap)
        self._save_list(ws_id, "gaps.json", [g.to_dict() for g in gaps])

    def delete_gap(self, workspace_id: str, gap_id: str) -> bool:
        gaps = self.list_gaps(workspace_id)
        filtered = [g for g in gaps if g.gap_id != gap_id]
        if len(filtered) != len(gaps):
            self._save_list(workspace_id, "gaps.json", [g.to_dict() for g in filtered])
            return True
        return False

    # --- Research Questions ---

    def list_questions(self, workspace_id: str) -> List[ResearchQuestion]:
        raw = self._load_list(workspace_id, "questions.json")
        return [ResearchQuestion.from_dict(item) for item in raw]

    def get_question(self, workspace_id: str, question_id: str) -> Optional[ResearchQuestion]:
        questions = self.list_questions(workspace_id)
        return next((q for q in questions if q.question_id == question_id), None)

    def save_question(self, question: ResearchQuestion) -> None:
        questions = self.list_questions(question.workspace_id)
        question.updated_at = current_iso_time()
        idx = next((i for i, q in enumerate(questions) if q.question_id == question.question_id), None)
        if idx is not None:
            questions[idx] = question
        else:
            questions.append(question)
        self._save_list(question.workspace_id, "questions.json", [q.to_dict() for q in questions])

    def delete_question(self, workspace_id: str, question_id: str) -> bool:
        questions = self.list_questions(workspace_id)
        filtered = [q for q in questions if q.question_id != question_id]
        if len(filtered) != len(questions):
            self._save_list(workspace_id, "questions.json", [q.to_dict() for q in filtered])
            return True
        return False

    # --- Hypotheses ---

    def list_hypotheses(self, workspace_id: str) -> List[ResearchHypothesis]:
        raw = self._load_list(workspace_id, "hypotheses.json")
        return [ResearchHypothesis.from_dict(item) for item in raw]

    def get_hypothesis(self, workspace_id: str, hypothesis_id: str) -> Optional[ResearchHypothesis]:
        hypotheses = self.list_hypotheses(workspace_id)
        return next((h for h in hypotheses if h.hypothesis_id == hypothesis_id), None)

    def save_hypothesis(self, hypothesis: ResearchHypothesis) -> None:

        hypotheses = self.list_hypotheses(hypothesis.workspace_id)
        idx = next((i for i, h in enumerate(hypotheses) if h.hypothesis_id == hypothesis.hypothesis_id), None)
        if idx is not None:
            hypotheses[idx] = hypothesis
        else:
            hypotheses.append(hypothesis)
        self._save_list(hypothesis.workspace_id, "hypotheses.json", [h.to_dict() for h in hypotheses])

    def delete_hypothesis(self, workspace_id: str, hypothesis_id: str) -> bool:
        hypotheses = self.list_hypotheses(workspace_id)
        filtered = [h for h in hypotheses if h.hypothesis_id != hypothesis_id]
        if len(filtered) != len(hypotheses):
            self._save_list(workspace_id, "hypotheses.json", [h.to_dict() for h in filtered])
            return True
        return False

    # --- Research Directions ---

    def list_directions(self, workspace_id: str) -> List[ResearchDirection]:
        raw = self._load_list(workspace_id, "directions.json")
        return [ResearchDirection.from_dict(item) for item in raw]

    def save_direction(self, direction: ResearchDirection) -> None:
        directions = self.list_directions(direction.workspace_id)
        idx = next((i for i, d in enumerate(directions) if d.direction_id == direction.direction_id), None)
        if idx is not None:
            directions[idx] = direction
        else:
            directions.append(direction)
        self._save_list(direction.workspace_id, "directions.json", [d.to_dict() for d in directions])

    def delete_direction(self, workspace_id: str, direction_id: str) -> bool:
        directions = self.list_directions(workspace_id)
        filtered = [d for d in directions if d.direction_id != direction_id]
        if len(filtered) != len(directions):
            self._save_list(workspace_id, "directions.json", [d.to_dict() for d in filtered])
            return True
        return False

    # --- Plans ---

    def list_plans(self, workspace_id: str) -> List[ResearchPlan]:
        raw = self._load_list(workspace_id, "plans.json")
        return [ResearchPlan.from_dict(item) for item in raw]

    def save_plan(self, plan: ResearchPlan) -> None:
        plans = self.list_plans(plan.workspace_id)
        plan.updated_at = current_iso_time()
        idx = next((i for i, p in enumerate(plans) if p.plan_id == plan.plan_id), None)
        if idx is not None:
            plans[idx] = plan
        else:
            plans.append(plan)
        self._save_list(plan.workspace_id, "plans.json", [p.to_dict() for p in plans])

    def delete_plan(self, workspace_id: str, plan_id: str) -> bool:
        plans = self.list_plans(workspace_id)
        filtered = [p for p in plans if p.plan_id != plan_id]
        if len(filtered) != len(plans):
            self._save_list(workspace_id, "plans.json", [p.to_dict() for p in filtered])
            return True
        return False

    # --- Experiments ---

    def list_experiments(self, workspace_id: str) -> List[ExperimentPlan]:
        raw = self._load_list(workspace_id, "experiments.json")
        return [ExperimentPlan.from_dict(item) for item in raw]

    def get_experiment(self, workspace_id: str, experiment_id: str) -> Optional[ExperimentPlan]:
        experiments = self.list_experiments(workspace_id)
        return next((e for e in experiments if e.experiment_id == experiment_id), None)

    def save_experiment(self, experiment: ExperimentPlan, workspace_id: str) -> None:
        experiments = self.list_experiments(workspace_id)
        idx = next((i for i, e in enumerate(experiments) if e.experiment_id == experiment.experiment_id), None)
        if idx is not None:
            experiments[idx] = experiment
        else:
            experiments.append(experiment)
        self._save_list(workspace_id, "experiments.json", [e.to_dict() for e in experiments])

    def delete_experiment(self, workspace_id: str, experiment_id: str) -> bool:
        experiments = self.list_experiments(workspace_id)
        filtered = [e for e in experiments if e.experiment_id != experiment_id]
        if len(filtered) != len(experiments):
            self._save_list(workspace_id, "experiments.json", [e.to_dict() for e in filtered])
            return True
        return False

    # --- Risks ---

    def list_risks(self, workspace_id: str) -> List[ResearchRisk]:
        raw = self._load_list(workspace_id, "risks.json")
        return [ResearchRisk.from_dict(item) for item in raw]

    def save_risk(self, risk: ResearchRisk) -> None:
        risks = self.list_risks(risk.workspace_id)
        idx = next((i for i, r in enumerate(risks) if r.risk_id == risk.risk_id), None)
        if idx is not None:
            risks[idx] = risk
        else:
            risks.append(risk)
        self._save_list(risk.workspace_id, "risks.json", [r.to_dict() for r in risks])

    def delete_risk(self, workspace_id: str, risk_id: str) -> bool:
        risks = self.list_risks(workspace_id)
        filtered = [r for r in risks if r.risk_id != risk_id]
        if len(filtered) != len(risks):
            self._save_list(workspace_id, "risks.json", [r.to_dict() for r in filtered])
            return True
        return False

    # --- Milestones ---

    def list_milestones(self, workspace_id: str) -> List[ResearchMilestone]:
        raw = self._load_list(workspace_id, "milestones.json")
        return [ResearchMilestone.from_dict(item) for item in raw]

    def save_milestone(self, milestone: ResearchMilestone) -> None:
        milestones = self.list_milestones(milestone.workspace_id)
        idx = next((i for i, m in enumerate(milestones) if m.milestone_id == milestone.milestone_id), None)
        if idx is not None:
            milestones[idx] = milestone
        else:
            milestones.append(milestone)
        self._save_list(milestone.workspace_id, "milestones.json", [m.to_dict() for m in milestones])

    def delete_milestone(self, workspace_id: str, milestone_id: str) -> bool:
        milestones = self.list_milestones(workspace_id)
        filtered = [m for m in milestones if m.milestone_id != milestone_id]
        if len(filtered) != len(milestones):
            self._save_list(workspace_id, "milestones.json", [m.to_dict() for m in filtered])
            return True
        return False

    # --- Workspace Search ---

    def search_workspace(self, workspace_id: str, query: str) -> Dict[str, List[Any]]:
        """Search across papers, evidence, annotations, questions, hypotheses, and directions."""
        q = query.strip().lower()
        if not q:
            return {
                "papers": [],
                "evidence": [],
                "annotations": [],
                "questions": [],
                "hypotheses": [],
                "directions": [],
            }

        papers = [p for p in self.list_papers(workspace_id) if q in p.title.lower() or q in p.notes.lower() or any(q in t.lower() for t in p.tags)]
        evidence = [e for e in self.list_evidence(workspace_id) if q in e.source_text.lower() or q in e.label.lower() or q in e.note.lower()]
        annotations = [a for a in self.list_annotations(workspace_id) if q in a.content.lower() or q in a.source_text.lower() or q in a.annotation_type.lower()]
        questions = [rq for rq in self.list_questions(workspace_id) if q in rq.text.lower()]
        hypotheses = [h for h in self.list_hypotheses(workspace_id) if q in h.statement.lower() or any(q in v.lower() for v in h.independent_variables + h.dependent_variables)]
        directions = [d for d in self.list_directions(workspace_id) if q in d.title.lower() or q in d.description.lower()]

        return {
            "papers": papers,
            "evidence": evidence,
            "annotations": annotations,
            "questions": questions,
            "hypotheses": hypotheses,
            "directions": directions,
        }

    # --- Collaboration: Members & Permissions ---

    def list_members(self, workspace_id: str) -> List[WorkspaceMember]:
        raw = self._load_list(workspace_id, "members.json")
        return [WorkspaceMember.from_dict(item) for item in raw]

    def save_member(self, member: WorkspaceMember) -> None:
        members = self.list_members(member.workspace_id)
        idx = next((i for i, m in enumerate(members) if m.member_id == member.member_id), None)
        if idx is not None:
            members[idx] = member
        else:
            members.append(member)
        self._save_list(member.workspace_id, "members.json", [m.to_dict() for m in members])

    def delete_member(self, workspace_id: str, member_id: str) -> bool:
        members = self.list_members(workspace_id)
        filtered = [m for m in members if m.member_id != member_id]
        if len(filtered) != len(members):
            self._save_list(workspace_id, "members.json", [m.to_dict() for m in filtered])
            return True
        return False

    def get_member(self, workspace_id: str, member_id: str) -> Optional[WorkspaceMember]:
        members = self.list_members(workspace_id)
        return next((m for m in members if m.member_id == member_id), None)

    def check_permission(self, workspace_id: str, member_id: str, required_role: str) -> bool:
        """Lightweight permission checking.

        Hierarchy: OWNER > RESEARCHER > REVIEWER > VIEWER
        """
        hierarchy = {"OWNER": 4, "RESEARCHER": 3, "REVIEWER": 2, "VIEWER": 1}
        member = self.get_member(workspace_id, member_id)
        if not member or member.status != "ACTIVE":
            # If no members exist yet in workspace, default allow owner
            if not self.list_members(workspace_id):
                return True
            return False

        user_level = hierarchy.get(member.role, 0)
        req_level = hierarchy.get(required_role, 0)
        return user_level >= req_level

    def has_permission(self, workspace_id: str, member_id: str, required_role: str) -> bool:
        """Alias for check_permission."""
        return self.check_permission(workspace_id, member_id, required_role)

    # --- Comments ---

    def list_comments(self, workspace_id: str) -> List[ResearchComment]:
        raw = self._load_list(workspace_id, "comments.json")
        return [ResearchComment.from_dict(item) for item in raw]

    def save_comment(self, comment: ResearchComment) -> None:
        comments = self.list_comments(comment.workspace_id)
        idx = next((i for i, c in enumerate(comments) if c.comment_id == comment.comment_id), None)
        if idx is not None:
            comments[idx] = comment
        else:
            comments.append(comment)
        self._save_list(comment.workspace_id, "comments.json", [c.to_dict() for c in comments])

    def delete_comment(self, workspace_id: str, comment_id: str) -> bool:
        comments = self.list_comments(workspace_id)
        filtered = [c for c in comments if c.comment_id != comment_id]
        if len(filtered) != len(comments):
            self._save_list(workspace_id, "comments.json", [c.to_dict() for c in filtered])
            return True
        return False

    def get_comments_for_target(self, workspace_id: str, target_type: str, target_id: str) -> List[ResearchComment]:
        comments = self.list_comments(workspace_id)
        return [c for c in comments if c.target_type == target_type and c.target_id == target_id]

    # --- Discussion Threads ---

    def list_threads(self, workspace_id: str) -> List[DiscussionThread]:
        raw = self._load_list(workspace_id, "threads.json")
        return [DiscussionThread.from_dict(item) for item in raw]

    def save_thread(self, thread: DiscussionThread) -> None:
        threads = self.list_threads(thread.workspace_id)
        thread.updated_at = current_iso_time()
        idx = next((i for i, t in enumerate(threads) if t.thread_id == thread.thread_id), None)
        if idx is not None:
            threads[idx] = thread
        else:
            threads.append(thread)
        self._save_list(thread.workspace_id, "threads.json", [t.to_dict() for t in threads])

    def delete_thread(self, workspace_id: str, thread_id: str) -> bool:
        threads = self.list_threads(workspace_id)
        filtered = [t for t in threads if t.thread_id != thread_id]
        if len(filtered) != len(threads):
            self._save_list(workspace_id, "threads.json", [t.to_dict() for t in filtered])
            return True
        return False

    def add_message_to_thread(self, workspace_id: str, thread_id: str, message: DiscussionMessage) -> bool:
        threads = self.list_threads(workspace_id)
        target = next((t for t in threads if t.thread_id == thread_id), None)
        if target:
            target.messages.append(message)
            target.updated_at = current_iso_time()
            self._save_list(workspace_id, "threads.json", [t.to_dict() for t in threads])
            return True
        return False

    # --- Experiment Runs ---

    def list_runs(self, workspace_id: str) -> List[ExperimentRun]:
        raw = self._load_list(workspace_id, "runs.json")
        return [ExperimentRun.from_dict(item) for item in raw]

    def save_run(self, run: ExperimentRun, workspace_id: str) -> None:
        runs = self.list_runs(workspace_id)
        idx = next((i for i, r in enumerate(runs) if r.run_id == run.run_id), None)
        if idx is not None:
            runs[idx] = run
        else:
            runs.append(run)
        self._save_list(workspace_id, "runs.json", [r.to_dict() for r in runs])

    def get_run(self, workspace_id: str, run_id: str) -> Optional[ExperimentRun]:
        runs = self.list_runs(workspace_id)
        return next((r for r in runs if r.run_id == run_id), None)

    def delete_run(self, workspace_id: str, run_id: str) -> bool:
        runs = self.list_runs(workspace_id)
        filtered = [r for r in runs if r.run_id != run_id]
        if len(filtered) != len(runs):
            self._save_list(workspace_id, "runs.json", [r.to_dict() for r in filtered])
            return True
        return False

    def list_results(self, workspace_id: str) -> List[ExperimentResult]:
        raw = self._load_list(workspace_id, "results.json")
        return [ExperimentResult.from_dict(item) for item in raw]

    def get_result(self, workspace_id: str, result_id: str) -> Optional[ExperimentResult]:
        results = self.list_results(workspace_id)
        return next((r for r in results if r.result_id == result_id), None)

    def save_result(self, result: ExperimentResult, workspace_id: str) -> None:

        results = self.list_results(workspace_id)
        idx = next((i for i, r in enumerate(results) if r.result_id == result.result_id), None)
        if idx is not None:
            results[idx] = result
        else:
            results.append(result)
        self._save_list(workspace_id, "results.json", [r.to_dict() for r in results])

    def save_results_batch(self, results_batch: List[ExperimentResult], workspace_id: str) -> None:
        existing = self.list_results(workspace_id)
        existing_map = {r.result_id: idx for idx, r in enumerate(existing)}
        for r in results_batch:
            if r.result_id in existing_map:
                existing[existing_map[r.result_id]] = r
            else:
                existing.append(r)
        self._save_list(workspace_id, "results.json", [r.to_dict() for r in existing])

    def delete_result(self, workspace_id: str, result_id: str) -> bool:
        results = self.list_results(workspace_id)
        filtered = [r for r in results if r.result_id != result_id]
        if len(filtered) != len(results):
            self._save_list(workspace_id, "results.json", [r.to_dict() for r in filtered])
            return True
        return False

    def get_results_for_run(self, workspace_id: str, run_id: str) -> List[ExperimentResult]:
        return [r for r in self.list_results(workspace_id) if r.run_id == run_id]

    # --- Dataset & Model References ---

    def list_dataset_refs(self, workspace_id: str) -> List[DatasetReference]:
        raw = self._load_list(workspace_id, "datasets.json")
        return [DatasetReference.from_dict(item) for item in raw]

    def save_dataset_ref(self, ref: DatasetReference, workspace_id: str) -> None:
        refs = self.list_dataset_refs(workspace_id)
        idx = next((i for i, r in enumerate(refs) if r.dataset_id == ref.dataset_id), None)
        if idx is not None:
            refs[idx] = ref
        else:
            refs.append(ref)
        self._save_list(workspace_id, "datasets.json", [r.to_dict() for r in refs])

    def list_model_refs(self, workspace_id: str) -> List[ModelReference]:
        raw = self._load_list(workspace_id, "models.json")
        return [ModelReference.from_dict(item) for item in raw]

    def save_model_ref(self, ref: ModelReference, workspace_id: str) -> None:
        refs = self.list_model_refs(workspace_id)
        idx = next((i for i, r in enumerate(refs) if r.model_id == ref.model_id), None)
        if idx is not None:
            refs[idx] = ref
        else:
            refs.append(ref)
        self._save_list(workspace_id, "models.json", [r.to_dict() for r in refs])

    # --- Research Findings ---

    def list_findings(self, workspace_id: str) -> List[ResearchFinding]:
        raw = self._load_list(workspace_id, "findings.json")
        return [ResearchFinding.from_dict(item) for item in raw]

    def get_finding(self, workspace_id: str, finding_id: str) -> Optional[ResearchFinding]:
        for finding in self.list_findings(workspace_id):
            if finding.finding_id == finding_id:
                return finding
        return None

    def save_finding(self, finding: ResearchFinding) -> None:
        findings = self.list_findings(finding.workspace_id)
        idx = next((i for i, f in enumerate(findings) if f.finding_id == finding.finding_id), None)
        if idx is not None:
            findings[idx] = finding
        else:
            findings.append(finding)
        self._save_list(finding.workspace_id, "findings.json", [f.to_dict() for f in findings])

    def delete_finding(self, workspace_id: str, finding_id: str) -> bool:
        findings = self.list_findings(workspace_id)
        filtered = [f for f in findings if f.finding_id != finding_id]
        if len(filtered) != len(findings):
            self._save_list(workspace_id, "findings.json", [f.to_dict() for f in filtered])
            return True
        return False

    # --- Reproducibility Reports ---

    def list_reproducibility_reports(self, workspace_id: str) -> List[ReproducibilityReport]:
        raw = self._load_list(workspace_id, "reproducibility.json")
        return [ReproducibilityReport.from_dict(item) for item in raw]

    def save_reproducibility_report(self, report: ReproducibilityReport, workspace_id: str) -> None:
        reports = self.list_reproducibility_reports(workspace_id)
        idx = next((i for i, r in enumerate(reports) if r.report_id == report.report_id), None)
        if idx is not None:
            reports[idx] = report
        else:
            reports.append(report)
        self._save_list(workspace_id, "reproducibility.json", [r.to_dict() for r in reports])

    # --- Research Reviews ---

    def list_reviews(self, workspace_id: str) -> List[ResearchReview]:
        raw = self._load_list(workspace_id, "reviews.json")
        return [ResearchReview.from_dict(item) for item in raw]

    def save_review(self, review: ResearchReview, workspace_id: str) -> None:
        reviews = self.list_reviews(workspace_id)
        idx = next((i for i, r in enumerate(reviews) if r.review_id == review.review_id), None)
        if idx is not None:
            reviews[idx] = review
        else:
            reviews.append(review)
        self._save_list(workspace_id, "reviews.json", [r.to_dict() for r in reviews])

    # --- Activity & Audit Logging ---

    def log_activity(self, workspace_id: str, actor_id: str, action: str, target_type: str, target_id: str, summary: str) -> ActivityEvent:
        event = ActivityEvent(
            event_id=generate_id("act"),
            workspace_id=workspace_id,
            actor_id=actor_id,
            action=action,
            target_type=target_type,
            target_id=target_id,
            summary=summary,
        )
        activities = self.list_activity(workspace_id)
        activities.insert(0, event)  # newest first
        self._save_list(workspace_id, "activity.json", [a.to_dict() for a in activities[:200]])
        return event

    def list_activity(self, workspace_id: str) -> List[ActivityEvent]:
        raw = self._load_list(workspace_id, "activity.json")
        return [ActivityEvent.from_dict(item) for item in raw]

    def log_audit(self, workspace_id: str, actor_id: str, action: str, target_type: str, target_id: str, metadata: Optional[Dict[str, Any]] = None) -> AuditEvent:
        clean_metadata = {k: v for k, v in (metadata or {}).items() if "secret" not in k.lower() and "key" not in k.lower()}
        event = AuditEvent(
            event_id=generate_id("aud"),
            workspace_id=workspace_id,
            actor_id=actor_id,
            action=action,
            target_type=target_type,
            target_id=target_id,
            metadata=clean_metadata,
        )
        audits = self.list_audit(workspace_id)
        audits.insert(0, event)
        self._save_list(workspace_id, "audit.json", [a.to_dict() for a in audits[:500]])
        return event

    def list_audit(self, workspace_id: str) -> List[AuditEvent]:
        raw = self._load_list(workspace_id, "audit.json")
        return [AuditEvent.from_dict(item) for item in raw]

    # --- Phase 12: Experiment Registry ---

    def list_registry_entries(self, workspace_id: str) -> List[ExperimentRegistryEntry]:
        raw = self._load_list(workspace_id, "registry.json")
        return [ExperimentRegistryEntry.from_dict(item) for item in raw]

    def save_registry_entry(self, entry: ExperimentRegistryEntry, workspace_id: str) -> None:
        entries = self.list_registry_entries(workspace_id)
        entry.updated_at = current_iso_time()
        idx = next((i for i, e in enumerate(entries) if e.experiment_id == entry.experiment_id), None)
        if idx is not None:
            entries[idx] = entry
        else:
            entries.append(entry)
        self._save_list(workspace_id, "registry.json", [e.to_dict() for e in entries])

    def get_registry_entry(self, workspace_id: str, experiment_id: str) -> Optional[ExperimentRegistryEntry]:
        entries = self.list_registry_entries(workspace_id)
        return next((e for e in entries if e.experiment_id == experiment_id), None)

    def delete_registry_entry(self, workspace_id: str, experiment_id: str) -> bool:
        entries = self.list_registry_entries(workspace_id)
        filtered = [e for e in entries if e.experiment_id != experiment_id]
        if len(filtered) != len(entries):
            self._save_list(workspace_id, "registry.json", [e.to_dict() for e in filtered])
            return True
        return False

    # --- Phase 12: Execution Profiles ---

    def list_execution_profiles(self, workspace_id: str) -> List[ExecutionProfile]:
        raw = self._load_list(workspace_id, "profiles.json")
        return [ExecutionProfile.from_dict(item) for item in raw]

    def save_execution_profile(self, profile: ExecutionProfile, workspace_id: str) -> None:
        profiles = self.list_execution_profiles(workspace_id)
        idx = next((i for i, p in enumerate(profiles) if p.profile_id == profile.profile_id), None)
        if idx is not None:
            profiles[idx] = profile
        else:
            profiles.append(profile)
        self._save_list(workspace_id, "profiles.json", [p.to_dict() for p in profiles])

    def get_execution_profile(self, workspace_id: str, profile_id: str) -> Optional[ExecutionProfile]:
        profiles = self.list_execution_profiles(workspace_id)
        return next((p for p in profiles if p.profile_id == profile_id), None)

    def delete_execution_profile(self, workspace_id: str, profile_id: str) -> bool:
        profiles = self.list_execution_profiles(workspace_id)
        filtered = [p for p in profiles if p.profile_id != profile_id]
        if len(filtered) != len(profiles):
            self._save_list(workspace_id, "profiles.json", [p.to_dict() for p in filtered])
            return True
        return False

    # --- Phase 12: Experiment Configurations ---

    def list_configurations(self, workspace_id: str) -> List[ExperimentConfiguration]:
        raw = self._load_list(workspace_id, "configs.json")
        return [ExperimentConfiguration.from_dict(item) for item in raw]

    def save_configuration(self, config: ExperimentConfiguration, workspace_id: str) -> None:
        configs = self.list_configurations(workspace_id)
        idx = next((i for i, c in enumerate(configs) if c.configuration_id == config.configuration_id), None)
        if idx is not None:
            configs[idx] = config
        else:
            configs.append(config)
        self._save_list(workspace_id, "configs.json", [c.to_dict() for c in configs])

    def get_configuration(self, workspace_id: str, configuration_id: str) -> Optional[ExperimentConfiguration]:
        configs = self.list_configurations(workspace_id)
        return next((c for c in configs if c.configuration_id == configuration_id), None)

    # --- Phase 12: Artifact Management ---

    def list_artifacts(self, workspace_id: str, run_id: Optional[str] = None) -> List[ExperimentArtifact]:
        raw = self._load_list(workspace_id, "artifacts.json")
        all_arts = [ExperimentArtifact.from_dict(item) for item in raw]
        if run_id:
            return [a for a in all_arts if a.run_id == run_id]
        return all_arts

    def save_artifact(self, artifact: ExperimentArtifact, workspace_id: str) -> None:
        artifacts = self.list_artifacts(workspace_id)
        idx = next((i for i, a in enumerate(artifacts) if a.artifact_id == artifact.artifact_id), None)
        if idx is not None:
            artifacts[idx] = artifact
        else:
            artifacts.append(artifact)
        self._save_list(workspace_id, "artifacts.json", [a.to_dict() for a in artifacts])

    def get_artifact(self, workspace_id: str, artifact_id: str) -> Optional[ExperimentArtifact]:
        artifacts = self.list_artifacts(workspace_id)
        return next((a for a in artifacts if a.artifact_id == artifact_id), None)

    # --- Phase 12: Execution Queue ---

    def list_queue_items(self, workspace_id: str) -> List[ExecutionQueueItem]:
        raw = self._load_list(workspace_id, "queue.json")
        return [ExecutionQueueItem.from_dict(item) for item in raw]

    def save_queue_item(self, item: ExecutionQueueItem, workspace_id: str) -> None:
        items = self.list_queue_items(workspace_id)
        idx = next((i for i, q in enumerate(items) if q.queue_id == item.queue_id), None)
        if idx is not None:
            items[idx] = item
        else:
            items.append(item)
        self._save_list(workspace_id, "queue.json", [q.to_dict() for q in items])

    # --- Phase 13: Model Registry ---

    def list_registered_models(self, workspace_id: str) -> List[ResearchModelRegistration]:
        raw = self._load_list(workspace_id, "models.json")
        return [ResearchModelRegistration.from_dict(item) for item in raw]

    def get_registered_model(self, workspace_id: str, model_id: str) -> Optional[ResearchModelRegistration]:
        models = self.list_registered_models(workspace_id)
        return next((m for m in models if m.model_id == model_id), None)

    def save_registered_model(self, model: ResearchModelRegistration, workspace_id: str) -> None:
        models = self.list_registered_models(workspace_id)
        idx = next((i for i, m in enumerate(models) if m.model_id == model.model_id), None)
        if idx is not None:
            models[idx] = model
        else:
            models.append(model)
        self._save_list(workspace_id, "models.json", [m.to_dict() for m in models])

    def delete_registered_model(self, workspace_id: str, model_id: str) -> bool:
        models = self.list_registered_models(workspace_id)
        filtered = [m for m in models if m.model_id != model_id]
        if len(filtered) != len(models):
            self._save_list(workspace_id, "models.json", [m.to_dict() for m in filtered])
            return True
        return False

    # --- Phase 14: Execution Jobs & Image Allowlist ---

    def list_jobs(self, workspace_id: str) -> List[ExecutionJob]:
        raw = self._load_list(workspace_id, "jobs.json")
        return [ExecutionJob.from_dict(item) for item in raw]

    def get_job(self, workspace_id: str, job_id: str) -> Optional[ExecutionJob]:
        jobs = self.list_jobs(workspace_id)
        return next((j for j in jobs if j.job_id == job_id), None)

    def save_job(self, job: ExecutionJob, workspace_id: str) -> None:
        jobs = self.list_jobs(workspace_id)
        idx = next((i for i, j in enumerate(jobs) if j.job_id == job.job_id), None)
        if idx is not None:
            jobs[idx] = job
        else:
            jobs.append(job)
        self._save_list(workspace_id, "jobs.json", [j.to_dict() for j in jobs])

    def save_jobs_batch(self, jobs_batch: List[ExecutionJob], workspace_id: str) -> None:
        existing = self.list_jobs(workspace_id)
        existing_map = {j.job_id: idx for idx, j in enumerate(existing)}
        for j in jobs_batch:
            if j.job_id in existing_map:
                existing[existing_map[j.job_id]] = j
            else:
                existing.append(j)
                existing_map[j.job_id] = len(existing) - 1
        self._save_list(workspace_id, "jobs.json", [j.to_dict() for j in existing])

    def list_allowed_images(self, workspace_id: str) -> List[ExecutionImageRegistryEntry]:
        raw = self._load_list(workspace_id, "allowed_images.json")
        if not raw:
            # Default built-in approved images
            defaults = [
                ExecutionImageRegistryEntry(
                    image="researchlens/nlp-experiment:1.0",
                    digest="NOT_PROVIDED",
                    allowed=True,
                    owner="system",
                    description="Standard ResearchLens NLP experiment runtime",
                ),
                ExecutionImageRegistryEntry(
                    image="researchlens/cpu-baseline:1.0",
                    digest="NOT_PROVIDED",
                    allowed=True,
                    owner="system",
                    description="Standard CPU baseline runtime",
                ),
            ]
            self._save_list(workspace_id, "allowed_images.json", [d.to_dict() for d in defaults])
            return defaults
        return [ExecutionImageRegistryEntry.from_dict(item) for item in raw]

    def get_allowed_image(self, workspace_id: str, image_name: str) -> Optional[ExecutionImageRegistryEntry]:
        images = self.list_allowed_images(workspace_id)
        return next((img for img in images if img.image == image_name or img.image.split(":")[0] == image_name.split(":")[0]), None)

    def save_allowed_image(self, entry: ExecutionImageRegistryEntry, workspace_id: str) -> None:
        images = self.list_allowed_images(workspace_id)
        idx = next((i for i, img in enumerate(images) if img.image == entry.image), None)
        if idx is not None:
            images[idx] = entry
        else:
            images.append(entry)
        self._save_list(workspace_id, "allowed_images.json", [img.to_dict() for img in images])

    def delete_allowed_image(self, workspace_id: str, image_name: str) -> bool:
        images = self.list_allowed_images(workspace_id)
        filtered = [img for img in images if img.image != image_name]
        if len(filtered) != len(images):
            self._save_list(workspace_id, "allowed_images.json", [img.to_dict() for img in filtered])
            return True
        return False

    def get_run_dir(self, workspace_id: str, run_id: str) -> str:
        """Get or create the sandboxed run workspace directory structure."""
        clean_ws = _sanitize_id(workspace_id)
        clean_run = _sanitize_id(run_id)
        run_dir = os.path.join(self._workspace_dir(clean_ws), "runs", clean_run)
        os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "artifacts"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "reports"), exist_ok=True)
        return run_dir

    def export_workspace_bundle(self, workspace_id: str) -> Dict[str, Any]:
        """Export the entire state of a workspace into a unified dict."""
        ws = self.get_workspace(workspace_id)
        return {
            "workspace": ws.to_dict() if ws else {},
            "collections": [c.to_dict() for c in self.list_collections(workspace_id)],
            "papers": [p.to_dict() for p in self.list_papers(workspace_id)],
            "evidence": [e.to_dict() for e in self.list_evidence(workspace_id)],
            "annotations": [a.to_dict() for a in self.list_annotations(workspace_id)],
            "questions": [q.to_dict() for q in self.list_questions(workspace_id)],
            "hypotheses": [h.to_dict() for h in self.list_hypotheses(workspace_id)],
            "directions": [d.to_dict() for d in self.list_directions(workspace_id)],
            "plans": [p.to_dict() for p in self.list_plans(workspace_id)],
            "experiments": [e.to_dict() for e in self.list_experiments(workspace_id)],
            "runs": [r.to_dict() for r in self.list_runs(workspace_id)],
            "results": [res.to_dict() for res in self.list_results(workspace_id)],
            "findings": [f.to_dict() for f in self.list_findings(workspace_id)],
            "reviews": [rev.to_dict() for rev in self.list_reviews(workspace_id)],
            "members": [m.to_dict() for m in self.list_members(workspace_id)],
            "comments": [c.to_dict() for c in self.list_comments(workspace_id)],
            "threads": [t.to_dict() for t in self.list_threads(workspace_id)],
            "risks": [r.to_dict() for r in self.list_risks(workspace_id)],
            "milestones": [m.to_dict() for m in self.list_milestones(workspace_id)],
            "activity": [act.to_dict() for act in self.list_activity(workspace_id)],
            "audit": [aud.to_dict() for aud in self.list_audit(workspace_id)],
            "registry": [reg.to_dict() for reg in self.list_registry_entries(workspace_id)],
            "profiles": [prof.to_dict() for prof in self.list_execution_profiles(workspace_id)],
            "artifacts": [art.to_dict() for art in self.list_artifacts(workspace_id)],
            "registered_models": [m.to_dict() for m in self.list_registered_models(workspace_id)],
            "jobs": [j.to_dict() for j in self.list_jobs(workspace_id)],
            "allowed_images": [img.to_dict() for img in self.list_allowed_images(workspace_id)],
            "workflows": [w.to_dict() for w in self.list_workflows(workspace_id)],
        }

    # --- Phase 15: Research Automation Workflows ---

    def list_workflows(self, workspace_id: str) -> List[ResearchWorkflowState]:
        raw = self._load_list(workspace_id, "workflows.json")
        return [ResearchWorkflowState.from_dict(item) for item in raw]

    def save_workflow(self, workflow: ResearchWorkflowState, workspace_id: str) -> None:
        workflows = self.list_workflows(workspace_id)
        workflow.updated_at = current_iso_time()
        idx = next((i for i, w in enumerate(workflows) if w.workflow_id == workflow.workflow_id), None)
        if idx is not None:
            workflows[idx] = workflow
        else:
            workflows.append(workflow)
        self._save_list(workspace_id, "workflows.json", [w.to_dict() for w in workflows])

    def get_workflow(self, workspace_id: str, workflow_id: str) -> Optional[ResearchWorkflowState]:
        workflows = self.list_workflows(workspace_id)
        return next((w for w in workflows if w.workflow_id == workflow_id), None)

    def delete_workflow(self, workspace_id: str, workflow_id: str) -> bool:
        workflows = self.list_workflows(workspace_id)
        filtered = [w for w in workflows if w.workflow_id != workflow_id]
        if len(filtered) != len(workflows):
            self._save_list(workspace_id, "workflows.json", [w.to_dict() for w in filtered])
            return True
        return False

