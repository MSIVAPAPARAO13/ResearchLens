"""Data models, tag mappings, and parameter sanitizers for MLflow integration (Phase 13)."""

import re
from typing import Any, Dict, Optional

from researchlens.workspace.schema import (
    ExperimentPlan,
    ExperimentRun,
    ResearchModelRegistration,
)

# Canonical Model Registry Stages
STAGE_NONE = "NONE"
STAGE_CANDIDATE = "CANDIDATE"
STAGE_VALIDATED = "VALIDATED"
STAGE_PRODUCTION = "PRODUCTION"
STAGE_ARCHIVED = "ARCHIVED"

ALLOWED_STAGES = (STAGE_NONE, STAGE_CANDIDATE, STAGE_VALIDATED, STAGE_PRODUCTION, STAGE_ARCHIVED)

__all__ = [
    "STAGE_NONE",
    "STAGE_CANDIDATE",
    "STAGE_VALIDATED",
    "STAGE_PRODUCTION",
    "STAGE_ARCHIVED",
    "ALLOWED_STAGES",
    "ResearchModelRegistration",
    "sanitize_parameters",
    "build_mlflow_tags",
]

SENSITIVE_PARAM_PATTERN = re.compile(
    r"(password|secret|token|api[_-]?key|credential|private[_-]?key|auth)",
    re.IGNORECASE,
)


def sanitize_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Scrub sensitive credential keys and sanitize types for MLflow parameter logging."""
    sanitized: Dict[str, Any] = {}
    for k, v in parameters.items():
        key_str = str(k).strip()
        if SENSITIVE_PARAM_PATTERN.search(key_str):
            sanitized[key_str] = "[MASKED]"
        elif isinstance(v, (int, float, bool, str)):
            sanitized[key_str] = v
        elif v is None:
            sanitized[key_str] = "None"
        elif isinstance(v, (list, dict, tuple)):
            # Flatten or stringify complex configurations up to 250 chars
            sanitized[key_str] = str(v)[:250]
        else:
            sanitized[key_str] = str(v)[:100]
    return sanitized


def build_mlflow_tags(
    workspace_id: str,
    run: ExperimentRun,
    experiment: Optional[ExperimentPlan] = None,
    hypothesis_id: Optional[str] = None,
    question_id: Optional[str] = None,
) -> Dict[str, str]:
    """Construct structured, provenance-grounded MLflow tags without fabricating missing values."""
    tags: Dict[str, str] = {
        "researchlens.workspace_id": workspace_id,
        "researchlens.run_id": run.run_id,
    }

    if experiment:
        if experiment.experiment_id:
            tags["researchlens.experiment_id"] = experiment.experiment_id
        if experiment.name:
            tags["researchlens.experiment_name"] = experiment.name
        if experiment.dataset:
            tags["researchlens.dataset_name"] = experiment.dataset
        if experiment.dataset_id:
            tags["researchlens.dataset_id"] = experiment.dataset_id
        if experiment.model:
            tags["researchlens.model_name"] = experiment.model
        if experiment.model_id:
            tags["researchlens.model_id"] = experiment.model_id

    if hypothesis_id:
        tags["researchlens.hypothesis_id"] = hypothesis_id
    if question_id:
        tags["researchlens.question_id"] = question_id

    if run.git_commit_sha and run.git_commit_sha != "NOT_AVAILABLE":
        tags["researchlens.code_commit"] = run.git_commit_sha
        tags["researchlens.commit"] = run.git_commit_sha

    # Phase 14 Lineage
    backend_val = getattr(run, "execution_backend", "LOCAL") or "LOCAL"
    tags["researchlens.execution_backend"] = backend_val

    img_val = getattr(run, "execution_image", None)
    if img_val:
        tags["researchlens.execution_image"] = img_val
    digest_val = getattr(run, "image_digest", None)
    if digest_val:
        tags["researchlens.image_digest"] = digest_val

    if experiment:
        if experiment.variables and "model_version" in experiment.variables:
            tags["researchlens.model_version"] = str(experiment.variables["model_version"])
        tags["researchlens.configuration_version"] = str(getattr(experiment, "version", 1))

    # Inspect git provenance if in git repo
    try:
        from researchlens.execution.environment import get_git_provenance
        git_prov = get_git_provenance(".")
        if git_prov.get("repository_url") != "NOT_AVAILABLE":
            tags["researchlens.repository"] = git_prov["repository_url"]
        if git_prov.get("branch") != "NOT_AVAILABLE":
            tags["researchlens.branch"] = git_prov["branch"]
    except Exception:
        pass

    if run.entrypoint:
        tags["researchlens.entrypoint"] = run.entrypoint

    if run.seed is not None:
        tags["researchlens.seed"] = str(run.seed)

    return tags
