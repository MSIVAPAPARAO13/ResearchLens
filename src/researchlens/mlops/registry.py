"""Scientific Model Registry integration and human-in-the-loop promotion workflow (Phase 13)."""

from typing import Any, Optional, Tuple

from researchlens.mlops.client import get_mlflow_client
from researchlens.mlops.configuration import MLflowConfig, get_mlflow_config
from researchlens.mlops.models import (
    ALLOWED_STAGES,
    STAGE_ARCHIVED,
    STAGE_CANDIDATE,
    STAGE_PRODUCTION,
    STAGE_VALIDATED,
)
from researchlens.utils.logging import get_logger
from researchlens.workspace.schema import (
    ExperimentArtifact,
    ExperimentRun,
    ResearchModelRegistration,
    current_iso_time,
    generate_id,
)
from researchlens.workspace.store import WorkspaceStore

logger = get_logger("mlops_registry")


class ResearchModelRegistry:
    """Manages versioned scientific models, artifact hashes, and human-in-the-loop promotion."""

    def __init__(
        self,
        store: WorkspaceStore,
        config: Optional[MLflowConfig] = None,
        client: Optional[Any] = None,
    ) -> None:
        self.store = store
        self.config = config or get_mlflow_config()
        self.client = client or get_mlflow_client(self.config)

    def register_candidate(
        self,
        workspace_id: str,
        run_id: str,
        model_name: str,
        artifact: ExperimentArtifact,
        dataset: str = "",
        metric: str = "",
        metric_value: float = 0.0,
        version: Optional[str] = None,
        notes: str = "",
        mlflow_artifact_uri: str = "",
    ) -> Tuple[Optional[ResearchModelRegistration], Optional[str]]:
        """Register a candidate model strictly after validating run success and artifact integrity."""
        # 1. Validate Run exists and is COMPLETED
        run: Optional[ExperimentRun] = self.store.get_run(workspace_id, run_id)
        if not run:
            return None, f"Source run '{run_id}' not found in workspace."
        if run.status != "COMPLETED":
            return None, f"Model registration requires COMPLETED run status, found '{run.status}'."

        # 2. Validate artifact hash
        if not artifact.sha256:
            return None, "Model registration requires calculated artifact SHA-256 integrity hash."

        # 3. Determine next non-destructive version
        existing_models = self.store.list_registered_models(workspace_id)
        same_name = [m for m in existing_models if m.name == model_name]

        if version:
            if any(m.version == version for m in same_name):
                return None, f"Model version '{version}' for '{model_name}' already exists. Overwrite prohibited."
            target_version = version
        else:
            target_version = f"1.0.{len(same_name)}"

        model_reg = ResearchModelRegistration(
            model_id=generate_id("model"),
            run_id=run_id,
            name=model_name,
            version=target_version,
            stage=STAGE_CANDIDATE,
            artifact_uri=mlflow_artifact_uri or artifact.path,
            dataset=dataset,
            metric=metric,
            metric_value=metric_value,
            artifact_hash=artifact.sha256,
            code_commit=run.git_commit_sha,
            notes=notes,
            execution_image=getattr(run, "execution_image", None),
            image_digest=getattr(run, "image_digest", None),
        )

        self.store.save_registered_model(model_reg, workspace_id)
        self.store.log_audit(
            workspace_id=workspace_id,
            actor_id="researcher",
            action="MODEL_REGISTERED",
            target_type="MODEL",
            target_id=model_reg.model_id,
            metadata={"name": model_name, "version": target_version, "stage": STAGE_CANDIDATE},
        )
        self.store.log_activity(
            workspace_id=workspace_id,
            actor_id="researcher",
            action="MODEL_REGISTERED",
            target_type="MODEL",
            target_id=model_reg.model_id,
            summary=f"Registered model candidate '{model_name}' v{target_version}.",
        )

        return model_reg, None

    def promote_model(
        self,
        workspace_id: str,
        model_id: str,
        target_stage: str,
        reviewer_id: str,
        review_notes: str = "",
    ) -> Tuple[Optional[ResearchModelRegistration], Optional[str]]:
        """Promote a registered model through stages strictly requiring human reviewer approval."""
        if target_stage not in ALLOWED_STAGES:
            return None, f"Invalid stage '{target_stage}'. Must be one of {ALLOWED_STAGES}."

        if not reviewer_id or not reviewer_id.strip():
            return None, "Model promotion strictly requires a non-empty human reviewer ID."

        # Permission verification
        if not self.store.has_permission(workspace_id, reviewer_id, "REVIEWER"):
            return None, f"Actor '{reviewer_id}' lacks permission to promote registered models."

        model = self.store.get_registered_model(workspace_id, model_id)
        if not model:
            return None, f"Registered model '{model_id}' not found."

        old_stage = model.stage
        if old_stage == target_stage:
            return model, None

        # Workflow progression rule: CANDIDATE -> VALIDATED -> PRODUCTION
        if target_stage == STAGE_PRODUCTION and old_stage != STAGE_VALIDATED:
            return None, f"Direct promotion from '{old_stage}' to PRODUCTION is prohibited. Must be VALIDATED first."

        model.stage = target_stage
        model.reviewed_by = reviewer_id
        if review_notes:
            model.notes = f"{model.notes}\n[Review by {reviewer_id}]: {review_notes}".strip()
        model.updated_at = current_iso_time()

        self.store.save_registered_model(model, workspace_id)

        audit_action = "MODEL_ARCHIVED" if target_stage == STAGE_ARCHIVED else "MODEL_PROMOTED"
        self.store.log_audit(
            workspace_id=workspace_id,
            actor_id=reviewer_id,
            action=audit_action,
            target_type="MODEL",
            target_id=model.model_id,
            metadata={"from_stage": old_stage, "to_stage": target_stage, "notes": review_notes},
        )
        self.store.log_activity(
            workspace_id=workspace_id,
            actor_id=reviewer_id,
            action=audit_action,
            target_type="MODEL",
            target_id=model.model_id,
            summary=f"Model '{model.name}' v{model.version} promoted from {old_stage} to {target_stage}.",
        )

        return model, None

    def archive_model(
        self,
        workspace_id: str,
        model_id: str,
        actor_id: str,
    ) -> Tuple[Optional[ResearchModelRegistration], Optional[str]]:
        """Move a registered model to ARCHIVED stage."""
        return self.promote_model(
            workspace_id=workspace_id,
            model_id=model_id,
            target_stage=STAGE_ARCHIVED,
            reviewer_id=actor_id,
            review_notes="Model archived.",
        )
