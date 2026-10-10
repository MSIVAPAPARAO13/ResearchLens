"""Unit tests for Scientific Model Registry and human promotion workflow (Phase 13)."""

import os
import shutil
import tempfile

from researchlens.mlops.models import (
    STAGE_ARCHIVED,
    STAGE_CANDIDATE,
    STAGE_PRODUCTION,
    STAGE_VALIDATED,
)
from researchlens.mlops.registry import ResearchModelRegistry
from researchlens.workspace.schema import ExperimentArtifact, ExperimentRun
from researchlens.workspace.store import WorkspaceStore


def test_model_registry_candidate_lifecycle():
    temp_dir = tempfile.mkdtemp()
    try:
        store = WorkspaceStore(base_dir=os.path.join(temp_dir, "workspaces"))
        ws = store.create_workspace("Registry Test Workspace")
        ws_id = ws.workspace_id

        run = ExperimentRun(run_id="run_reg_01", experiment_id="exp_01", status="COMPLETED", git_commit_sha="c123456")
        store.save_run(run, ws_id)

        art = ExperimentArtifact(
            artifact_id="art_reg_01",
            run_id=run.run_id,
            name="model.pt",
            path="model.pt",
            sha256="abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        )

        reg = ResearchModelRegistry(store=store)

        # 1. Register candidate
        model, err = reg.register_candidate(
            workspace_id=ws_id,
            run_id=run.run_id,
            model_name="BioBERT-Classifier",
            artifact=art,
            dataset="PubMed_QA",
            metric="f1",
            metric_value=0.895,
            notes="Initial candidate checkpoint",
        )

        assert err is None
        assert model is not None
        assert model.stage == STAGE_CANDIDATE
        assert model.version == "1.0.0"
        assert model.artifact_hash == art.sha256

        # 2. Reject promotion without reviewer
        _, err_no_rev = reg.promote_model(ws_id, model.model_id, STAGE_VALIDATED, reviewer_id="")
        assert "strictly requires a non-empty human reviewer ID" in err_no_rev

        # 3. Reject direct promotion to PRODUCTION (must be VALIDATED first)
        _, err_direct_prod = reg.promote_model(ws_id, model.model_id, STAGE_PRODUCTION, reviewer_id="Dr. Alice")
        assert "Direct promotion from 'CANDIDATE' to PRODUCTION is prohibited" in err_direct_prod

        # 4. Promote to VALIDATED
        val_model, err_val = reg.promote_model(ws_id, model.model_id, STAGE_VALIDATED, reviewer_id="Dr. Alice", review_notes="Tested on holdout set")
        assert err_val is None
        assert val_model.stage == STAGE_VALIDATED
        assert val_model.reviewed_by == "Dr. Alice"

        # 5. Promote to PRODUCTION
        prod_model, err_prod = reg.promote_model(ws_id, model.model_id, STAGE_PRODUCTION, reviewer_id="Lead Scientist", review_notes="Approved for deployment")
        assert err_prod is None
        assert prod_model.stage == STAGE_PRODUCTION

        # 6. Archive model
        arch_model, err_arch = reg.archive_model(ws_id, model.model_id, actor_id="Lead Scientist")
        assert err_arch is None
        assert arch_model.stage == STAGE_ARCHIVED

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_model_registration_rejects_failed_run_or_missing_hash():
    temp_dir = tempfile.mkdtemp()
    try:
        store = WorkspaceStore(base_dir=os.path.join(temp_dir, "workspaces"))
        ws = store.create_workspace("Registry Guard Workspace")
        ws_id = ws.workspace_id

        run_failed = ExperimentRun(run_id="run_fail_01", experiment_id="exp_01", status="FAILED")
        store.save_run(run_failed, ws_id)

        art_no_hash = ExperimentArtifact(
            artifact_id="art_bad",
            run_id=run_failed.run_id,
            name="model.pt",
            path="model.pt",
            sha256="",
        )

        reg = ResearchModelRegistry(store=store)

        # Fails because run is not COMPLETED
        _, err_status = reg.register_candidate(
            workspace_id=ws_id,
            run_id=run_failed.run_id,
            model_name="BadModel",
            artifact=art_no_hash,
        )
        assert "requires COMPLETED run status" in err_status

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
