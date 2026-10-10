"""Unit tests for MLflow integration security guards (Phase 13)."""

from unittest.mock import MagicMock

from researchlens.mlops.artifacts import MLflowArtifactManager
from researchlens.mlops.models import sanitize_parameters
from researchlens.mlops.registry import ResearchModelRegistry
from researchlens.workspace.schema import ExperimentArtifact, ExperimentRun, WorkspaceMember
from researchlens.workspace.store import WorkspaceStore


def test_secret_parameter_masking_comprehensive():
    params = {
        "huggingface_api_key": "hf_abc123456",
        "wandb_api_key": "wandb_xyz789",
        "secret_token": "super_secret",
        "db_password": "admin_password",
        "private_key": "MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQD...",
        "auth_header": "Bearer token",
        "normal_lr": 0.0005,
    }

    sanitized = sanitize_parameters(params)
    for k in ["huggingface_api_key", "wandb_api_key", "secret_token", "db_password", "private_key", "auth_header"]:
        assert sanitized[k] == "[MASKED]"
    assert sanitized["normal_lr"] == 0.0005


def test_artifact_path_traversal_blocked():
    mock_client = MagicMock()
    mgr = MLflowArtifactManager(client=mock_client)

    malicious_path = "../../../etc/passwd"
    uri = mgr.log_artifact("run_123", malicious_path)
    assert uri is None
    mock_client.log_artifact.assert_not_called()


def test_unauthorized_model_promotion_is_rejected(tmp_path):
    store = WorkspaceStore(base_dir=str(tmp_path / "workspaces"))
    ws = store.create_workspace("Sec Workspace")
    ws_id = ws.workspace_id

    # Add a VIEWER role member
    viewer = WorkspaceMember(
        member_id="viewer_bob",
        workspace_id=ws_id,
        display_name="Bob Viewer",
        email="bob@lab.org",
        role="VIEWER",
    )
    store.save_member(viewer)

    run = ExperimentRun(run_id="run_sec_01", experiment_id="exp_01", status="COMPLETED")
    store.save_run(run, ws_id)

    art = ExperimentArtifact(
        artifact_id="art_01",
        run_id=run.run_id,
        name="weights.bin",
        path="weights.bin",
        sha256="1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
    )

    reg = ResearchModelRegistry(store=store)
    model, _ = reg.register_candidate(ws_id, run.run_id, "Model-A", art)

    # Bob Viewer attempts to promote model -> denied
    _, err_unauth = reg.promote_model(ws_id, model.model_id, "VALIDATED", reviewer_id="viewer_bob")
    assert "lacks permission" in err_unauth
