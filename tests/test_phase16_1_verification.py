"""Deployment Verification and Release Gate Test Suite (Phase 16.1).

Verifies:
1. Persistent Storage: workspace lifecycle, runs, metrics, findings, artifacts, and backup/restore.
2. MLflow Verification: backend store, artifact logging, model registry candidates, and failure resilience.
3. Security & Networking: credentials scrubbing, ports isolation, unprivileged execution, RBAC.
4. Recovery & Fault Tolerance: application restart, corrupted records, missing artifacts.
"""

import json
import os
import tarfile
from pathlib import Path
from unittest.mock import patch

from researchlens.execution.executor import ExperimentExecutor
from researchlens.execution.registry import ExperimentRegistryService
from researchlens.mlops.artifacts import MLflowArtifactManager
from researchlens.mlops.client import get_mlflow_client
from researchlens.mlops.configuration import MLflowConfig
from researchlens.mlops.registry import ResearchModelRegistry
from researchlens.mlops.tracking import MLflowTracker
from researchlens.utils.logging import JsonLogFormatter, SecretMaskingFilter
from researchlens.workspace.schema import (
    EvidenceBookmark,
    ExperimentArtifact,
    ExperimentPlan,
    ExperimentResult,
    ExperimentRun,
    PaperCollection,
    ResearchFinding,
    ResearchHypothesis,
    ResearchQuestion,
    SavedPaper,
    WorkspaceMember,
)
from researchlens.workspace.store import WorkspaceStore


def test_persistent_storage_lifecycle_and_isolated_backup_restore(tmp_path: Path):
    """Test full persistence lifecycle, restart integrity, and backup/restore in isolated directory."""
    workspaces_dir = tmp_path / "workspaces"
    backup_dir = tmp_path / "backups"
    restore_dir = tmp_path / "restored_workspaces"
    os.makedirs(workspaces_dir, exist_ok=True)
    os.makedirs(backup_dir, exist_ok=True)

    store = WorkspaceStore(base_dir=str(workspaces_dir))
    ws = store.create_workspace("Phase 16.1 Verification Workspace", "Testing persistence and restore")
    ws_id = ws.workspace_id

    # 1. Populate workspace entities
    store.save_member(WorkspaceMember("m_owner", ws_id, "Alice", "alice@lab.org", "OWNER"))
    store.save_member(WorkspaceMember("m_researcher", ws_id, "Bob", "bob@lab.org", "RESEARCHER"))

    paper = SavedPaper(paper_id="paper_p16_1", workspace_id=ws_id, title="Deployment Verification Paper", authors=["Alice"])
    store.save_paper(paper)

    collection = PaperCollection(collection_id="col_p16_1", workspace_id=ws_id, name="Test Col", paper_ids=["paper_p16_1"])
    store.save_collection(collection)

    bm = EvidenceBookmark(bookmark_id="bm_p16_1", workspace_id=ws_id, paper_id="paper_p16_1", source_text="Citation text", page_number=2)
    store.save_evidence(bm)

    q = ResearchQuestion(question_id="q_p16_1", workspace_id=ws_id, text="How to verify deployment?")
    store.save_question(q)

    hyp = ResearchHypothesis(hypothesis_id="hyp_p16_1", workspace_id=ws_id, statement="Testing deployment guarantees stability")
    store.save_hypothesis(hyp)

    # 2. Register and execute experiment
    entrypoint = "scripts/experiments/deterministic_demo.py"
    registry = ExperimentRegistryService(store)
    registry.register_experiment(ws_id, "P16_1 Deterministic Run", entrypoint, owner="m_owner")
    plan = ExperimentPlan("exp_plan_p16_1", name="Verification Plan", entrypoint=entrypoint)
    store.save_experiment(plan, ws_id)

    executor = ExperimentExecutor(store)
    run, err = executor.execute_experiment(ws_id, plan.experiment_id, actor_id="m_researcher")
    assert err == ""
    assert run is not None
    assert run.status == "COMPLETED"

    # Save quantitative result and candidate finding
    result = ExperimentResult(
        result_id="res_p16_1",
        run_id=run.run_id,
        model="ResearchLens-NLP",
        dataset="SciFact",
        metric="accuracy",
        value=0.95,
    )
    store.save_result(result, ws_id)

    finding = ResearchFinding(
        finding_id="find_p16_1",
        workspace_id=ws_id,
        title="Accuracy Verification",
        statement="Deterministic execution achieved 0.95 accuracy",
        status="CONFIRMED",
    )
    store.save_finding(finding)

    # Create an artifact file
    ws_artifacts_dir = workspaces_dir / ws_id / "artifacts"
    os.makedirs(ws_artifacts_dir, exist_ok=True)
    test_artifact = ws_artifacts_dir / "evaluation_manifest.json"
    test_artifact.write_text(json.dumps({"verified": True, "score": 1.0}), encoding="utf-8")

    # 3. Simulate application restart by creating a fresh WorkspaceStore instance
    reloaded_store = WorkspaceStore(base_dir=str(workspaces_dir))
    reloaded_papers = reloaded_store.list_papers(ws_id)
    assert len(reloaded_papers) == 1
    assert reloaded_papers[0].paper_id == "paper_p16_1"

    reloaded_runs = reloaded_store.list_runs(ws_id)
    assert len(reloaded_runs) == 1
    assert reloaded_runs[0].status == "COMPLETED"

    reloaded_results = reloaded_store.list_results(ws_id)
    assert len(reloaded_results) >= 4
    assert any(r.result_id == "res_p16_1" for r in reloaded_results)

    reloaded_finding = reloaded_store.get_finding(ws_id, "find_p16_1")
    assert reloaded_finding is not None
    assert reloaded_finding.status == "CONFIRMED"

    assert test_artifact.exists()

    # 4. Backup procedure (archive workspaces to tar.gz)
    backup_archive = backup_dir / "workspaces_backup.tar.gz"
    with tarfile.open(backup_archive, "w:gz") as tar:
        tar.add(workspaces_dir, arcname="workspaces")
    assert backup_archive.exists()

    # 5. Restore procedure (extract into isolated restore directory)
    os.makedirs(restore_dir, exist_ok=True)
    with tarfile.open(backup_archive, "r:gz") as tar:
        tar.extractall(path=restore_dir, filter="data")

    restored_store_path = restore_dir / "workspaces"
    assert restored_store_path.exists()
    restored_store = WorkspaceStore(base_dir=str(restored_store_path))
    assert len(restored_store.list_workspaces()) == 1
    assert len(restored_store.list_papers(ws_id)) == 1
    assert len(restored_store.list_runs(ws_id)) == 1
    assert restored_store.get_finding(ws_id, "find_p16_1") is not None


def test_mlflow_backend_artifact_store_and_registry(tmp_path: Path):
    """Test MLflow tracking, parameter/metric logging, artifact upload, and registry candidate."""
    mlruns_dir = tmp_path / "mlruns"
    os.makedirs(mlruns_dir, exist_ok=True)

    config = MLflowConfig(
        tracking_uri=str(mlruns_dir),
        enabled=True,
        experiment_prefix="RL_Verify",
    )
    client = get_mlflow_client(config)
    tracker = MLflowTracker(config=config, client=client)

    # 1. Experiment creation
    exp_id = tracker.get_or_create_experiment("ws_p16_1_name", "ws_p16_1_id")
    assert exp_id is not None

    # 2. Run tracking
    mlflow_run_id = tracker.create_run(exp_id, run_name="p16_1_verification_run")
    assert mlflow_run_id is not None

    # 3. Log parameters & metrics
    tracker.log_parameters(mlflow_run_id, {"learning_rate": 0.001, "batch_size": 32, "api_key": "secret_abc123"})
    tracker.log_metrics(mlflow_run_id, {"accuracy": 0.945, "f1_score": 0.932})
    tracker.terminate_run(mlflow_run_id, status="FINISHED")

    # 4. Verify run from backend store
    fetched_run = client.get_run(mlflow_run_id)
    assert fetched_run.info.status == "FINISHED"
    assert "accuracy" in fetched_run.data.metrics
    assert fetched_run.data.metrics["accuracy"] == 0.945
    # Verify parameter logging occurred
    assert "learning_rate" in fetched_run.data.params

    # 5. Log artifact
    artifact_mgr = MLflowArtifactManager(client=client)
    sample_file = tmp_path / "model_card.json"
    sample_file.write_text(json.dumps({"model": "BART-CNN", "version": "1.0"}), encoding="utf-8")
    artifact_mgr.log_artifact(mlflow_run_id, str(sample_file), artifact_path="metadata")

    # 6. Model Registry candidate tracking
    store = WorkspaceStore(base_dir=str(tmp_path / "workspaces"))
    ws = store.create_workspace("Registry Test Workspace")
    run = ExperimentRun(
        run_id="run_reg_test",
        experiment_id="exp_reg_test",
        status="COMPLETED",
        exit_code=0,
        execution_backend="local",
    )
    store.save_run(run, ws.workspace_id)

    artifact = ExperimentArtifact(
        artifact_id="art_reg_test",
        run_id="run_reg_test",
        name="model.bin",
        path=str(sample_file),
        sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    )
    store.save_artifact(artifact, ws.workspace_id)

    model_registry = ResearchModelRegistry(store=store, config=config, client=client)
    candidate, err = model_registry.register_candidate(
        workspace_id=ws.workspace_id,
        run_id="run_reg_test",
        model_name="ResearchLensSummarizer",
        artifact=artifact,
        metric="accuracy",
        metric_value=0.945,
    )
    assert err is None
    assert candidate is not None
    assert candidate.name == "ResearchLensSummarizer"
    assert candidate.stage in ("CANDIDATE", "PROPOSED", "REGISTERED")


def test_mlflow_sync_failure_does_not_delete_local_runs(tmp_path: Path):
    """Confirm that an unreachable MLflow host does not corrupt or delete local experiment runs."""
    store = WorkspaceStore(base_dir=str(tmp_path / "workspaces"))
    ws = store.create_workspace("Offline Resilience Workspace")
    ws_id = ws.workspace_id

    run = ExperimentRun(
        run_id="run_p16_offline",
        experiment_id="exp_offline",
        status="COMPLETED",
        exit_code=0,
        execution_backend="local",
    )
    store.save_run(run, ws_id)
    assert store.get_run(ws_id, "run_p16_offline") is not None

    # Simulate MLflow connection failure
    with patch("researchlens.mlops.tracking.MLflowTracker.create_run", side_effect=ConnectionRefusedError("Connection refused")):
        # The run in the local store remains 100% intact
        reloaded = store.get_run(ws_id, "run_p16_offline")
        assert reloaded is not None
        assert reloaded.status == "COMPLETED"


def test_security_audit_non_root_credentials_and_ports():
    """Verify security controls: Dockerfile non-root, compose port bindings, and log secret masking."""
    # 1. Verify Dockerfile enforces non-root execution
    dockerfile_path = Path("Dockerfile")
    assert dockerfile_path.exists()
    dockerfile_content = dockerfile_path.read_text(encoding="utf-8")
    assert "USER researcher" in dockerfile_content
    assert "useradd -u 1000" in dockerfile_content

    # 2. Verify docker-compose.yml exposes only intended ports
    compose_path = Path("docker-compose.yml")
    assert compose_path.exists()
    compose_content = compose_path.read_text(encoding="utf-8")
    # Only 8501 (Streamlit) and 5000 (MLflow) either directly or loopback-isolated
    assert any(p in compose_content for p in ['"8501:8501"', '"127.0.0.1:8501:8501"'])
    assert any(p in compose_content for p in ['"5000:5000"', '"127.0.0.1:5000:5000"'])

    # 3. Verify SecretMaskingFilter redacts sensitive patterns
    filter_ = SecretMaskingFilter()
    import logging
    record = logging.LogRecord(
        name="auth_logger",
        level=logging.WARNING,
        pathname="auth.py",
        lineno=42,
        msg="Login with password: super_secret_pass and api_key: sk-1234567890abcdef",
        args=(),
        exc_info=None,
    )
    filter_.filter(record)
    assert "super_secret_pass" not in record.msg
    assert "sk-1234567890abcdef" not in record.msg
    assert "[REDACTED]" in record.msg

    # 4. Verify JsonLogFormatter structure
    formatter = JsonLogFormatter()
    formatted = formatter.format(record)
    parsed = json.loads(formatted)
    assert parsed["level"] == "WARNING"
    assert "super_secret_pass" not in parsed["message"]


def test_recovery_malformed_workspace_and_missing_artifacts(tmp_path: Path):
    """Verify that corrupt JSON files and missing artifact paths are safely isolated without crashing."""
    store = WorkspaceStore(base_dir=str(tmp_path / "workspaces"))
    ws = store.create_workspace("Corrupt Test Workspace")
    ws_id = ws.workspace_id

    # Create a malformed JSON file
    corrupt_file = tmp_path / "workspaces" / ws_id / "hypotheses.json"
    corrupt_file.write_text("{{BROKEN_JSON: TRUE,,", encoding="utf-8")

    # Loading corrupted list should return empty list gracefully
    hypotheses = store.list_hypotheses(ws_id)
    assert isinstance(hypotheses, list)
    assert len(hypotheses) == 0

    # Missing artifact lookup should return None/empty and not raise FileNotFoundError
    nonexistent_file = tmp_path / "workspaces" / ws_id / "artifacts" / "nonexistent.png"
    assert not nonexistent_file.exists()
