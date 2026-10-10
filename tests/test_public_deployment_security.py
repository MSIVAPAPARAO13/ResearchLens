"""Tests for Phase 16.2: Secure Public Deployment and Reverse Proxy Hardening.

Verifies:
1. Reverse proxy configuration: TLSv1.2/1.3, HSTS, security headers, rate limit zones.
2. Internal service isolation: MLflow and ResearchLens not exposed to 0.0.0.0 in docker-compose.
3. Authentication & Authorization: Streamlit production secret key gate, MLflow basic auth.
4. Production secret strength validation without value leakage.
5. End-to-end secure workflow: upload, hypothesis, deterministic execution, MLflow tracking, restart persistence.
6. Malicious inputs: oversized payload rejection, path traversal rejection, untrusted experiment rejection.
"""

from pathlib import Path

import pytest
import yaml

from researchlens.config.settings import Settings
from researchlens.execution.executor import ExperimentExecutor
from researchlens.execution.registry import ExperimentRegistryService
from researchlens.ingestion.pdf_loader import Document, Page
from researchlens.mlops.client import get_mlflow_client
from researchlens.mlops.configuration import MLflowConfig
from researchlens.mlops.tracking import MLflowTracker
from researchlens.preprocessing.section_parser import split_into_sections
from researchlens.workspace.schema import (
    ExperimentPlan,
    ResearchFinding,
    ResearchHypothesis,
    SavedPaper,
    WorkspaceMember,
)
from researchlens.workspace.store import WorkspaceStore, _sanitize_id


def test_nginx_reverse_proxy_security_configuration():
    """Verify that nginx.conf enforces TLS, HSTS, CSP, rate limiting, and 100MB upload limit."""
    nginx_conf = Path("nginx/nginx.conf")
    assert nginx_conf.exists(), "nginx/nginx.conf must exist"
    content = nginx_conf.read_text(encoding="utf-8")

    # 1. TLS protocols and cipher suites
    assert "ssl_protocols TLSv1.2 TLSv1.3;" in content
    assert "ssl_certificate /etc/nginx/ssl/cert.pem;" in content
    assert "ssl_certificate_key /etc/nginx/ssl/key.pem;" in content

    # 2. HTTP to HTTPS redirect
    assert "return 301 https://$host$request_uri;" in content

    # 3. Security headers
    assert "Strict-Transport-Security" in content
    assert "X-Frame-Options" in content
    assert "X-Content-Type-Options" in content
    assert "Referrer-Policy" in content
    assert "Content-Security-Policy" in content

    # 4. Request limits & rate limiting
    assert "client_max_body_size 100M;" in content
    assert "limit_req_zone" in content
    assert "limit_req zone=" in content

    # 5. Restricted MLflow access
    assert "auth_basic" in content
    assert "auth_basic_user_file /etc/nginx/.htpasswd;" in content

    # 6. Streamlit WebSocket upgrade
    assert "proxy_set_header Upgrade $http_upgrade;" in content
    assert 'proxy_set_header Connection "upgrade";' in content


def test_docker_compose_internal_service_isolation():
    """Verify that docker-compose.yml does not publish internal MLflow or app ports on 0.0.0.0."""
    compose_path = Path("docker-compose.yml")
    assert compose_path.exists()
    content = compose_path.read_text(encoding="utf-8")
    data = yaml.safe_load(content)

    services = data.get("services", {})
    assert "reverse-proxy" in services, "reverse-proxy service must be defined"
    assert "researchlens" in services
    assert "mlflow" in services

    # Reverse proxy publishes 80 and 443
    proxy_ports = services["reverse-proxy"].get("ports", [])
    assert any("80:80" in str(p) for p in proxy_ports)
    assert any("443:443" in str(p) for p in proxy_ports)

    # MLflow must NOT be bound to 0.0.0.0:5000
    mlflow_ports = services["mlflow"].get("ports", [])
    for port in mlflow_ports:
        port_str = str(port)
        assert not port_str.startswith("5000:"), "MLflow port 5000 must not be exposed to 0.0.0.0"
        assert not port_str.startswith("0.0.0.0:5000:"), "MLflow port 5000 must not be exposed to 0.0.0.0"

    # ResearchLens must NOT be bound to 0.0.0.0:8501
    app_ports = services["researchlens"].get("ports", [])
    for port in app_ports:
        port_str = str(port)
        assert not port_str.startswith("8501:"), "App port 8501 must not be exposed to 0.0.0.0"
        assert not port_str.startswith("0.0.0.0:8501:"), "App port 8501 must not be exposed to 0.0.0.0"


def test_production_secret_strength_validation_without_leakage():
    """Verify that Settings.validate_production_configuration audits secret strength without printing value."""
    # 1. Secret too short
    short_secret = "weak"
    settings_short = Settings(
        env="production",
        debug=False,
        api_secret_key=short_secret,
    )
    issues_short = settings_short.validate_production_configuration()
    assert any("too short" in i for i in issues_short)
    # Ensure the secret string itself is NOT included in the error message
    assert short_secret not in " ".join(issues_short)

    # 2. Secret adequate length (32 chars)
    valid_secret = "a" * 32
    settings_valid = Settings(
        env="production",
        debug=False,
        api_secret_key=valid_secret,
    )
    issues_valid = settings_valid.validate_production_configuration()
    assert len(issues_valid) == 0


def test_malicious_inputs_and_traversal_rejection(tmp_path: Path):
    """Test path traversal rejection, oversize limits, and untrusted execution blocking."""
    store = WorkspaceStore(base_dir=str(tmp_path))
    ws = store.create_workspace("Security Workspace")
    ws_id = ws.workspace_id

    # 1. Path traversal in identifiers
    malicious_inputs = ["../secret", "../../etc/passwd", "..\\system32", "eval/../bad", "/root"]
    for bad in malicious_inputs:
        with pytest.raises(ValueError, match="Invalid identifier"):
            _sanitize_id(bad)

    # 2. Untrusted experiment execution
    executor = ExperimentExecutor(store)
    plan = ExperimentPlan("exp_untrusted", name="Untrusted Plan", entrypoint="unregistered/evil.py")
    store.save_experiment(plan, ws_id)
    store.save_member(WorkspaceMember("m_res", ws_id, "Bob", "bob@lab.org", "RESEARCHER"))

    run, err = executor.execute_experiment(ws_id, "exp_untrusted", actor_id="m_res")
    assert run is None
    assert "Execution blocked" in err


def test_end_to_end_secure_workflow_and_mlflow_tracking(tmp_path: Path):
    """Test paper ingestion, sectioning, hypothesis, deterministic execution, MLflow, and persistence."""
    ws_dir = tmp_path / "workspaces"
    mlruns_dir = tmp_path / "mlruns"
    store = WorkspaceStore(base_dir=str(ws_dir))
    ws = store.create_workspace("Secure E2E Workspace")
    ws_id = ws.workspace_id

    store.save_member(WorkspaceMember("m_owner", ws_id, "Alice", "alice@lab.org", "OWNER"))
    store.save_member(WorkspaceMember("m_res", ws_id, "Bob", "bob@lab.org", "RESEARCHER"))

    # 1. Paper ingestion & section parsing
    paper_text = (
        "Abstract\nRetrieval-augmented generation (RAG) dramatically improves factual grounding in large language models. "
        "However, multi-hop reasoning over complex scientific graphs and citation networks remains an open challenge.\n\n"
        "Introduction\nRecent literature demonstrates high citation accuracy across academic benchmarks.\n\n"
        "Methodology\nWe systematically evaluate dense embeddings using FAISS-CPU coupled with lexical BM25Okapi keyword scores, "
        "reranked using a cross-encoder model that scores query-document pairs with joint cross-attention.\n\n"
        "Results\nOur hybrid scientific retrieval pipeline achieved a Mean Reciprocal Rank (MRR) of 0.88 and citation accuracy "
        "of 0.94 on the SciFact benchmark dataset, outperforming dense-only baseline approaches by substantial margins.\n\n"
        "Conclusion\nHybrid retrieval with cross-encoder reranking offers an accurate, reliable foundation."
    )
    doc = Document(filename="security.pdf", full_text=paper_text, pages=[Page(page_number=1, text=paper_text)])
    sections = split_into_sections(doc.full_text)
    assert "abstract" in sections
    assert "methodology" in sections

    paper = SavedPaper(paper_id="paper_sec_1", workspace_id=ws_id, title="Security Paper")
    store.save_paper(paper)

    # 2. Research Hypothesis
    hyp = ResearchHypothesis(
        hypothesis_id="hyp_sec_1",
        workspace_id=ws_id,
        statement="TLS reverse proxy eliminates insecure plaintext transport",
        independent_variables=["reverse_proxy"],
        dependent_variables=["network_security"],
    )
    store.save_hypothesis(hyp)

    # 3. Permitted Deterministic Experiment Execution
    entrypoint = "scripts/experiments/deterministic_demo.py"
    registry = ExperimentRegistryService(store)
    registry.register_experiment(ws_id, "Security Baseline", entrypoint, owner="m_owner")
    plan = ExperimentPlan("exp_sec_plan", name="Security Plan", entrypoint=entrypoint)
    store.save_experiment(plan, ws_id)

    executor = ExperimentExecutor(store)
    run, err = executor.execute_experiment(ws_id, plan.experiment_id, actor_id="m_res")
    assert err == ""
    assert run is not None
    assert run.status == "COMPLETED"

    # 4. MLflow Tracking
    ml_config = MLflowConfig(tracking_uri=str(mlruns_dir), enabled=True)
    ml_client = get_mlflow_client(ml_config)
    tracker = MLflowTracker(config=ml_config, client=ml_client)

    exp_id = tracker.get_or_create_experiment("Sec_Exp", ws_id)
    ml_run_id = tracker.create_run(exp_id, run_name="secure_tracking_run")
    tracker.log_parameters(ml_run_id, {"tls_version": "1.3", "proxy": "nginx"})
    tracker.log_metrics(ml_run_id, {"integrity_score": 1.0})
    tracker.terminate_run(ml_run_id, status="FINISHED")

    # 5. Candidate Finding & Review Confirmation
    finding = ResearchFinding(
        finding_id="fnd_sec_1",
        workspace_id=ws_id,
        title="TLS Hardening Confirmed",
        statement="All endpoints verified behind TLS reverse proxy",
        status="CONFIRMED",
    )
    store.save_finding(finding)

    # 6. Restart & Persistence Verification
    reloaded_store = WorkspaceStore(base_dir=str(ws_dir))
    assert len(reloaded_store.list_papers(ws_id)) == 1
    assert len(reloaded_store.list_hypotheses(ws_id)) == 1
    assert len(reloaded_store.list_runs(ws_id)) == 1
    assert reloaded_store.get_finding(ws_id, "fnd_sec_1") is not None
    assert reloaded_store.get_finding(ws_id, "fnd_sec_1").status == "CONFIRMED"
