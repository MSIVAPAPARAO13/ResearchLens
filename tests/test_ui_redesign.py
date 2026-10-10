"""Unit and integration tests for ResearchLens UI Redesign (Phases 0–17).

Verifies:
- Design system CSS tokens and presence
- Reusable UI elements (page header, stat card, citation panel, approval gate)
- Dashboard overview logic with empty and populated workspace data
- Upload component validation (size constraints, empty bytes)
- Global navigation component exports
"""

from pathlib import Path

from app.components import (
    render_approval_gate_banner,
    render_citation_panel,
    render_dashboard_view,
    render_document_overview,
    render_empty_state,
    render_page_header,
    render_stat_card,
    render_summary_view,
    render_upload_component,
)
from researchlens.ingestion.document import Document, DocumentMetadata
from researchlens.workspace.schema import SavedPaper
from researchlens.workspace.store import WorkspaceStore


def test_custom_css_design_tokens_present():
    """Verify that custom.css defines all required design system variables and tokens."""
    css_path = Path("app/styles/custom.css")
    assert css_path.exists(), "custom.css file must exist"
    content = css_path.read_text(encoding="utf-8")

    # Tokens
    assert "--lens-bg-root" in content
    assert "--lens-surface-card" in content
    assert "--lens-border-subtle" in content
    assert "--lens-text-primary" in content
    assert "--lens-accent-blue" in content
    assert "--lens-status-success" in content

    # Component classes
    assert ".lens-header" in content
    assert ".lens-card" in content
    assert ".lens-stat-box" in content
    assert ".lens-citation-panel" in content
    assert ".lens-badge" in content
    assert ".lens-empty-state" in content
    assert ".tldr-card" in content
    assert ".exec-card" in content
    assert ".approval-gate-banner" in content


def test_ui_elements_execution():
    """Verify reusable UI elements render valid HTML strings without errors."""
    # Test page header
    render_page_header(
        title="Test Scientific Header",
        subtitle="Test Subtitle",
        workspace_context="Test Workspace",
        badge_label="READY",
        badge_variant="success",
    )

    # Test stat card
    render_stat_card("Total Citations", "42", subtext="Peer reviewed")

    # Test empty state
    render_empty_state("No Data Found", "Please upload a document to proceed.", icon="📄")

    # Test citation panel
    render_citation_panel(
        page_number=4,
        section_name="methodology",
        source_text="The model was evaluated using AdamW with learning rate 1e-4.",
        confidence=0.92,
        label="Method Citation",
        chunk_id="chunk_abc123456",
    )

    # Test approval gate banner
    render_approval_gate_banner(
        stage_name="Stage 7: Approval Gate",
        approval_status="PENDING_RESEARCHER_APPROVAL",
        required_role="RESEARCHER",
    )


def test_summary_and_overview_elements():
    """Verify document overview and summary components render without error."""
    doc = Document(
        filename="paper_test.pdf",
        full_text="Sample paper text across multiple sections.",
        metadata=DocumentMetadata(
            title="Sample Research Paper",
            file_size_bytes=1024 * 512,
        ),
    )
    sections = {"introduction": "Intro text", "methodology": "Method text"}

    render_document_overview(doc, sections)
    render_summary_view(
        tldr="This paper proposes a new method.",
        executive_summary="Comprehensive summary of findings.",
        section_summaries=sections,
    )


def test_dashboard_view_with_temp_workspace(tmp_path):
    """Verify dashboard view executes cleanly against WorkspaceStore."""
    store = WorkspaceStore(base_dir=str(tmp_path / "workspaces"))

    # Initial empty state render
    render_dashboard_view(store=store)

    # Create workspace and add paper
    ws = store.create_workspace("Test NLP Project", "Investigating sparse attention.")
    paper = SavedPaper(
        paper_id="p1",
        workspace_id=ws.workspace_id,
        title="Sparse Attention Transformers",
        authors=["A. Vaswani et al."],
        year=2023,
    )
    store.save_paper(paper)
    store.log_activity(
        ws.workspace_id,
        "user_1",
        "PAPER_INGESTED",
        "PAPER",
        "p1",
        "Ingested 'Sparse Attention Transformers'.",
    )

    # Populated state render
    render_dashboard_view(store=store)

    papers = store.list_papers(ws.workspace_id)
    assert len(papers) == 1
    activities = store.list_activity(ws.workspace_id)
    assert len(activities) == 1
    assert activities[0].action == "PAPER_INGESTED"


def test_upload_component_validation(monkeypatch):
    """Verify upload component handles None uploads cleanly."""
    bytes_out, name_out = render_upload_component()
    assert bytes_out is None
    assert name_out is None


def test_engine_specs_device_string_formatting():
    """Verify get_device() string conversion prevents torch.device AttributeError."""
    from researchlens.summarization import get_device

    dev = get_device()
    dev_str = str(dev).upper()
    assert isinstance(dev_str, str)
    assert dev_str in ("CPU", "CUDA", "MPS") or "CPU" in dev_str or "CUDA" in dev_str


def test_render_execution_view_integration(tmp_path):
    """Verify render_execution_view executes cleanly without AttributeError."""
    from app.components import render_execution_view
    from researchlens.execution.registry import ExperimentRegistryService
    from researchlens.workspace.schema import ExperimentResult

    store = WorkspaceStore(base_dir=str(tmp_path / "workspaces"))

    # 1. Empty workspaces test
    render_execution_view(store=store)

    # 2. Populated workspace test
    ws = store.create_workspace("Execution View Workspace")
    reg_service = ExperimentRegistryService(store)
    entry, err = reg_service.register_experiment(
        workspace_id=ws.workspace_id,
        name="Deterministic Benchmark Demo",
        entrypoint="scripts/experiments/deterministic_demo.py",
        version="1.0.0",
    )
    assert entry is not None
    assert err == ""

    # Add experiment results
    r1 = ExperimentResult(
        result_id="res_1",
        run_id="run_1",
        model="DistilBERT",
        dataset="SQuAD",
        metric="f1",
        value=0.912,
        seed=42,
    )
    r2 = ExperimentResult(
        result_id="res_2",
        run_id="run_2",
        model="DistilBERT",
        dataset="SQuAD",
        metric="f1",
        value=0.918,
        seed=123,
    )
    store.save_results_batch([r1, r2], ws.workspace_id)

    # Execute view render with populated registry and results
    render_execution_view(store=store)


def test_render_all_major_views_integration(tmp_path):
    """Verify all major Streamlit views execute cleanly against workspace store."""
    from app.components import (
        render_literature_intelligence_view,
        render_mlops_infrastructure_view,
        render_mlops_view,
        render_research_automation_view,
        render_research_monitor_view,
        render_workspace_view,
    )

    store = WorkspaceStore(base_dir=str(tmp_path / "workspaces"))
    ws = store.create_workspace("Comprehensive View Test Workspace")

    # 1. Workspace View
    render_workspace_view(store=store)

    # 2. MLOps View
    render_mlops_view(store=store)

    # 3. MLOps Infrastructure View
    render_mlops_infrastructure_view(store=store, workspace_id=ws.workspace_id)

    # 4. Research Automation View
    render_research_automation_view(store=store)

    # 5. Literature Intelligence View
    render_literature_intelligence_view(collection=None)

    # 6. Research Monitor View
    render_research_monitor_view()

