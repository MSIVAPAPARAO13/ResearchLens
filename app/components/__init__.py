"""Streamlit UI modular components."""

from app.components.automation_view import render_research_automation_view
from app.components.comparison_view import render_comparison_view
from app.components.dashboard_view import render_dashboard_view
from app.components.execution_view import render_execution_view
from app.components.gap_view import render_gap_intelligence_view
from app.components.literature_view import render_literature_intelligence_view
from app.components.metadata_view import render_metadata_view, render_paper_intelligence_view
from app.components.mlops_infra_view import render_mlops_infrastructure_view
from app.components.mlops_view import render_mlops_view
from app.components.monitoring_view import render_research_monitor_view
from app.components.qa_view import render_qa_view
from app.components.summary_view import render_document_overview, render_summary_view
from app.components.ui_elements import (
    inject_seo_metadata,
    render_approval_gate_banner,
    render_badge,
    render_citation_panel,
    render_empty_state,
    render_page_header,
    render_public_landing_view,
    render_section_heading,
    render_stat_card,
)
from app.components.upload import render_upload_component
from app.components.workspace_view import render_workspace_view

__all__ = [
    "inject_seo_metadata",
    "render_public_landing_view",
    "render_upload_component",
    "render_document_overview",
    "render_summary_view",
    "render_metadata_view",
    "render_paper_intelligence_view",
    "render_gap_intelligence_view",
    "render_comparison_view",
    "render_qa_view",
    "render_literature_intelligence_view",
    "render_research_monitor_view",
    "render_workspace_view",
    "render_execution_view",
    "render_mlops_view",
    "render_mlops_infrastructure_view",
    "render_research_automation_view",
    "render_dashboard_view",
    "render_page_header",
    "render_stat_card",
    "render_empty_state",
    "render_citation_panel",
    "render_approval_gate_banner",
    "render_section_heading",
    "render_badge",
]
