"""ResearchLens Dashboard View (Phase 4).

Renders a high-level research intelligence overview using real workspace models:
- Active research project statistics (papers, hypotheses, plans, runs, findings)
- Subsystem health diagnostics (storage, compute, docker, mlflow)
- Recent research papers in active workspace
- Activity stream from real workspace events
- Quick-launch scientific action shortcuts
"""

from typing import Optional

import streamlit as st

from app.components.ui_elements import (
    render_empty_state,
    render_page_header,
    render_stat_card,
)
from researchlens.health import check_system_health
from researchlens.workspace.store import WorkspaceStore


def render_dashboard_view(store: Optional[WorkspaceStore] = None) -> None:
    """Render the Research Overview & Dashboard view."""
    if store is None:
        store = WorkspaceStore()

    workspaces = store.list_workspaces()
    current_ws = None
    ws_id = None

    if workspaces:
        ws_map = {f"{w.name} ({w.workspace_id[:8]})": w for w in workspaces}
        selected_label = st.selectbox(
            "Active Research Workspace",
            list(ws_map.keys()),
            key="dash_ws_select",
            help="Select the workspace to display metrics and recent activity for.",
        )
        current_ws = ws_map[selected_label]
        ws_id = current_ws.workspace_id

    render_page_header(
        title="Research Overview",
        subtitle="Executive scientific summary, real-time workspace metrics, and system diagnostics.",
        workspace_context=current_ws.name if current_ws else "No Workspace Active",
        badge_label="PLATFORM READY",
        badge_variant="success",
    )

    if not current_ws:
        render_empty_state(
            title="No Active Research Workspace",
            description="Create a research workspace to begin organizing papers, formulating hypotheses, executing experiment runs, and generating peer-reviewed findings.",
            icon="🧪",
        )
        with st.expander("➕ Create New Research Workspace", expanded=True):
            with st.form("dash_create_ws_form"):
                new_name = st.text_input("Workspace Name", placeholder="e.g. Multimodal Language Models")
                new_desc = st.text_area("Description", placeholder="Research scope, objectives, and baseline datasets...")
                if st.form_submit_button("Create Workspace", type="primary") and new_name.strip():
                    created = store.create_workspace(new_name.strip(), new_desc.strip())
                    store.log_activity(
                        created.workspace_id,
                        "user",
                        "WORKSPACE_CREATED",
                        "WORKSPACE",
                        created.workspace_id,
                        f"Workspace '{created.name}' initialized.",
                    )
                    st.success(f"Workspace '{created.name}' initialized successfully!")
                    st.rerun()
        return

    # Real Workspace Metrics
    papers = store.list_papers(ws_id)
    collections = store.list_collections(ws_id)
    hypotheses = store.list_hypotheses(ws_id)
    plans = store.list_plans(ws_id)
    runs = store.list_runs(ws_id)
    findings = store.list_findings(ws_id)
    unreviewed_findings = [f for f in findings if getattr(f, "status", "") == "PENDING_RESEARCHER_REVIEW"]

    # 4-Column Metric Grid
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_stat_card("Ingested Papers", str(len(papers)), subtext=f"{len(collections)} Collections")
    with col2:
        render_stat_card("Active Hypotheses", str(len(hypotheses)), subtext="Literature Grounded")
    with col3:
        render_stat_card("Experiment Plans", str(len(plans)), subtext=f"{len(runs)} Total Runs")
    with col4:
        render_stat_card("Findings In Review", str(len(unreviewed_findings)), subtext=f"{len(findings)} Total Formalized")

    st.markdown("---")

    # Diagnostic & Health Banner
    with st.expander("📡 System Health & Engine Diagnostics", expanded=False):
        health = check_system_health()
        h_col1, h_col2, h_col3, h_col4 = st.columns(4)
        with h_col1:
            st.metric("System Status", health.get("status", "UNKNOWN"))
        with h_col2:
            st.metric("Compute Device", health.get("compute", {}).get("configured_device", "CPU").upper())
        with h_col3:
            free_disk = health.get("storage", {}).get("free_space_gb", 0)
            st.metric("Free Disk", f"{free_disk:.1f} GB")
        with h_col4:
            mlflow_st = health.get("mlflow", {}).get("status", "DISABLED")
            st.metric("MLflow Tracking", mlflow_st)

    # 2-Column Detail View: Recent Papers & Activity Log
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### 📚 Recent Research Papers")
        if papers:
            for p in papers[:5]:
                st.markdown(
                    f"""
                    <div class="lens-card">
                        <div class="lens-card-header">
                            <strong style="color: #F8FAFC; font-size: 0.95rem;">{p.title}</strong>
                            <span class="lens-badge badge-info">{p.year or 'Year N/A'}</span>
                        </div>
                        <p style="margin: 0; font-size: 0.85rem; color: #94A3B8;">Authors: {p.authors or 'Not detected'}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            render_empty_state(
                title="No Papers Saved Yet",
                description="Upload and analyze research papers in the 'Paper Analysis & Q&A' tab to save them to this workspace.",
                icon="📄",
            )

    with col_right:
        st.markdown("### 🕒 Recent Activity Stream")
        activities = store.list_activity(ws_id)
        if activities:
            for act in activities[:8]:
                timestamp_str = act.timestamp[:19].replace("T", " ") if getattr(act, "timestamp", None) else ""
                st.markdown(
                    f"""
                    <div style="border-left: 2px solid #3B82F6; padding-left: 12px; margin-bottom: 12px;">
                        <div style="font-size: 0.88rem; color: #F8FAFC; font-weight: 500;">{act.summary}</div>
                        <div style="font-size: 0.78rem; color: #64748B;">{act.action} · {timestamp_str}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            render_empty_state(
                title="No Activity Logged",
                description="Activities will automatically record as you bookmark quotes, formulate hypotheses, and execute runs.",
                icon="📝",
            )
