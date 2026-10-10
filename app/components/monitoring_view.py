"""Streamlit UI component for Research Monitoring and Literature Evolution Intelligence."""

import uuid
from typing import Dict, List, Optional

import streamlit as st

from researchlens.analysis.gap_evolution import ResearchGapEvolutionRecord
from researchlens.analysis.monitoring_report import MonitoringReport, generate_monitoring_report
from researchlens.analysis.monitoring_service import (
    ResearchAlert,
    ResearchMonitoringService,
    detect_paper_changes,
)
from researchlens.analysis.research_topic import (
    MonitoringConfig,
    MonitoringRun,
    ResearchTopic,
    TopicStore,
)


def _init_topic_store() -> TopicStore:
    return TopicStore(storage_dir="data/topics")


def render_research_monitor_view(
    monitoring_service: Optional[ResearchMonitoringService] = None,
) -> None:
    """Render the Research Monitoring and Literature Evolution tab."""
    st.subheader("📡 Continuous Research Monitoring & Literature Evolution")
    st.markdown(
        "Track dynamic research topics across external scholarly providers over time. "
        "Automatically detect newly published papers, emerging architectures, new benchmark datasets, "
        "and monitor the lifecycle of scientific research gaps."
    )

    store = _init_topic_store()
    service = monitoring_service or ResearchMonitoringService(topic_store=store)

    # Initialize session state collections
    if "monitored_topics" not in st.session_state:
        st.session_state.monitored_topics = store.list_topics()
    if "topic_gap_histories" not in st.session_state:
        st.session_state.topic_gap_histories = {}
    if "topic_alerts" not in st.session_state:
        st.session_state.topic_alerts = {}

    topics: List[ResearchTopic] = st.session_state.monitored_topics

    # Top Control Bar: Topic Selection & Creation
    col_sel, col_new = st.columns([3, 1])

    with col_new:
        with st.popover("➕ Create New Topic", use_container_width=True):
            st.markdown("#### Define Research Topic to Monitor")
            new_name = st.text_input("Topic Name", placeholder="e.g. LLM Reasoning & Chain of Thought")
            new_query = st.text_input("Scholarly Search Query", placeholder="e.g. chain-of-thought large language models reasoning benchmark")
            new_desc = st.text_area("Description (Optional)", placeholder="Tracking reasoning architectures and evaluation datasets.")
            new_providers = st.multiselect(
                "Providers",
                ["arXiv", "Semantic Scholar", "Crossref", "PubMed"],
                default=["arXiv", "Semantic Scholar", "Crossref", "PubMed"],
            )
            col_y1, col_y2 = st.columns(2)
            with col_y1:
                year_f = st.number_input("Year From", min_value=1990, max_value=2030, value=2023)
            with col_y2:
                year_t = st.number_input("Year To", min_value=1990, max_value=2030, value=2026)

            max_res = st.slider("Max Results Per Search", min_value=5, max_value=50, value=20)

            if st.button("Save & Start Monitoring", type="primary"):
                if not new_name.strip() or not new_query.strip():
                    st.error("Please specify both a Topic Name and Search Query.")
                else:
                    t_id = f"top_{str(uuid.uuid4())[:8]}"
                    new_topic = ResearchTopic(
                        topic_id=t_id,
                        name=new_name.strip(),
                        query=new_query.strip(),
                        description=new_desc.strip(),
                        providers=new_providers,
                        year_from=year_f,
                        year_to=year_t,
                        max_results=max_res,
                        config=MonitoringConfig(
                            search_query=new_query.strip(),
                            providers=new_providers,
                            max_results=max_res,
                        ),
                    )
                    store.save_topic(new_topic)
                    st.session_state.monitored_topics = store.list_topics()
                    st.success(f"Topic '{new_name}' created successfully!")
                    st.rerun()

    if not topics:
        st.info(
            "📋 **No research topics are being monitored yet.** "
            "Click **'➕ Create New Topic'** above to configure a research query, "
            "select providers (arXiv, Semantic Scholar, Crossref, PubMed), and start tracking literature shifts over time."
        )
        return

    topic_names = [t.name for t in topics]
    with col_sel:
        selected_name = st.selectbox("Select Monitored Research Topic:", topic_names, index=0)

    selected_topic = next((t for t in topics if t.name == selected_name), topics[0])

    # Topic Action Bar
    st.markdown("---")
    col_act1, col_act2, col_act3, col_act4 = st.columns([2, 2, 2, 4])

    with col_act1:
        run_now_btn = st.button("🚀 Run Search Now", type="primary", use_container_width=True)

    with col_act2:
        delete_btn = st.button("🗑️ Delete Topic", use_container_width=True)

    with col_act3:
        st.caption(f"**Last Checked**: {selected_topic.last_checked_at[:16] if selected_topic.last_checked_at else 'Never'}")

    with col_act4:
        st.caption(f"**Query**: `{selected_topic.query}` | **Providers**: {', '.join(selected_topic.providers)}")

    if delete_btn:
        store.delete_topic(selected_topic.topic_id)
        st.session_state.monitored_topics = store.list_topics()
        st.success(f"Deleted topic '{selected_topic.name}'")
        st.rerun()

    # Execute Search Run If Requested
    if run_now_btn:
        with st.spinner(f"🔍 Querying scholarly literature for '{selected_topic.name}' across {len(selected_topic.providers)} providers..."):
            hist = st.session_state.topic_gap_histories.get(selected_topic.topic_id, {})
            run_res, new_alerts, updated_gaps = service.execute_monitoring_run(selected_topic, hist)

            st.session_state.topic_gap_histories[selected_topic.topic_id] = updated_gaps
            existing_alerts = st.session_state.topic_alerts.get(selected_topic.topic_id, [])
            st.session_state.topic_alerts[selected_topic.topic_id] = new_alerts + existing_alerts
            st.session_state.monitored_topics = store.list_topics()

            if run_res.status == "COMPLETED":
                st.success(f"✅ Monitoring run completed! Found {run_res.results_count} papers ({run_res.new_papers_count} new, {run_res.duplicate_count} deduplicated).")
            elif run_res.status == "PARTIAL":
                st.warning(f"⚠️ Run completed with partial provider coverage. Errors: {'; '.join(run_res.errors[:2])}")
            else:
                st.error(f"❌ Monitoring run failed. Errors: {'; '.join(run_res.errors)}")
            st.rerun()

    # Metric Dashboard Cards
    latest_run: Optional[MonitoringRun] = selected_topic.runs[-1] if selected_topic.runs else None
    gaps_record: Dict[str, ResearchGapEvolutionRecord] = st.session_state.topic_gap_histories.get(selected_topic.topic_id, {})
    pot_addressed_count = sum(1 for g in gaps_record.values() if g.status == "POTENTIALLY_ADDRESSED")

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Tracked Papers", len(selected_topic.history))
    with m2:
        st.metric("New Papers (Latest)", latest_run.new_papers_count if latest_run else 0)
    with m3:
        st.metric("Monitoring Runs", len(selected_topic.runs))
    with m4:
        st.metric("Active Gaps Tracked", len(gaps_record))
    with m5:
        st.metric("Potentially Addressed", pot_addressed_count)

    st.markdown("---")

    # Detailed Sub-Tabs
    sub_overview, sub_papers, sub_gaps, sub_alerts, sub_compare_export = st.tabs([
        "📊 Monitoring Timeline & Change Summary",
        "📄 Newly Discovered Papers",
        "🔬 Research Gap Evolution",
        "🔔 Evidence Alerts",
        "🔄 Run Comparison & Export",
    ])

    with sub_overview:
        st.markdown("### 📈 Longitudinal Monitoring Timeline")
        if not selected_topic.runs:
            st.info("No runs executed yet. Click **'🚀 Run Search Now'** above to perform your first monitoring run.")
        else:
            timeline_rows = []
            for idx, r in enumerate(selected_topic.runs, start=1):
                timeline_rows.append({
                    "Run": f"Run #{idx}",
                    "Date": r.started_at[:16].replace("T", " "),
                    "Status": r.status,
                    "Total Papers": r.results_count,
                    "New Papers": r.new_papers_count,
                    "Duplicates Caught": r.duplicate_count,
                    "Errors": len(r.errors),
                })
            st.dataframe(timeline_rows, use_container_width=True)

            # Change summary for latest run
            if latest_run:
                st.markdown("### 🔍 What Changed in the Latest Monitoring Run?")
                st.markdown(
                    f"> *Within the monitored literature corpus retrieved for query `{selected_topic.query}`, "
                    f"ResearchLens identified **{latest_run.new_papers_count} newly indexed publications** "
                    f"and deduplicated {latest_run.duplicate_count} candidate records across {', '.join(latest_run.providers)}.*"
                )

    with sub_papers:
        st.markdown("### 📄 Newly Discovered Papers in Latest Run")
        if not latest_run or not latest_run.new_paper_ids:
            st.info("No new papers discovered in the most recent run.")
        else:
            new_ids = set(latest_run.new_paper_ids)
            latest_new_papers = [p for p in latest_run.papers if p.paper_id in new_ids]

            for p in latest_new_papers:
                with st.expander(f"📌 {p.title} ({p.year or 'Year N/A'}) — {p.provider}", expanded=False):
                    c_a, c_b = st.columns([3, 1])
                    with c_a:
                        st.markdown(f"**Authors**: {', '.join(p.authors) if p.authors else 'Not detected'}")
                        if p.doi:
                            st.caption(f"**DOI**: {p.doi}")
                        if p.arxiv_id:
                            st.caption(f"**arXiv**: {p.arxiv_id}")
                        if p.abstract:
                            st.markdown(f"**Abstract Excerpt**: {p.abstract[:350]}...")
                    with c_b:
                        st.metric("Citations", p.citation_count)
                        if p.is_open_access:
                            st.badge("Open Access")
                        if p.url:
                            st.markdown(f"[🔗 View Source Paper]({p.url})")
                        if p.pdf_url:
                            st.markdown(f"[📥 Download PDF]({p.pdf_url})")

    with sub_gaps:
        st.markdown("### 🔬 Research Gap Evolution Lifecycle")
        st.caption(
            "Tracking whether previously observed scientific limitations are persistent, strengthening across recent papers, "
            "or potentially addressed by newly published methodologies."
        )

        if not gaps_record:
            st.info("No research gaps logged yet for this topic. Gaps are populated automatically as papers are monitored and analyzed.")
        else:
            for g_id, g in gaps_record.items():
                badge_color = {
                    "NEW": "#3B82F6",
                    "STRENGTHENING": "#F59E0B",
                    "PERSISTENT": "#6B7280",
                    "POTENTIALLY_ADDRESSED": "#10B981",
                    "WEAKENING": "#8B5CF6",
                }.get(g.status, "#6B7280")

                with st.container():
                    st.markdown(
                        f"""
                        <div style="border: 1px solid #E5E7EB; border-left: 4px solid {badge_color}; border-radius: 6px; padding: 12px; margin-bottom: 12px; background: #FFFFFF;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <strong style="font-size: 15px;">{g.gap_statement}</strong>
                                <span style="background: {badge_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold;">
                                    {g.status}
                                </span>
                            </div>
                            <p style="margin: 6px 0; color: #4B5563; font-size: 13px;">
                                <strong>Category:</strong> {g.category} | <strong>Observations:</strong> {g.observation_count} | <strong>First Observed:</strong> {g.first_detected[:10]}
                            </p>
                            <p style="margin: 4px 0; color: #374151; font-size: 13px;">
                                <strong>Supporting Literature:</strong> {', '.join(g.supporting_papers[:3])}
                            </p>
                            {f'<p style="margin: 4px 0; color: #047857; font-size: 13px;"><strong>Candidate Solutions in:</strong> {", ".join(g.addressed_by_papers)}</p>' if g.addressed_by_papers else ''}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    with sub_alerts:
        st.markdown("### 🔔 Evidence-Backed Monitoring Alerts")
        alerts: List[ResearchAlert] = st.session_state.topic_alerts.get(selected_topic.topic_id, [])

        if not alerts:
            st.info("No alerts generated yet. Significant literature shifts (new models, new datasets, addressed gaps) trigger alerts automatically.")
        else:
            for a in alerts:
                sev_color = "#10B981" if a.severity == "INFO" else ("#F59E0B" if a.severity == "IMPORTANT" else "#EF4444")
                st.markdown(
                    f"""
                    <div style="border-left: 4px solid {sev_color}; padding: 8px 12px; background: #F9FAFB; margin-bottom: 8px; border-radius: 4px;">
                        <strong>[{a.severity}] {a.title}</strong> — <span style="color: #6B7280; font-size: 12px;">{a.created_at[:16].replace('T', ' ')}</span>
                        <div style="font-size: 13px; margin-top: 4px;">{a.message}</div>
                        <div style="font-size: 12px; color: #4B5563; margin-top: 2px;"><em>Evidence:</em> {a.evidence}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with sub_compare_export:
        st.markdown("### 🔄 Run Comparison & Intelligence Export")

        if len(selected_topic.runs) >= 2:
            st.markdown("#### Compare Two Monitoring Runs")
            c_r1, c_r2 = st.columns(2)
            run_labels = [f"Run #{i+1} ({r.started_at[:10]})" for i, r in enumerate(selected_topic.runs)]
            with c_r1:
                sel_r1 = st.selectbox("Baseline Run (Run A):", run_labels, index=0)
            with c_r2:
                sel_r2 = st.selectbox("Comparison Run (Run B):", run_labels, index=len(run_labels) - 1)

            idx1 = run_labels.index(sel_r1)
            idx2 = run_labels.index(sel_r2)
            run_a = selected_topic.runs[idx1]
            run_b = selected_topic.runs[idx2]

            diff_new, diff_updated, _ = detect_paper_changes(run_b.papers, {p.paper_id: p for p in run_a.papers})
            st.write(f"**Differences from {sel_r1} to {sel_r2}:**")
            st.write(f"- Newly added papers in Run B: **{len(diff_new)}**")
            st.write(f"- Papers with updated metadata: **{len(diff_updated)}**")
        else:
            st.caption("Execute at least two runs to compare longitudinal differences.")

        st.markdown("---")
        st.markdown("#### 📥 Export Monitoring Intelligence Report")

        report: MonitoringReport = generate_monitoring_report(
            selected_topic,
            latest_run=latest_run,
            gap_history=gaps_record,
        )

        exp1, exp2, exp3 = st.columns(3)
        with exp1:
            st.download_button(
                "📄 Download Markdown Report",
                data=report.to_markdown(),
                file_name=f"{selected_topic.name.replace(' ', '_')}_monitoring_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with exp2:
            st.download_button(
                "📦 Download JSON Summary",
                data=report.to_json(),
                file_name=f"{selected_topic.name.replace(' ', '_')}_monitoring.json",
                mime="application/json",
                use_container_width=True,
            )
        with exp3:
            st.download_button(
                "📊 Export Tracked Papers (CSV)",
                data=report.to_csv(),
                file_name=f"{selected_topic.name.replace(' ', '_')}_papers.csv",
                mime="text/csv",
                use_container_width=True,
            )
