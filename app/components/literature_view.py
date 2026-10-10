"""Streamlit component rendering multi-paper Literature Intelligence dashboard."""

from typing import Optional

import streamlit as st

from researchlens.analysis import (
    LiteratureCollection,
    LiteratureDocument,
    LiteratureIntelligenceReport,
    generate_deterministic_document_id,
    generate_literature_report,
)
from researchlens.extraction import extract_research_profile
from researchlens.ingestion import load_pdf
from researchlens.preprocessing import split_into_sections


def render_literature_intelligence_view(
    collection: Optional[LiteratureCollection],
) -> None:
    """Render comprehensive Literature Intelligence UI tab."""
    st.subheader("📚 Multi-Paper Literature Intelligence Dashboard")
    st.markdown(
        "Analyze multiple research papers simultaneously to generate **cross-paper comparison matrices, "
        "method/dataset benchmarks, recurring limitation themes, empirical result conflicts, and literature-level research gaps**."
    )

    # Ingestion Modes: Local Upload or External Literature Discovery
    mode = st.radio(
        "Select Literature Source:",
        ["📁 Upload Local PDFs", "🌐 Discover External Literature (arXiv / Semantic Scholar / Crossref / PubMed)"],
        horizontal=True,
        key="lit_ingest_mode",
    )

    if mode == "📁 Upload Local PDFs":
        # 1. Batch Upload Section
        uploaded_files = st.file_uploader(
            "📁 Upload Multiple Research Paper PDFs (2 to 10 PDFs recommended)",
            type=["pdf"],
            accept_multiple_files=True,
            key="batch_pdf_uploader",
        )

        if uploaded_files:
            if st.button("🚀 Process Literature Collection", key="btn_process_collection"):
                new_collection = LiteratureCollection(
                    collection_id="col_user_batch",
                    collection_name=f"Collection ({len(uploaded_files)} Papers)",
                )
                progress_bar = st.progress(0, text="Processing batch PDFs...")

                for idx, file_obj in enumerate(uploaded_files):
                    filename = file_obj.name
                    file_bytes = file_obj.getvalue()
                    doc_id = generate_deterministic_document_id(filename, str(file_bytes[:1000]))

                    progress_bar.progress(
                        (idx + 1) / len(uploaded_files),
                        text=f"Extracting ResearchProfile for '{filename}' ({idx+1}/{len(uploaded_files)})...",
                    )

                    doc = load_pdf(file_bytes, filename=filename)
                    doc.sections = split_into_sections(doc.full_text)
                    profile = extract_research_profile(doc)

                    lit_doc = LiteratureDocument(
                        document_id=doc_id,
                        filename=filename,
                        title=profile.title or filename,
                        authors=[a.value for a in profile.authors if a.value != "Not detected"],
                        year=profile.year,
                        profile=profile,
                        document_obj=doc,
                    )
                    new_collection.add_document(lit_doc)

                progress_bar.empty()
                st.session_state.current_collection = new_collection
                st.session_state.current_lit_report = generate_literature_report(new_collection)
                st.success(f"✅ Successfully processed {len(new_collection.documents)} papers into Literature Collection!")

    else:
        # External Literature Discovery Section
        st.markdown("#### 🔍 Search External Literature by Topic")
        search_col1, search_col2 = st.columns([3, 1])
        with search_col1:
            search_query = st.text_input(
                "Enter Research Topic or Query:",
                placeholder="e.g., efficient transformer memory long context attention",
                key="ext_search_query",
            )
        with search_col2:
            provider_choices = st.multiselect(
                "Providers:",
                ["arXiv", "Semantic Scholar", "Crossref", "PubMed"],
                default=["arXiv", "Semantic Scholar"],
                key="ext_providers",
            )

        if st.button("🔎 Search External Literature", key="btn_search_ext"):
            if search_query.strip():
                from researchlens.analysis import UnifiedLiteratureSearchService
                with st.spinner("Searching repositories, deduplicating, and ranking papers..."):
                    search_service = UnifiedLiteratureSearchService()
                    provider_keys = [p.lower().replace(" ", "") for p in provider_choices]
                    discovered = search_service.search_literature(
                        query=search_query,
                        providers=provider_keys,
                        max_per_provider=4,
                        use_semantic_ranking=True,
                    )
                    st.session_state.discovered_papers = discovered
                    st.success(f"Found {len(discovered)} deduplicated and ranked candidate research papers!")

        discovered_papers = st.session_state.get("discovered_papers", [])
        if discovered_papers:
            st.markdown(f"**Discovered Papers for '{st.session_state.get('ext_search_query', '')}'** (Ranked by Relevance):")

            selected_paper_ids = []
            for p in discovered_papers:
                col_chk, col_info = st.columns([0.5, 9.5])
                with col_chk:
                    checked = st.checkbox("Select", key=f"chk_{p.paper_id}", value=True)
                    if checked:
                        selected_paper_ids.append(p.paper_id)
                with col_info:
                    rel_pct = int(p.relevance_score * 100)
                    st.markdown(
                        f"""
                        <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.08); padding: 10px; border-radius: 6px; margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between;">
                                <strong style="color: #38BDF8;">{p.title}</strong>
                                <span style="background: rgba(56,189,248,0.15); color: #38BDF8; padding: 2px 8px; border-radius: 10px; font-size: 0.8rem;">
                                    Relevance: {rel_pct}% · {p.provider}
                                </span>
                            </div>
                            <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 4px;">
                                Authors: {', '.join(p.authors[:3]) if p.authors else 'Unknown'} | Year: {p.year or 'N/A'} | Venue: {p.venue or 'N/A'} | Citations: {p.citation_count}
                            </div>
                            <div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 4px;">
                                {p.abstract[:240]}...
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            if st.button(f"📥 Import {len(selected_paper_ids)} Selected Papers into Literature Collection", key="btn_ingest_ext"):
                from researchlens.analysis import convert_external_paper_to_document
                with st.spinner("Importing, parsing sections, and generating ResearchProfiles..."):
                    if "current_collection" not in st.session_state or not st.session_state.current_collection:
                        st.session_state.current_collection = LiteratureCollection(
                            collection_id="col_external_discovery",
                            collection_name=f"External Literature ({search_query[:30]})",
                        )

                    added_count = 0
                    for p in discovered_papers:
                        if p.paper_id in selected_paper_ids:
                            lit_doc = convert_external_paper_to_document(p)
                            st.session_state.current_collection.add_document(lit_doc)
                            added_count += 1

                    st.session_state.current_lit_report = generate_literature_report(st.session_state.current_collection)
                    st.success(f"✅ Successfully ingested {added_count} external papers into the Literature Collection!")
                    st.rerun()

    report: Optional[LiteratureIntelligenceReport] = st.session_state.get("current_lit_report")

    if not report or report.paper_count == 0:
        st.info("👆 Please upload and process multiple PDF research papers above to view Literature Intelligence.")
        return

    # Export Buttons Header
    col_exp_md, col_exp_json, col_exp_csv = st.columns(3)
    with col_exp_md:
        st.download_button(
            "📥 Download Markdown Report",
            data=report.to_markdown(),
            file_name="literature_intelligence_report.md",
            mime="text/markdown",
            key="dl_lit_md",
        )
    with col_exp_json:
        import json
        st.download_button(
            "📥 Download JSON Report",
            data=json.dumps(report.to_dict(), indent=2),
            file_name="literature_intelligence_report.json",
            mime="application/json",
            key="dl_lit_json",
        )
    with col_exp_csv:
        st.download_button(
            "📥 Download Matrix CSV",
            data=report.to_matrix_csv(),
            file_name="paper_comparison_matrix.csv",
            mime="text/csv",
            key="dl_lit_csv",
        )

    st.markdown("---")

    # Sub-tabs Navigation
    tab_ov, tab_mat, tab_meth, tab_ds, tab_res, tab_lims, tab_conf, tab_gaps = st.tabs([
        "📌 Overview",
        "📋 Paper Matrix",
        "🧠 Methods & Models",
        "📊 Datasets",
        "📈 Results & Comparability",
        "🔄 Recurring Limitations",
        "⚡ Conflicting Findings",
        "💡 Literature Gaps",
    ])

    # Tab 1: Overview
    with tab_ov:
        st.markdown("### 📌 Executive Literature Summary")
        st.info(report.executive_summary)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Papers Analyzed", report.paper_count)
        c2.metric("Methods Detected", len(report.method_table))
        c3.metric("Datasets Detected", len(report.dataset_table))
        c4.metric("Recurring Limitation Themes", len(report.recurring_limitations))

    # Tab 2: Paper Matrix
    with tab_mat:
        st.markdown("### 📋 Cross-Paper Comparison Matrix")
        matrix_dict = report.paper_matrix.to_table_dict()
        if matrix_dict:
            st.dataframe(matrix_dict, use_container_width=True)

    # Tab 3: Methods & Models
    with tab_meth:
        st.markdown("### 🧠 Methods & Model Usage Across Literature")
        if report.method_table:
            st.dataframe(report.method_table, use_container_width=True)

    # Tab 4: Datasets
    with tab_ds:
        st.markdown("### 📊 Benchmark Datasets & Corpora")
        if report.dataset_table:
            st.dataframe(report.dataset_table, use_container_width=True)

    # Tab 5: Results & Comparability
    with tab_res:
        st.markdown("### 📈 Result Comparisons & Scientific Comparability")
        if report.result_table:
            st.dataframe(report.result_table, use_container_width=True)

    # Tab 6: Recurring Limitations
    with tab_lims:
        st.markdown("### 🔄 Recurring Limitation Themes Across Papers")
        if not report.recurring_limitations:
            st.write("*No recurring limitation themes detected.*")
        else:
            for cluster in report.recurring_limitations:
                st.markdown(
                    f"""
                    <div style="background: rgba(255, 255, 255, 0.03); border-left: 4px solid #F59E0B; padding: 14px; border-radius: 6px; margin-bottom: 12px;">
                        <h4 style="margin: 0; color: #F8FAFC;">Theme: {cluster.theme_name}</h4>
                        <p style="margin-top: 6px; margin-bottom: 4px; color: #CBD5E1;">
                            <strong>Category:</strong> <code>{cluster.category}</code> ·
                            <strong>Supporting Papers ({cluster.recurrence_count}):</strong> {', '.join(cluster.supporting_papers)}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.expander(f"🔍 View Evidence Snippets ({len(cluster.supporting_evidence)} quotes)"):
                    for ev in cluster.supporting_evidence:
                        st.markdown(f"> *\"{ev.source_text}\"*  \n  📍 `Page {ev.page_number}` · `{ev.section_name.upper()}`")

    # Tab 7: Conflicting Findings
    with tab_conf:
        st.markdown("### ⚡ Potential Empirical Conflicts & Discrepancies")
        if not report.conflicts:
            st.success("✅ No conflicting empirical results detected across analyzed papers.")
        else:
            for conf in report.conflicts:
                st.markdown(
                    f"""
                    <div style="background: rgba(239, 68, 68, 0.05); border-left: 4px solid #EF4444; padding: 14px; border-radius: 6px; margin-bottom: 12px;">
                        <h4 style="margin: 0; color: #EF4444;">Conflict: {conf.model} ({conf.metric})</h4>
                        <p style="margin-top: 6px; margin-bottom: 4px; color: #F8FAFC;">
                            <strong>Paper A ({conf.paper_a_title}):</strong> <code>{conf.value_a}</code>
                        </p>
                        <p style="margin-top: 2px; margin-bottom: 4px; color: #F8FAFC;">
                            <strong>Paper B ({conf.paper_b_title}):</strong> <code>{conf.value_b}</code>
                        </p>
                        <p style="margin-top: 4px; margin-bottom: 0; color: #CBD5E1;">
                            <strong>Comparability Status:</strong> <code>{conf.comparability_status}</code> — {conf.comparison_reason}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Tab 8: Literature-Level Gaps
    with tab_gaps:
        st.markdown("### 💡 Literature-Level Research Gaps")
        if not report.literature_gaps or report.literature_gaps[0].gap_id == "lit_gap_refusal":
            st.info("Insufficient evidence to identify reliable literature-level research gaps.")
        else:
            for idx, gap in enumerate(report.literature_gaps, 1):
                badge_color = "#38BDF8"
                conf_color = "#22C55E" if gap.confidence_level == "High" else "#EAB308"

                st.markdown(
                    f"""
                    <div style="background: rgba(255, 255, 255, 0.03); border-left: 4px solid {badge_color}; padding: 16px; border-radius: 6px; margin-bottom: 16px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h4 style="margin: 0; color: #F8FAFC;">Literature Gap #{idx}: {gap.title}</h4>
                            <span style="background: {conf_color}22; color: {conf_color}; border: 1px solid {conf_color}; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; font-weight: 600;">
                                {gap.confidence_level} ({gap.confidence_score:.0%})
                            </span>
                        </div>
                        <p style="margin-top: 8px; margin-bottom: 6px; color: #E2E8F0;">
                            <strong>Category:</strong> <code>{gap.category}</code> · <strong>Recurrence:</strong> {gap.recurrence_count} Papers ({', '.join(gap.supporting_papers)})
                        </p>
                        <p style="margin-top: 4px; margin-bottom: 6px; color: #CBD5E1;"><strong>Statement:</strong> {gap.statement}</p>
                        <p style="margin-top: 4px; margin-bottom: 6px; color: #94A3B8; font-style: italic;"><strong>Rationale:</strong> {gap.rationale}</p>
                        <div style="background: rgba(56, 189, 248, 0.08); padding: 10px; border-radius: 4px; margin-top: 8px;">
                            <strong style="color: #38BDF8;">🚀 Literature Research Direction:</strong>
                            <p style="margin-top: 4px; margin-bottom: 0; color: #F1F5F9;">{gap.potential_direction}</p>
                        </div>
                        <div style="margin-top: 8px; font-size: 0.82rem; color: #64748B;">
                            🔒 Independence Status: <code>{gap.independence_status}</code>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Interactive Gap Validation against External Literature
        st.markdown("---")
        st.markdown("#### 🔬 Validate Gap Against Discovered External Literature")
        gap_options = [f"Literature Gap #{i+1}: {g.statement[:80]}..." for i, g in enumerate(report.literature_gaps)]
        gap_options.append("✍️ Enter Custom Research Gap Statement")

        chosen_gap_label = st.selectbox("Select Gap Statement to Validate:", gap_options, key="sb_val_gap")
        if chosen_gap_label.startswith("✍️"):
            target_gap_stmt = st.text_input("Enter Research Gap Statement:", placeholder="e.g. Quadratic memory scaling during long context attention", key="ti_custom_gap")
        else:
            chosen_idx = int(chosen_gap_label.split(":")[0].replace("Literature Gap #", "")) - 1
            target_gap_stmt = report.literature_gaps[chosen_idx].statement

        discovered = st.session_state.get("discovered_papers", [])
        if not discovered:
            st.caption("💡 Tip: Search external papers above in the 'Discover External Literature' section to validate gaps against live external publications.")

        if st.button("🧪 Validate Against External Literature", key="btn_validate_gap"):
            if not target_gap_stmt.strip():
                st.warning("Please specify a gap statement to validate.")
            else:
                from researchlens.analysis import (
                    LiteratureGapValidator,
                    UnifiedLiteratureSearchService,
                )

                ext_papers_pool = discovered
                if not ext_papers_pool:
                    # Auto-discover top 4 papers for this gap query
                    with st.spinner("Retrieving external literature for validation..."):
                        srv = UnifiedLiteratureSearchService()
                        ext_papers_pool = srv.search_literature(target_gap_stmt, max_per_provider=2, use_semantic_ranking=False)

                with st.spinner("Analyzing semantic literature alignment and status..."):
                    validator = LiteratureGapValidator(similarity_threshold=0.25)
                    v_res = validator.validate_gap(target_gap_stmt, ext_papers_pool)

                    status_color = "#22C55E" if v_res.status == "CONFIRMED_OPEN_GAP" else "#EAB308"
                    st.markdown(
                        f"""
                        <div style="background: rgba(255, 255, 255, 0.04); border: 1px solid {status_color}; padding: 14px; border-radius: 6px; margin-top: 10px;">
                            <div style="display: flex; justify-content: space-between;">
                                <strong style="color: {status_color}; font-size: 1.05rem;">Status: {v_res.status}</strong>
                                <span style="color: #94A3B8;">Confidence: {v_res.confidence_score:.0%}</span>
                            </div>
                            <p style="margin-top: 6px; color: #CBD5E1;"><strong>Rationale:</strong> {v_res.rationale}</p>
                            <strong style="color: #38BDF8;">Suggested Research Directions:</strong>
                            <ul>
                                {"".join(f'<li style="color: #E2E8F0;">{d}</li>' for d in v_res.suggested_directions)}
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

