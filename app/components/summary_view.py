"""Summary visualization component (Phase 5).

Renders:
- Document overview metrics
- High-contrast TL;DR cards
- Executive summaries
- Granular section summaries with character counts and expanders
"""

from typing import Dict

import streamlit as st

from app.components.ui_elements import render_stat_card
from researchlens.ingestion.document import Document


def render_document_overview(doc: Document, sections: Dict[str, str]) -> None:
    """Render high-level metrics describing the ingested document."""
    st.markdown("### 📊 Document Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_stat_card("Total Pages", str(doc.page_count), subtext="Full Document", icon="📄", variant="indigo")
    with col2:
        render_stat_card("Word Count", f"{doc.word_count:,}", subtext="Cleaned Text", icon="📝", variant="cyan")
    with col3:
        render_stat_card("Detected Sections", str(len(sections)), subtext="Structural Boundary", icon="📑", variant="purple")
    with col4:
        size_kb = doc.metadata.file_size_bytes / 1024
        size_str = f"{size_kb / 1024:.2f} MB" if size_kb > 1024 else f"{size_kb:.1f} KB"
        render_stat_card("File Size", size_str, subtext="PDF Payload", icon="💾", variant="emerald")


def render_summary_view(
    tldr: str,
    executive_summary: str,
    section_summaries: Dict[str, str],
) -> None:
    """Render structured hierarchical summaries."""
    st.markdown("### 📑 Hierarchical Synthesis")

    if tldr:
        st.markdown(
            f"""
            <div class="tldr-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: 700; color: #38BDF8; font-size: 0.85rem; letter-spacing: 0.05em; font-family: 'Space Grotesk', sans-serif;">
                        ⚡ TL;DR (ONE-SENTENCE SYNTHESIS)
                    </span>
                    <span class="lens-badge badge-info">CONDENSED</span>
                </div>
                <div style="color: #F8FAFC; font-size: 1.02rem; line-height: 1.6;">{tldr}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if executive_summary:
        st.markdown(
            f"""
            <div class="exec-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: 700; color: #34D399; font-size: 0.85rem; letter-spacing: 0.05em; font-family: 'Space Grotesk', sans-serif;">
                        🎯 EXECUTIVE SUMMARY
                    </span>
                    <span class="lens-badge badge-success">PEER COMPREHENSIVE</span>
                </div>
                <div style="color: #F8FAFC; font-size: 1.02rem; line-height: 1.65;">{executive_summary}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### 📖 Section-by-Section Summaries")
    if not section_summaries:
        st.info("No section summaries generated. Enable 'Generate Deep Summaries' in configuration to synthesize sections.")
        return

    for section_name, summary in section_summaries.items():
        char_count = len(summary)
        with st.expander(f"📌 {section_name.upper()} ({char_count:,} chars)", expanded=False):
            st.markdown(
                f"""
                <div style="font-size: 0.95rem; line-height: 1.6; color: #E2E8F0; padding: 6px 0;">
                    {summary}
                </div>
                """,
                unsafe_allow_html=True,
            )
