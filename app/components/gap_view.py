"""Streamlit component rendering evidence-driven Research Gap Intelligence."""

from typing import Optional

import streamlit as st

from researchlens.analysis.research_gap_report import ResearchGapReport
from researchlens.extraction.schema import ResearchProfile


def render_gap_intelligence_view(
    report: Optional[ResearchGapReport], profile: Optional[ResearchProfile] = None
) -> None:
    """Render comprehensive Research-Gap Intelligence UI tab."""
    st.subheader("💡 Research-Gap Intelligence & Hypothesis Discovery")
    st.markdown(
        "Evidence-driven analysis converting extracted paper limitations, future work, and methodology "
        "evidence into **structured research gaps and candidate hypotheses** with full provenance."
    )

    if not report or not report.paper_title or report.paper_title == "Untitled Research Paper":
        st.info(
            "👆 Please upload and process a research paper PDF in the **Paper Analysis** tab first to generate Research-Gap Intelligence."
        )
        return

    # 1. Executive Summary
    st.markdown("### 📌 Executive Gap Summary")
    st.info(report.executive_summary)

    st.markdown("---")

    # 2. Top Identified Research Gaps
    st.markdown("### 🎯 Top Identified Research Gaps")
    top_gaps = [g for g in report.top_gaps if g.gap_type != "REFRUSAL"]

    if not top_gaps:
        st.warning(
            "⚠️ **Insufficient evidence detected**: The analyzed sections do not contain enough explicit limitations or detectable methodology gaps."
        )
    else:
        for idx, gap in enumerate(top_gaps, 1):
            badge_color = "#38BDF8" if gap.explicit_or_inferred == "EXPLICIT" else "#F59E0B"
            conf_color = "#22C55E" if gap.confidence_level == "High" else "#EAB308" if gap.confidence_level == "Medium" else "#EF4444"

            pages_str = ", ".join(map(str, gap.source_pages)) if gap.source_pages else "N/A"
            sections_str = ", ".join(gap.source_sections) if gap.source_sections else "N/A"

            st.markdown(
                f"""
                <div style="background: rgba(255, 255, 255, 0.03); border-left: 4px solid {badge_color}; padding: 16px; border-radius: 6px; margin-bottom: 16px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin: 0; color: #F8FAFC;">Gap #{idx}: {gap.title}</h4>
                        <div>
                            <span style="background: {badge_color}22; color: {badge_color}; border: 1px solid {badge_color}; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; margin-right: 6px;">
                                {gap.explicit_or_inferred}
                            </span>
                            <span style="background: {conf_color}22; color: {conf_color}; border: 1px solid {conf_color}; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; font-weight: 600;">
                                {gap.confidence_level} ({gap.confidence_score:.0%})
                            </span>
                        </div>
                    </div>
                    <p style="margin-top: 10px; margin-bottom: 8px; color: #E2E8F0;"><strong>Category:</strong> <code>{gap.category}</code> · <strong>Type:</strong> <code>{gap.gap_type}</code></p>
                    <p style="margin-top: 4px; margin-bottom: 8px; color: #CBD5E1;"><strong>Statement:</strong> {gap.statement}</p>
                    <p style="margin-top: 4px; margin-bottom: 8px; color: #94A3B8; font-style: italic;"><strong>Why this matters:</strong> {gap.rationale}</p>
                    <div style="background: rgba(56, 189, 248, 0.08); padding: 10px; border-radius: 4px; margin-top: 8px;">
                        <strong style="color: #38BDF8;">🚀 Potential Research Direction:</strong>
                        <p style="margin-top: 4px; margin-bottom: 0; color: #F1F5F9;">{gap.potential_direction}</p>
                    </div>
                    <div style="margin-top: 8px; font-size: 0.82rem; color: #64748B;">
                        📍 Provenance: <code>Page {pages_str}</code> · Section: <code>{sections_str}</code>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Evidentiary quote accordion
            if gap.supporting_evidence:
                with st.expander(f"🔍 View Supporting Evidence for Gap #{idx}"):
                    for ev in gap.supporting_evidence:
                        st.markdown(f"> *\"{ev.source_text}\"*")
                        st.caption(f"Page {ev.page_number} · Section: {ev.section_name} · Chunk: {ev.chunk_id}")

    st.markdown("---")

    # 3. Explicit Limitations vs Future Work
    col_lim, col_fw = st.columns(2)

    with col_lim:
        st.markdown("### ⚠️ Explicit Paper Limitations")
        if not report.explicit_limitations or report.explicit_limitations[0].value in ("Not detected", "Not explicitly stated"):
            st.write("*No explicit limitations were detected in the parsed text.*")
        else:
            for item in report.explicit_limitations:
                if item.value not in ("Not detected", "Not explicitly stated"):
                    p_str = f"Page {item.page_number}" if item.page_number else "Page N/A"
                    st.markdown(f"- **{item.value}**  \n  📍 `{p_str}` · `{item.section_name.upper()}`")

    with col_fw:
        st.markdown("### 🔮 Explicit Future Work Directions")
        if not report.explicit_future_work or report.explicit_future_work[0].value in ("Not detected", "Not explicitly stated"):
            st.write("*No explicit future work statements were detected in the parsed text.*")
        else:
            for item in report.explicit_future_work:
                if item.value not in ("Not detected", "Not explicitly stated"):
                    p_str = f"Page {item.page_number}" if item.page_number else "Page N/A"
                    st.markdown(f"- **{item.value}**  \n  📍 `{p_str}` · `{item.section_name.upper()}`")

    st.markdown("---")

    # 4. Candidate Hypotheses
    st.markdown("### 🧪 Potential Hypotheses (Candidate Research Directions)")
    st.warning("⚠️ **Notice:** Generated hypotheses represent candidate research directions derived from paper evidence and require independent scientific validation.")

    if not report.potential_hypotheses:
        st.write("*No hypotheses generated for this paper.*")
    else:
        for hyp in report.potential_hypotheses:
            st.markdown(
                f"""
                <div style="background: rgba(234, 179, 8, 0.05); border: 1px solid rgba(234, 179, 8, 0.2); padding: 14px; border-radius: 6px; margin-bottom: 12px;">
                    <strong style="color: #EAB308; font-size: 1.05rem;">{hyp.title}</strong>
                    <p style="margin-top: 6px; margin-bottom: 4px; color: #F8FAFC;"><strong>Hypothesis:</strong> {hyp.statement}</p>
                    <p style="margin-top: 4px; margin-bottom: 4px; color: #CBD5E1;"><strong>Motivation:</strong> {hyp.motivation}</p>
                    <p style="margin-top: 4px; margin-bottom: 0; color: #38BDF8;"><strong>Suggested Evaluation:</strong> {hyp.suggested_evaluation}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
