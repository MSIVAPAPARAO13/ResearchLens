"""Paper Intelligence UI component rendering structured ResearchProfile with provenance cards."""

from typing import Any, Dict, List, Optional

import streamlit as st

from researchlens.extraction.schema import (
    ExtractedItem,
    ResearchProfile,
)


def _render_provenance_badge(item: Any) -> str:
    p_num = getattr(item, "page_number", None)
    sec = getattr(item, "section_name", "unknown")
    page_str = f"Page {p_num}" if p_num else "Page N/A"
    return f"📍 `{page_str}` · `{sec.upper()}`"


def _render_item_card(item: ExtractedItem, label: str = ""):
    val = item.value
    if val in ("Not detected", "Not explicitly stated"):
        st.write(f"*{val}*")
        return

    prov = _render_provenance_badge(item)
    conf_str = f"Confidence: {item.confidence:.0%}" if item.confidence > 0 else ""

    st.markdown(
        f"""
        <div style="background: rgba(255, 255, 255, 0.03); border-left: 3px solid #38BDF8; padding: 12px; border-radius: 4px; margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <strong style="color: #F8FAFC; font-size: 0.95rem;">{label or val}</strong>
                <span style="font-size: 0.8rem; color: #94A3B8;">{prov} {f'· {conf_str}' if conf_str else ''}</span>
            </div>
            {f'<p style="margin-top: 6px; margin-bottom: 0; font-size: 0.9rem; color: #CBD5E1; font-style: italic;">"{item.source_text}"</p>' if item.source_text and item.source_text != val else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_paper_intelligence_view(profile: Optional[ResearchProfile]) -> None:
    """Render comprehensive Paper Intelligence tab with structured ResearchProfile cards."""
    st.subheader("🔬 Paper Intelligence & Knowledge Extraction")
    st.markdown(
        "Structured scientific facts extracted directly from the paper. "
        "Every extracted item retains **page, section, and verbatim text provenance**."
    )

    if not profile or not profile.title or profile.title == "Not detected":
        st.info("👆 Please upload and process a research paper PDF in the **Paper Analysis** tab first to view Paper Intelligence.")
        return

    # 1. Paper Overview
    st.markdown("### 📑 1. Paper Overview")
    st.markdown(f"**Title:** `{profile.title}`")
    c_auth, c_org, c_yr = st.columns([2, 2, 1])
    with c_auth:
        author_names = ", ".join([a.value for a in profile.authors if a.value != "Not detected"])
        st.write(f"**Authors:** {author_names if author_names else 'Not detected'}")
    with c_org:
        org_names = ", ".join([o.value for o in profile.organizations if o.value != "Not detected"])
        st.write(f"**Organizations:** {org_names if org_names else 'Not detected'}")
    with c_yr:
        st.write(f"**Year:** `{profile.year or 'Not detected'}`")

    st.markdown("---")

    # 2. Research Problem & Objective
    col_prob, col_obj = st.columns(2)
    with col_prob:
        st.markdown("### 🎯 2. Research Problem")
        for item in profile.problem:
            _render_item_card(item)

    with col_obj:
        st.markdown("### 📌 3. Research Objective")
        for item in profile.objective:
            _render_item_card(item)

    st.markdown("---")

    # 4. Research Questions & Hypotheses
    col_rq, col_hyp = st.columns(2)
    with col_rq:
        st.markdown("### ❓ 4. Research Questions")
        for item in profile.research_questions:
            _render_item_card(item)

    with col_hyp:
        st.markdown("### 💡 Hypotheses")
        for item in profile.hypotheses:
            _render_item_card(item)

    st.markdown("---")

    # 5. Methods, Models & Algorithms
    st.markdown("### 🧠 5. Methods, Models & Algorithms")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("#### Models")
        for item in profile.models:
            _render_item_card(item)
    with m2:
        st.markdown("#### Methods")
        for item in profile.methods:
            _render_item_card(item)
    with m3:
        st.markdown("#### Algorithms")
        for item in profile.algorithms:
            _render_item_card(item)

    st.markdown("---")

    # 6. Datasets & Experimental Setup
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("### 📂 6. Datasets & Corpora")
        for item in profile.datasets:
            _render_item_card(item)
    with d2:
        st.markdown("### ⚙️ 7. Experimental Setup")
        for item in profile.experimental_setup:
            _render_item_card(item)

    st.markdown("---")

    # 8. Metrics, Quantitative Results & Baselines
    st.markdown("### 📊 8. Metrics & Quantitative Results")
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("#### Quantitative Results")
        if profile.results:
            for r in profile.results:
                prov = _render_provenance_badge(r)
                unit_str = f" ({r.unit})" if r.unit else ""
                st.markdown(
                    f"""
                    <div style="background: rgba(16, 185, 129, 0.08); border-left: 3px solid #10B981; padding: 10px; border-radius: 4px; margin-bottom: 8px;">
                        <strong>{r.metric}:</strong> <code style="font-size: 1.05rem;">{r.value}{unit_str}</code>
                        <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 4px;">{prov}</div>
                        {f'<p style="margin-top: 4px; margin-bottom: 0; font-size: 0.85rem; color: #CBD5E1; font-style: italic;">"{r.source_text}"</p>' if r.source_text else ''}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.write("*Not detected*")

    with r2:
        st.markdown("#### Baselines & Comparisons")
        if profile.baselines:
            for b in profile.baselines:
                prov = _render_provenance_badge(b)
                diff_str = f" (Diff: `{b.difference}`)" if b.difference else ""
                st.markdown(
                    f"""
                    <div style="background: rgba(56, 189, 248, 0.08); border-left: 3px solid #38BDF8; padding: 10px; border-radius: 4px; margin-bottom: 8px;">
                        <strong>Baseline:</strong> {b.baseline_model} {diff_str}
                        <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 4px;">{prov}</div>
                        {f'<p style="margin-top: 4px; margin-bottom: 0; font-size: 0.85rem; color: #CBD5E1; font-style: italic;">"{b.source_text}"</p>' if b.source_text else ''}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.write("*Not detected*")

    st.markdown("---")

    # 9. Contributions, Limitations & Future Work
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 🏆 9. Main Contributions")
        for item in profile.contributions:
            _render_item_card(item)
    with c2:
        st.markdown("### ⚠️ 10. Limitations")
        for item in profile.limitations:
            _render_item_card(item)
    with c3:
        st.markdown("### 🚀 11. Future Work")
        for item in profile.future_work:
            _render_item_card(item)

    st.markdown("---")
    st.markdown("### 🔑 12. Keywords & Salient Concepts")
    kw_vals = [k.value for k in profile.keywords if k.value != "Not detected"]
    if kw_vals:
        html_tags = "".join([f'<span class="lens-tag">{k}</span>' for k in kw_vals])
        st.markdown(html_tags, unsafe_allow_html=True)
    else:
        st.write("*Not detected*")


def render_metadata_view(
    keywords: List[str],
    entities: Dict[str, List[str]],
    methods: List[str],
    datasets: List[str],
    metrics: Dict[str, Any],
    gaps: Dict[str, List[str]],
) -> None:
    """Backward-compatible metadata view component."""
    st.subheader("🔍 Extracted Scientific Intelligence")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔑 Key Concepts & Phrases")
        if keywords and keywords != ["Not detected"]:
            html_tags = "".join([f'<span class="lens-tag">{k}</span>' for k in keywords])
            st.markdown(html_tags, unsafe_allow_html=True)
        else:
            st.write("Not detected")

        st.markdown("#### 🧠 Methods & Architectures")
        if methods and methods != ["Not detected"]:
            html_tags = "".join([f'<span class="lens-tag">{m}</span>' for m in methods])
            st.markdown(html_tags, unsafe_allow_html=True)
        else:
            st.write("Not detected")

        st.markdown("#### 📂 Datasets & Benchmarks")
        if datasets and datasets != ["Not detected"]:
            html_tags = "".join([f'<span class="lens-tag">{d}</span>' for d in datasets])
            st.markdown(html_tags, unsafe_allow_html=True)
        else:
            st.write("Not detected")

    with col2:
        st.markdown("#### 👥 Authors & Affiliations")
        authors = entities.get("authors", ["Not detected"])
        orgs = entities.get("organizations", ["Not detected"])
        st.write(f"**Authors:** {', '.join(authors)}")
        st.write(f"**Organizations:** {', '.join(orgs)}")

        st.markdown("#### 📈 Experimental Metrics & Results")
        if metrics and metrics != {"result": ["Not detected"]}:
            for m_name, vals in metrics.items():
                if vals and vals != ["Not detected"]:
                    st.write(f"- **{m_name}:** {', '.join(vals)}")
        else:
            st.write("Not detected")

    st.markdown("---")
    st.markdown("#### 🔬 Research Gaps & Future Directions")
    lims = gaps.get("potential_limitations", ["Not detected"])
    futures = gaps.get("potential_future_directions", ["Not detected"])

    with st.expander("⚠️ Potential Research Gaps & Limitations", expanded=False):
        for item in lims:
            st.markdown(f"- {item}")

    with st.expander("🚀 Potential Future Directions", expanded=False):
        for item in futures:
            st.markdown(f"- {item}")
