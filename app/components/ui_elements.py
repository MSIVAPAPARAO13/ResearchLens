"""Reusable UI elements and styled components for ResearchLens.

Provides standardized scientific presentation widgets:
- Page headers with context badges and live status indicators
- Stat and metric cards with luminous accents and icons
- Provenance citation panels with quote glyphs and confidence pills
- Empty and loading states with radial glow
- Status badges, editorial callouts, and section headings
- Public product landing page & capability tour
- SEO metadata and strict anti-indexing privacy controls
"""

import json
from pathlib import Path
from typing import Optional

import streamlit as st

from researchlens.workspace.store import WorkspaceStore


def inject_seo_metadata(
    page_type: str = "landing",
    title_suffix: str = "",
    canonical_url: str = "https://researchlens.ai/",
) -> None:
    """Inject semantic SEO metadata, Open Graph, JSON-LD, and strict privacy robots directives.

    Parameters
    ----------
    page_type: "landing" for public product tour, or "workspace" for private research sessions.
    title_suffix: Optional additional title text.
    canonical_url: Canonical platform URL.
    """
    is_public = page_type == "landing"
    robots_content = (
        "index, follow, max-snippet:-1, max-image-preview:large"
        if is_public
        else "noindex, nofollow, noarchive, nosnippet"
    )

    full_title = (
        f"ResearchLens — {title_suffix}"
        if title_suffix
        else "ResearchLens — AI-Powered Scientific Literature Intelligence & Guided MLOps"
    )
    meta_desc = (
        "Deterministic, explainable NLP and intelligence platform for academic research papers: "
        "structural section reconstruction, hybrid RAG with verified citations, and reproducible MLflow experiment tracking."
    )

    json_ld = {
        "@context": "https://schema.org",
        "@type": "SoftwareApplication",
        "name": "ResearchLens",
        "headline": "AI-Powered Scientific Literature Intelligence & Guided MLOps Platform",
        "description": meta_desc,
        "applicationCategory": "ResearchApplication",
        "operatingSystem": "Linux, macOS, Windows",
        "offers": {
            "@type": "Offer",
            "price": "0",
            "priceCurrency": "USD",
        },
        "author": {
            "@type": "Organization",
            "name": "ResearchLens Open Science Team",
        },
    }

    st.markdown(
        f"""
        <!-- Semantic SEO & Social Metadata -->
        <meta name="description" content="{meta_desc}" />
        <meta name="keywords" content="academic paper summarization, scientific literature review, hybrid RAG, BM25, FAISS, Cross-Encoder, MLflow tracking, research gap analysis, reproducibility" />
        <meta name="author" content="ResearchLens" />
        <meta name="robots" content="{robots_content}" />
        <link rel="canonical" href="{canonical_url}" />

        <!-- Open Graph / Social Sharing -->
        <meta property="og:type" content="website" />
        <meta property="og:site_name" content="ResearchLens" />
        <meta property="og:title" content="{full_title}" />
        <meta property="og:description" content="{meta_desc}" />
        <meta property="og:url" content="{canonical_url}" />

        <!-- Twitter Card -->
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content="{full_title}" />
        <meta name="twitter:description" content="{meta_desc}" />

        <!-- Schema.org JSON-LD Structured Data -->
        <script type="application/ld+json">
        {json.dumps(json_ld, indent=2)}
        </script>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(
    title: str,
    subtitle: str,
    workspace_context: Optional[str] = None,
    badge_label: Optional[str] = None,
    badge_variant: str = "info",
) -> None:
    """Render a unified scientific page header banner."""
    context_html = ""
    if workspace_context:
        context_html += (
            f'<span class="lens-context-badge">'
            f'<span class="lens-live-dot"></span> 📁 {workspace_context}'
            f'</span>'
        )
    if badge_label:
        context_html += f'<span class="lens-badge badge-{badge_variant}">{badge_label}</span>'

    st.markdown(
        f"""
        <div class="lens-header">
            <div class="lens-header-top">
                <h1 class="lens-title">{title}</h1>
                <div style="display: flex; align-items: center; gap: 8px;">{context_html}</div>
            </div>
            <p class="lens-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_card(
    label: str,
    number: str,
    subtext: Optional[str] = None,
    icon: Optional[str] = None,
    variant: str = "default",
) -> None:
    """Render an editorial stat metric card with optional icon and variant accents."""
    sub_html = f'<div class="lens-stat-sub">{subtext}</div>' if subtext else ""
    icon_html = (
        f'<div style="font-size: 1.25rem; background: rgba(255,255,255,0.06); width: 36px; height: 36px; display: flex; align-items: center; justify-content: center; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08);">{icon}</div>'
        if icon
        else ""
    )

    top_row = (
        f"""
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 6px;">
            <div class="lens-stat-label">{label}</div>
            {icon_html}
        </div>
        """
        if icon
        else f'<div class="lens-stat-label">{label}</div>'
    )

    st.markdown(
        f"""
        <div class="lens-stat-box lens-stat-{variant}">
            {top_row}
            <div class="lens-stat-number">{number}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(
    title: str,
    description: str,
    icon: str = "🔬",
) -> None:
    """Render an informative scientific empty state with radial glow."""
    st.markdown(
        f"""
        <div class="lens-empty-state">
            <div class="lens-empty-icon">{icon}</div>
            <div class="lens-empty-title">{title}</div>
            <div class="lens-empty-desc">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_citation_panel(
    page_number: Optional[int],
    section_name: str,
    source_text: str,
    confidence: Optional[float] = None,
    label: Optional[str] = None,
    chunk_id: Optional[str] = None,
) -> None:
    """Render a strict evidence citation panel with provenance indicators and quote glyphs."""
    page_str = f"Page {page_number}" if page_number is not None else "Page N/A"
    sec_str = section_name.upper() if section_name else "GENERAL"
    chunk_str = f" · ID: {chunk_id[:8]}" if chunk_id else ""

    conf_badge = ""
    if confidence is not None and confidence > 0:
        variant = "success" if confidence >= 0.75 else ("info" if confidence >= 0.5 else "warning")
        conf_badge = f'<span class="lens-badge badge-{variant}">Conf: {confidence:.0%}</span>'

    meta_info = f"📍 {page_str} · {sec_str}{chunk_str}"
    label_html = f'<strong style="color: #F8FAFC; font-size: 0.95rem;">{label}</strong>' if label else ""

    st.markdown(
        f"""
        <div class="lens-citation-panel">
            <div class="lens-citation-header">
                <div>{label_html}</div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span class="lens-citation-badge">{meta_info}</span>
                    {conf_badge}
                </div>
            </div>
            <p class="lens-citation-quote">"{source_text}"</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_approval_gate_banner(
    stage_name: str,
    approval_status: str,
    approved_by: Optional[str] = None,
    timestamp: Optional[str] = None,
    required_role: str = "RESEARCHER / OWNER",
) -> None:
    """Render a prominent human-in-the-loop approval gate callout."""
    status_badge = f'<span class="lens-badge badge-warning">{approval_status}</span>'
    if approval_status == "APPROVED":
        status_badge = '<span class="lens-badge badge-success">APPROVED</span>'
    elif approval_status in ("REJECTED", "BLOCKED"):
        status_badge = f'<span class="lens-badge badge-danger">{approval_status}</span>'

    approver_info = (
        f"<br><small style='color: #94A3B8;'>Authorized by: <strong style='color: #F8FAFC;'>{approved_by}</strong> at {timestamp}</small>"
        if approved_by
        else ""
    )

    st.markdown(
        f"""
        <div class="approval-gate-banner">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <strong style="color: #FBBF24; font-size: 1rem; display: flex; align-items: center; gap: 8px;">
                    🛡️ Human Researcher Approval Gate: {stage_name}
                </strong>
                {status_badge}
            </div>
            <p style="margin: 8px 0 0 0; font-size: 0.92rem; color: #CBD5E1; line-height: 1.5;">
                Strict scientific integrity requires explicit authorization before compute execution.
                Required Role: <code style="color: #38BDF8;">{required_role}</code>.{approver_info}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_heading(
    title: str,
    subtitle: Optional[str] = None,
    icon: Optional[str] = None,
) -> None:
    """Render a consistent section title with icon and optional subtitle."""
    icon_str = f"{icon} " if icon else ""
    sub_html = f'<p style="color: #94A3B8; font-size: 0.92rem; margin: 4px 0 16px 0;">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
        <div style="margin-top: 14px; margin-bottom: 8px;">
            <h3 style="margin: 0; color: #F8FAFC; font-size: 1.25rem;">{icon_str}{title}</h3>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_badge(label: str, variant: str = "info") -> str:
    """Generate HTML string for a styled status badge."""
    return f'<span class="lens-badge badge-{variant}">{label}</span>'


def render_public_landing_view(store: Optional[WorkspaceStore] = None) -> None:
    """Render the public product landing page showcasing capabilities, architecture, and security charter."""
    inject_seo_metadata(page_type="landing", title_suffix="Scientific Intelligence & Guided MLOps Platform")

    # Hero Banner
    st.markdown(
        """
        <div class="lens-hero">
            <div class="lens-hero-brand">🔬 Scientific SaaS Intelligence Platform</div>
            <h1 class="lens-hero-title">Deterministic Research Intelligence<br>& Guided Experimentation</h1>
            <p class="lens-hero-desc">
                ResearchLens transforms academic literature workflows with structural section reconstruction,
                two-stage hybrid RAG (BM25 + FAISS + Cross-Encoder), automated research gap discovery,
                and sandboxed MLflow execution governance. Built for scientific reproducibility, not probabilistic guesswork.
            </p>
            <div class="lens-hero-badges">
                <span class="lens-hero-badge-pill">⚡ Local-First Execution</span>
                <span class="lens-hero-badge-pill">🔒 Zero Cloud Leakage</span>
                <span class="lens-hero-badge-pill">📑 Deterministic Provenance</span>
                <span class="lens-hero-badge-pill">🛡️ Human Approval Gates</span>
                <span class="lens-hero-badge-pill">📊 MLflow Model Registry</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Architectural Blueprint / Core Pillars Grid
    st.markdown("### 🏛️ Core Architectural Pillars")
    st.markdown(
        """
        <div class="lens-feature-grid">
            <div class="lens-feature-card">
                <div class="lens-feature-icon">📑</div>
                <h3 class="lens-feature-title">Structural Paper Parsing</h3>
                <p class="lens-feature-desc">
                    Reconstructs canonical academic sections (Abstract, Methodology, Results, Limitations)
                    using deterministic boundary rules and noise reduction while strictly preserving scientific acronyms.
                </p>
            </div>
            <div class="lens-feature-card">
                <div class="lens-feature-icon">🔍</div>
                <h3 class="lens-feature-title">Two-Stage Hybrid RAG</h3>
                <p class="lens-feature-desc">
                    Fuses dense embeddings (MiniLM-L6-v2) with lexical BM25Okapi scoring, re-ranked via Cross-Encoder
                    with exact page, section, and chunk citations. Strict anti-hallucination refusals.
                </p>
            </div>
            <div class="lens-feature-card">
                <div class="lens-feature-icon">💡</div>
                <h3 class="lens-feature-title">Research Gap Intelligence</h3>
                <p class="lens-feature-desc">
                    Detects explicit limitations and future research directions across single papers and multi-paper collections
                    with semantic clustering and comparability enforcement.
                </p>
            </div>
            <div class="lens-feature-card">
                <div class="lens-feature-icon">🤖</div>
                <h3 class="lens-feature-title">13-Stage Guided Workflow</h3>
                <p class="lens-feature-desc">
                    End-to-end scientific progression from question formulation to peer-reviewed findings with mandatory
                    researcher approval gates before compute execution.
                </p>
            </div>
            <div class="lens-feature-card">
                <div class="lens-feature-icon">📊</div>
                <h3 class="lens-feature-title">MLflow Experiment Governance</h3>
                <p class="lens-feature-desc">
                    Comprehensive metric tracking, parameter sanitization, SHA-256 artifact hashing, reproducibility manifests,
                    and a five-tier scientific Model Registry.
                </p>
            </div>
            <div class="lens-feature-card">
                <div class="lens-feature-icon">🐳</div>
                <h3 class="lens-feature-title">Sandboxed Execution</h3>
                <p class="lens-feature-desc">
                    Controlled subprocesses, non-root Docker sandboxes, CPU/memory quotas, and image allowlists ensuring
                    hermetic computational safety.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Scientific Capability Comparison Matrix
    st.markdown("### ⚖️ Scientific Rigor: ResearchLens vs. Generic Prompt Wrappers")
    st.markdown(
        """
        <div class="lens-table-container">
            <table class="lens-table">
                <thead>
                    <tr>
                        <th>Capability / Dimension</th>
                        <th class="col-brand">ResearchLens Platform</th>
                        <th>Generic LLM Wrappers</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Document Parsing</strong></td>
                        <td class="col-brand">Deterministic section-boundary reconstruction & noise filter</td>
                        <td>Raw text flattening with lost section hierarchy</td>
                    </tr>
                    <tr>
                        <td><strong>Evidence Grounding</strong></td>
                        <td class="col-brand">Two-stage Hybrid RAG (BM25 + FAISS + Cross-Encoder)</td>
                        <td>Single dense embedding search with fuzzy retrieval</td>
                    </tr>
                    <tr>
                        <td><strong>Citation Fidelity</strong></td>
                        <td class="col-brand">Granular page numbers, section headers, chunk IDs & quote bounds</td>
                        <td>Vague or synthesized inline references</td>
                    </tr>
                    <tr>
                        <td><strong>Hallucination Defense</strong></td>
                        <td class="col-brand">Strict refusal contract if evidence is insufficient</td>
                        <td>Probabilistic confabulation without guardrails</td>
                    </tr>
                    <tr>
                        <td><strong>Experiment Governance</strong></td>
                        <td class="col-brand">MLflow tracking, SHA-256 manifests & reproducibility diffs</td>
                        <td>No execution tracking or provenance storage</td>
                    </tr>
                    <tr>
                        <td><strong>Scientific Integrity</strong></td>
                        <td class="col-brand">Mandatory Human Approval Gate before compute execution</td>
                        <td>Autonomous unverified execution</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Interactive Sample Paper Showcase
    st.markdown("### 🧪 Live Demonstration: Preloaded Academic Benchmark Paper")
    sample_pdf_path = Path("data/sample_research_paper.pdf")
    if sample_pdf_path.exists():
        st.info(
            "A standard synthetic evaluation paper (**'Scalable Multi-Head Attention for Deep Scientific Understanding'**) "
            "is packaged with the repository to test ingestion, extraction, and Q&A without uploading proprietary files."
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Benchmark Paper", "Attention Scaling")
        with col2:
            st.metric("Canonical Sections", "8 Detected")
        with col3:
            st.metric("Key Extraction", "Accuracy, BLEU, F1")
    else:
        st.caption("Sample evaluation paper available in `data/sample_research_paper.pdf`.")

    # Privacy, Security & Anti-Indexing Charter
    st.markdown(
        """
        <div class="lens-charter-box">
            <div class="lens-charter-title">🔒 Research Data Privacy & Search Engine Protection Charter</div>
            <p class="lens-charter-desc">
                <strong>Zero Public Exposure:</strong> Private research workspaces, uploaded manuscripts, notes,
                hypotheses, and MLflow experiment runs are strictly isolated from the public internet.
                <br><br>
                <strong>Strict Anti-Indexing:</strong> Authenticated application routes, workspace sessions, and artifact downloads
                are protected by <code>X-Robots-Tag: noindex, nofollow</code> headers and dynamic <code>noindex</code> meta controls,
                ensuring your proprietary scientific findings are never crawled or exposed through search engines.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
