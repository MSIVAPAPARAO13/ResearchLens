"""Reusable UI elements and styled components for ResearchLens.

Provides standardized scientific presentation widgets:
- Page headers with context badges
- Stat and metric cards
- Provenance citation panels
- Empty and loading states
- Status badges and editorial callouts
"""

from typing import Optional

import streamlit as st


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
        context_html += f'<span class="lens-context-badge">📁 {workspace_context}</span>'
    if badge_label:
        context_html += f'<span class="lens-badge badge-{badge_variant}">{badge_label}</span>'

    st.markdown(
        f"""
        <div class="lens-header">
            <div class="lens-header-top">
                <h1 class="lens-title">{title}</h1>
                <div>{context_html}</div>
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
) -> None:
    """Render an editorial stat metric card."""
    sub_html = f'<div class="lens-stat-sub">{subtext}</div>' if subtext else ""
    st.markdown(
        f"""
        <div class="lens-stat-box">
            <div class="lens-stat-label">{label}</div>
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
    """Render an informative scientific empty state."""
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
    """Render a strict evidence citation panel with provenance indicators."""
    page_str = f"Page {page_number}" if page_number is not None else "Page N/A"
    sec_str = section_name.upper() if section_name else "GENERAL"
    chunk_str = f" · ID: {chunk_id[:8]}" if chunk_id else ""
    conf_str = f" · Conf: {confidence:.0%}" if confidence is not None and confidence > 0 else ""
    meta_info = f"📍 {page_str} · {sec_str}{chunk_str}{conf_str}"

    label_html = f'<strong style="color: #F8FAFC;">{label}</strong>' if label else ""

    st.markdown(
        f"""
        <div class="lens-citation-panel">
            <div class="lens-citation-header">
                {label_html}
                <span class="lens-citation-badge">{meta_info}</span>
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

    approver_info = f"<br><small>Authorized by: <strong>{approved_by}</strong> at {timestamp}</small>" if approved_by else ""

    st.markdown(
        f"""
        <div class="approval-gate-banner">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <strong>🛡️ Human Researcher Approval Gate: {stage_name}</strong>
                {status_badge}
            </div>
            <p style="margin: 6px 0 0 0; font-size: 0.9rem; color: #CBD5E1;">
                Strict scientific integrity requires explicit authorization before compute execution.
                Required Role: <code>{required_role}</code>.{approver_info}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
