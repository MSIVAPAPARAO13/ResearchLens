"""Upload component for research paper PDFs (Phase 5).

Provides a polished, secure file upload area with:
- Format validation (PDF only)
- Strict file size checks (100 MB limit)
- Guidance on supported document structures
"""

from typing import Optional, Tuple

import streamlit as st

from researchlens.config.settings import get_settings


def render_upload_component() -> Tuple[Optional[bytes], Optional[str]]:
    """Render the PDF upload UI widget and validate security constraints."""
    settings = get_settings()
    max_mb = settings.max_upload_size_bytes // (1024 * 1024)

    st.markdown(
        f"""
        <div style="background: rgba(37, 99, 235, 0.05); border: 1px dashed #3B82F6; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <strong style="color: #F8FAFC; font-size: 0.95rem;">📄 Academic Paper Ingestion</strong>
                <span class="lens-badge badge-info">PDF · Max {max_mb} MB</span>
            </div>
            <p style="margin: 0; font-size: 0.84rem; color: #94A3B8;">
                Upload academic publications, preprints (arXiv), or workshop papers. ResearchLens extracts text,
                reconstructs section boundaries, cleans academic noise, and builds local FAISS & BM25 indices.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Select or drag & drop a research paper PDF:",
        type=["pdf"],
        help=f"Strictly enforced maximum upload limit: {max_mb} MB. Unscanned, searchable PDFs recommended.",
        key="main_pdf_uploader",
    )

    if uploaded_file is None:
        return None, None

    file_bytes = uploaded_file.read()

    # Security Size Validation
    if len(file_bytes) > settings.max_upload_size_bytes:
        st.error(
            f"❌ File exceeds maximum permitted upload limit of {max_mb} MB "
            f"(Uploaded size: {len(file_bytes) / (1024 * 1024):.1f} MB)."
        )
        return None, None

    if len(file_bytes) == 0:
        st.error("❌ The selected file is completely empty (0 bytes). Please upload a valid document.")
        return None, None

    return file_bytes, uploaded_file.name
