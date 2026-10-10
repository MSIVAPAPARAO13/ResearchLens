"""Paper Question Answering UI component with hybrid retrieval and cross-encoder reranking (Phase 6)."""

from typing import Any, Optional

import streamlit as st

from app.components.ui_elements import render_citation_panel, render_empty_state
from researchlens.qa.question_answering import BaseQuestionAnswerer, QAResponse
from researchlens.retrieval.retriever import BaseRetriever

EXAMPLE_QUESTIONS = [
    "Select an example question...",
    "What problem does this paper address?",
    "What dataset was used?",
    "What architecture was proposed?",
    "What evaluation metrics were reported?",
    "What were the best results?",
    "What limitations did the authors identify?",
]


def render_qa_view(
    retriever: Optional[BaseRetriever] = None,
    qa_engine: Optional[BaseQuestionAnswerer] = None,
    document_loaded: bool = False,
    pipeline: Optional[Any] = None,
) -> None:
    """Render the interactive evidence-grounded Paper Q&A tab with retrieval strategy selection."""
    st.markdown("### 💬 Evidence-Grounded Scientific Q&A")
    st.markdown(
        "Ask questions about the uploaded paper. Answers are **strictly constrained** "
        "to retrieved excerpts with granular page, section, and chunk citations. "
        "If evidence is insufficient, the system explicitly refuses to hallucinate."
    )

    active_searcher = pipeline or retriever
    if not document_loaded or active_searcher is None:
        render_empty_state(
            title="No Document Loaded",
            description="Upload and index a research paper PDF in the 'Paper Analysis' tab to begin querying.",
            icon="📄",
        )
        return

    # Optional Retrieval Settings Section
    with st.expander("⚙️ Retrieval Strategy & Ranking Settings", expanded=False):
        col_strat, col_weights = st.columns([2, 3])
        with col_strat:
            selected_strategy_label = st.selectbox(
                "Retrieval Strategy:",
                ["Hybrid + Reranker", "Hybrid", "Dense", "BM25"],
                index=0,
                key="qa_retrieval_strategy_select",
                help="Select candidate retrieval and scoring methodology.",
            )
        with col_weights:
            st.caption(
                "**Hybrid + Reranker** (Default): Two-stage search retrieving dense + BM25 pools, "
                "normalized linear fusion, followed by Cross-Encoder joint-attention reranking."
            )

    strategy_key_map = {
        "Hybrid + Reranker": "hybrid_reranked",
        "Hybrid": "hybrid",
        "Dense": "dense",
        "BM25": "bm25",
    }
    strategy_code = strategy_key_map.get(selected_strategy_label, "hybrid_reranked")

    # Question Input
    col_input, col_examples = st.columns([3, 2])

    with col_examples:
        selected_example = st.selectbox(
            "💡 Preset questions:",
            EXAMPLE_QUESTIONS,
            index=0,
            key="qa_example_select",
        )

    with col_input:
        default_q = "" if selected_example == EXAMPLE_QUESTIONS[0] else selected_example
        question_text = st.text_input(
            "Enter research query:",
            value=default_q,
            placeholder="e.g. What dataset was evaluated in this study?",
            key="qa_question_input",
        )

    # Quick Suggestion Chips
    st.markdown(
        """
        <div style="display: flex; gap: 6px; flex-wrap: wrap; margin-top: -6px; margin-bottom: 12px;">
            <span style="font-size: 0.78rem; color: #64748B; align-self: center;">Suggested:</span>
            <span class="lens-tag">Architecture</span>
            <span class="lens-tag">Datasets</span>
            <span class="lens-tag">Baselines</span>
            <span class="lens-tag">Limitations</span>
            <span class="lens-tag">Key Metrics</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    submit_button = st.button("🔍 Search & Synthesize Answer", type="primary")

    if submit_button or (question_text and question_text != st.session_state.get("last_answered_q")):
        if not question_text.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner(f"Retrieving evidence via [{selected_strategy_label}] and verifying citations..."):
            if hasattr(active_searcher, "retrieve"):
                try:
                    retrieved_chunks = active_searcher.retrieve(question_text, strategy=strategy_code)
                except TypeError:
                    # Backward compatibility if retriever does not take strategy param
                    retrieved_chunks = active_searcher.retrieve(question_text)
            else:
                retrieved_chunks = []

            response: QAResponse = qa_engine.answer_question(question_text, retrieved_chunks)
            st.session_state.current_qa_response = response
            st.session_state.last_answered_q = question_text
            st.session_state.last_strategy_label = selected_strategy_label

    # Render previous or current response
    if "current_qa_response" in st.session_state and st.session_state.current_qa_response:
        res: QAResponse = st.session_state.current_qa_response
        last_strat = st.session_state.get("last_strategy_label", selected_strategy_label)

        st.markdown("---")
        st.markdown(f"#### ❓ Query: *{res.question}*")

        # Answer container
        is_refusal = "could not find enough evidence" in res.answer.lower()
        border_color = "var(--lens-status-danger)" if is_refusal else "var(--lens-status-success)"
        bg_gradient = (
            "linear-gradient(135deg, rgba(244, 63, 94, 0.1) 0%, rgba(15, 23, 42, 0.7) 100%)"
            if is_refusal
            else "linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(15, 23, 42, 0.7) 100%)"
        )
        status_tag = (
            '<span class="lens-badge badge-danger">INSUFFICIENT EVIDENCE REFUSAL</span>'
            if is_refusal
            else '<span class="lens-badge badge-success">GROUNDED IN EVIDENCE</span>'
        )

        st.markdown(
            f"""
            <div style="background: {bg_gradient}; backdrop-filter: blur(14px); border-left: 4px solid {border_color}; border-top: 1px solid var(--lens-border-subtle); border-right: 1px solid var(--lens-border-subtle); border-bottom: 1px solid var(--lens-border-subtle); padding: 20px 24px; border-radius: var(--lens-radius-md); margin: 18px 0; box-shadow: var(--lens-shadow-md);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <strong style="font-size: 1.1rem; color: #F8FAFC; font-family: 'Space Grotesk', sans-serif;">Synthesized Answer</strong>
                    <div style="display: flex; gap: 8px; align-items: center;">
                        {status_tag}
                        <span class="lens-badge badge-neutral">{last_strat}</span>
                    </div>
                </div>
                <p style="margin: 6px 0 0 0; font-size: 1.02rem; line-height: 1.65; color: #F1F5F9;">{res.answer}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if res.sources:
            st.markdown("#### 📑 Granular Evidence Citations")
            st.markdown(
                f"Retrieved **{len(res.sources)}** supporting excerpts "
                f"(Mean Relevance Confidence: **{res.confidence:.1%}**):"
            )

            for idx, src in enumerate(res.sources, start=1):
                render_citation_panel(
                    page_number=src.page_number,
                    section_name=src.section_name,
                    source_text=src.source_text,
                    confidence=src.relevance_score,
                    label=f"Citation #{idx}",
                    chunk_id=src.chunk_id,
                )

            # Debugging Candidate Information in an Expandable Section
            with st.expander("🔍 Retrieved Candidates & Ranking Breakdown (Diagnostic)", expanded=False):
                st.caption(f"Strategy: `{last_strat}` | Candidates Returned: `{len(res.retrieved_chunks)}`")

                for rank, chunk in enumerate(res.retrieved_chunks, start=1):
                    c_id = getattr(chunk, "doc_id", getattr(chunk, "chunk_id", f"chunk_{rank}"))
                    p_num = getattr(chunk, "page_number", "N/A")
                    s_name = getattr(chunk, "section_name", "unknown")
                    final_score = getattr(chunk, "score", 0.0)
                    meta = getattr(chunk, "metadata", {})

                    d_raw = meta.get("dense_raw_score", "N/A")
                    b_raw = meta.get("bm25_raw_score", "N/A")
                    hyb_score = meta.get("hybrid_score", "N/A")
                    r_score = meta.get("reranker_score", getattr(chunk, "reranker_score", "N/A"))

                    st.markdown(
                        f"**Rank {rank} — Chunk:** `{c_id}` | Score: `{final_score:.4f}` | Page: `{p_num}` | Section: `{s_name}`"
                    )
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.caption(f"**Dense Score:** {d_raw if isinstance(d_raw, str) else f'{d_raw:.4f}'}")
                    with c2:
                        st.caption(f"**BM25 Score:** {b_raw if isinstance(b_raw, str) else f'{b_raw:.4f}'}")
                    with c3:
                        st.caption(f"**Hybrid Score:** {hyb_score if isinstance(hyb_score, str) else f'{hyb_score:.4f}'}")
                    with c4:
                        st.caption(f"**Reranker Score:** {r_score if isinstance(r_score, str) else f'{r_score:.4f}'}")
                    st.text(f"Excerpt: {chunk.text[:140]}...")
                    st.markdown("---")

        elif not is_refusal:
            st.info("No explicit source citations retrieved for this query.")
