"""Multi-paper comparative matrix UI component."""

import pandas as pd
import streamlit as st

from researchlens.analysis.paper_comparison import PaperComparisonMatrix, PaperProfile


def render_comparison_view(matrix: PaperComparisonMatrix) -> None:
    """Render a comparative matrix table across multiple processed research papers."""
    st.subheader("📑 Cross-Paper Comparison Matrix")

    table_data = matrix.to_table_dict()
    if not table_data or not matrix.profiles:
        st.info(
            "Upload and process papers to populate the cross-paper comparison matrix. "
            "You can compare problem statements, methods, datasets, and benchmark results."
        )

        with st.expander("ℹ️ Preview Sample Matrix Schema"):
            sample_profile = PaperProfile(
                paper_title="Attention Is All You Need",
                authors="Vaswani et al.",
                year=2017,
                problem_statement="Sequence-to-sequence transduction without recurrence",
                dataset="WMT 2014 English-German",
                method="Multi-head Self-Attention",
                model="Transformer",
                metrics="BLEU: 28.4",
                results="State-of-the-art translation with reduced training time",
                limitations="Quadratic computational complexity with sequence length",
                future_work="Extension to other modalities",
            )
            sample_matrix = PaperComparisonMatrix([sample_profile])
            df = pd.DataFrame(sample_matrix.to_table_dict())
            st.dataframe(df, use_container_width=True)
        return

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True)
