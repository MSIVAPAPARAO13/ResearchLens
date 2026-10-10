"""ResearchLens: AI-Powered Research Paper Summarization & Scientific Intelligence Platform."""

import sys
from pathlib import Path

import streamlit as st

# Ensure repository root and src directory are on Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.components import (
    render_comparison_view,
    render_dashboard_view,
    render_document_overview,
    render_execution_view,
    render_gap_intelligence_view,
    render_literature_intelligence_view,
    render_metadata_view,
    render_mlops_infrastructure_view,
    render_mlops_view,
    render_page_header,
    render_paper_intelligence_view,
    render_qa_view,
    render_research_automation_view,
    render_research_monitor_view,
    render_summary_view,
    render_upload_component,
    render_workspace_view,
)
from researchlens.analysis import generate_research_gap_report
from researchlens.analysis.paper_comparison import PaperComparisonMatrix, PaperProfile
from researchlens.analysis.research_gaps import extract_research_gaps
from researchlens.config.settings import get_settings
from researchlens.extraction import (
    extract_datasets,
    extract_entities,
    extract_keywords,
    extract_methods,
    extract_metrics,
    extract_research_profile,
)
from researchlens.health import check_system_health
from researchlens.ingestion import load_pdf
from researchlens.preprocessing import create_document_chunks, split_into_sections
from researchlens.qa.question_answering import LocalQuestionAnswerer
from researchlens.retrieval import (
    BM25Retriever,
    DenseRetriever,
    FAISSVectorStore,
    ScientificRetrievalPipeline,
    get_embedding_model,
    get_reranker,
)
from researchlens.summarization import (
    generate_executive_summary,
    generate_tldr,
    get_device,
    load_model_and_tokenizer,
    summarize_section,
)
from researchlens.utils.logging import get_logger
from researchlens.workspace.store import WorkspaceStore

logger = get_logger("app")

# Page Configuration
st.set_page_config(
    page_title="ResearchLens — Scientific Intelligence Platform",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Custom CSS Design System
css_path = Path(__file__).parent / "styles" / "custom.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Cache heavy ML models
@st.cache_resource(show_spinner="Loading BART summarization model into memory...")
def initialize_cached_summarizer():
    """Load and cache the BART summarization pipeline once per session."""
    return load_model_and_tokenizer()


@st.cache_resource(show_spinner="Loading SentenceTransformer embedding model into memory...")
def initialize_cached_embeddings():
    """Load and cache the SentenceTransformer embedding model once per session."""
    return get_embedding_model()


@st.cache_resource(show_spinner="Loading Cross-Encoder reranker model into memory...")
def initialize_cached_reranker():
    """Load and cache the Cross-Encoder reranker model once per session."""
    return get_reranker()


# Initialize session state
if "comparison_matrix" not in st.session_state:
    st.session_state.comparison_matrix = PaperComparisonMatrix()

if "current_document" not in st.session_state:
    st.session_state.current_document = None
if "current_profile" not in st.session_state:
    st.session_state.current_profile = None
if "current_gap_report" not in st.session_state:
    st.session_state.current_gap_report = None
if "current_collection" not in st.session_state:
    st.session_state.current_collection = None
if "current_lit_report" not in st.session_state:
    st.session_state.current_lit_report = None
if "current_pipeline" not in st.session_state:
    st.session_state.current_pipeline = None
if "current_retriever" not in st.session_state:
    st.session_state.current_retriever = None
if "current_qa_engine" not in st.session_state:
    st.session_state.current_qa_engine = None
if "indexed_filename" not in st.session_state:
    st.session_state.indexed_filename = None

settings = get_settings()
ws_store = WorkspaceStore()

# Production Authentication Gate
if settings.is_production and settings.api_secret_key:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("🔒 ResearchLens Secure Login")
        st.info("Public production access is restricted. Enter your access token or API Secret Key to continue.")
        auth_input = st.text_input("Access Key / API Secret Key", type="password")
        if st.button("Authenticate", key="auth_btn"):
            if auth_input == settings.api_secret_key:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid Secret Key. Access denied.")
        st.stop()

# Sidebar: Navigation & Configuration
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 6px;">
            <div style="background: linear-gradient(135deg, #6366F1, #06B6D4); width: 38px; height: 38px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.25rem; box-shadow: 0 0 14px rgba(99,102,241,0.4);">📑</div>
            <div>
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.3rem; font-weight: 700; color: #FFFFFF; letter-spacing: -0.02em; line-height: 1.1;">ResearchLens</div>
                <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 2px;">Scientific AI & MLOps</div>
            </div>
        </div>
        <div style="display: flex; gap: 8px; align-items: center; margin-bottom: 16px;">
            <span class="lens-badge badge-success">v1.0 PRO</span>
            <span style="font-size: 0.75rem; color: #64748B; display: inline-flex; align-items: center; gap: 5px;">
                <span class="lens-live-dot"></span> Core Ready
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_selection = st.radio(
        "Navigation",
        [
            "Research Overview",
            "Paper Analysis & Intelligence",
            "Literature & Cross-Paper Analysis",
            "Collaborative Workspace",
            "Experiments & MLOps",
            "Research Automation",
            "Architecture & Health",
        ],
        index=0,
        key="app_nav_radio",
    )

    st.markdown("---")

    # Document Status Pill
    if st.session_state.current_document:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(6, 182, 212, 0.05) 100%); border-left: 3px solid #10B981; border-top: 1px solid rgba(16,185,129,0.25); border-right: 1px solid rgba(16,185,129,0.25); border-bottom: 1px solid rgba(16,185,129,0.25); padding: 12px 14px; border-radius: 8px; margin-bottom: 14px; box-shadow: 0 2px 10px rgba(16,185,129,0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span style="font-size: 0.75rem; color: #34D399; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Indexed Paper</span>
                    <span class="lens-badge badge-success">ACTIVE</span>
                </div>
                <div style="font-size: 0.88rem; color: #F8FAFC; font-weight: 600; word-break: break-all; font-family: 'Space Grotesk', sans-serif;">{st.session_state.indexed_filename}</div>
                <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 6px; font-family: 'JetBrains Mono', monospace;">
                    ⚡ {len(st.session_state.current_document.chunks)} chunks · {st.session_state.current_document.page_count} pages
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Ingestion Controls
    with st.expander("🛠️ Ingestion Configuration", expanded=False):
        enable_full_pipeline = st.checkbox("Generate Deep Summaries", value=True, key="cfg_deep_sum")
        enable_extraction = st.checkbox("Extract Intelligence (Entities/Methods)", value=True, key="cfg_ext_intel")

    # Engine Information
    with st.expander("⚙️ Engine Specifications", expanded=False):
        st.caption(f"**Summarizer:** `{settings.model_name}`")
        st.caption(f"**Embeddings:** `{settings.embedding_model_name}`")
        st.caption(f"**Reranker:** `{settings.reranker_model_name}`")
        st.caption(f"**Vector Store:** `{settings.vector_store_type.upper()}`")
        st.caption("**Lexical Index:** `BM25Okapi`")
        st.caption(f"**Compute Device:** `{str(get_device()).upper()}`")


# Main Viewport Navigation Routing
if nav_selection == "Research Overview":
    render_dashboard_view(store=ws_store)

elif nav_selection == "Paper Analysis & Intelligence":
    render_page_header(
        title="Paper Analysis & Intelligence",
        subtitle="End-to-end single-paper research pipeline: PDF parsing, section reconstruction, summarization, entity extraction, gap discovery, and Q&A.",
        workspace_context="Single-Paper Engine",
    )

    p_tabs = st.tabs([
        "📄 Document Ingestion & Summaries",
        "🔬 Structured Paper Intelligence",
        "💡 Research Gaps & Hypotheses",
        "💬 Evidence-Grounded Q&A",
    ])

    with p_tabs[0]:
        file_bytes, filename = render_upload_component()

        if file_bytes and filename:
            try:
                with st.spinner("Parsing PDF and extracting structured sections..."):
                    doc = load_pdf(file_bytes, filename=filename)
                    sections = split_into_sections(doc.full_text)
                    doc.sections = sections

                render_document_overview(doc, sections)

                if not sections:
                    st.warning(
                        "No structured sections detected. The PDF might be scanned or formatted irregularly. "
                        "Displaying raw text overview."
                    )
                    sections = {"document": doc.full_text}
                    doc.sections = sections

                # Index Vector Chunks Once per Document Upload
                if st.session_state.indexed_filename != filename:
                    with st.spinner("Indexing token-bounded chunks into local FAISS and BM25 stores..."):
                        chunks = create_document_chunks(doc)
                        emb_model = initialize_cached_embeddings()
                        embeddings = emb_model.embed_documents(chunks)

                        store = FAISSVectorStore()
                        store.add_chunks(chunks, embeddings)
                        dense_retriever = DenseRetriever(
                            embedding_model=emb_model,
                            vector_store=store,
                            similarity_threshold=settings.similarity_threshold,
                        )

                        bm25_retriever = BM25Retriever()
                        bm25_retriever.index_chunks(chunks)

                        reranker = initialize_cached_reranker()

                        pipeline = ScientificRetrievalPipeline(
                            dense_retriever=dense_retriever,
                            bm25_retriever=bm25_retriever,
                            reranker=reranker,
                            dense_weight=settings.dense_weight,
                            bm25_weight=settings.bm25_weight,
                        )
                        qa_engine = LocalQuestionAnswerer(confidence_threshold=settings.similarity_threshold)

                        with st.spinner("Extracting ResearchProfile and generating Research-Gap Intelligence..."):
                            research_profile = extract_research_profile(doc)
                            gap_report = generate_research_gap_report(research_profile)

                        st.session_state.current_document = doc
                        st.session_state.current_profile = research_profile
                        st.session_state.current_gap_report = gap_report
                        st.session_state.current_pipeline = pipeline
                        st.session_state.current_retriever = dense_retriever
                        st.session_state.current_bm25 = bm25_retriever
                        st.session_state.current_qa_engine = qa_engine
                        st.session_state.indexed_filename = filename
                        st.success(
                            f"✅ '{filename}' indexed successfully! "
                            f"({len(chunks)} chunks across {doc.page_count} pages). "
                            f"Available across all analysis subtabs."
                        )

                # Summarization Pipeline Execution
                with st.spinner("Synthesizing section summaries..."):
                    initialize_cached_summarizer()

                    section_summaries = {}
                    if enable_full_pipeline:
                        progress_bar = st.progress(0, text="Summarizing sections...")
                        total_sections = len(sections)

                        for idx, (sec_name, sec_text) in enumerate(sections.items()):
                            progress_bar.progress(
                                (idx + 1) / total_sections,
                                text=f"Summarizing {sec_name.upper()}...",
                            )
                            summary = summarize_section(sec_name, sec_text)
                            if summary:
                                section_summaries[sec_name] = summary

                        progress_bar.empty()

                    tldr = ""
                    executive_summary = ""
                    if section_summaries:
                        first_text = (
                            section_summaries.get("abstract")
                            or section_summaries.get("introduction")
                            or next(iter(section_summaries.values()))
                        )
                        tldr = generate_tldr(first_text)
                        executive_summary = generate_executive_summary(section_summaries)

                render_summary_view(tldr, executive_summary, section_summaries)

                if enable_extraction:
                    with st.spinner("Extracting entities, keywords, and methods..."):
                        keywords = extract_keywords(doc.full_text)
                        entities = extract_entities(doc.full_text)
                        methods = extract_methods(doc.full_text)
                        datasets = extract_datasets(doc.full_text)
                        metrics = extract_metrics(doc.full_text)
                        gaps = extract_research_gaps(doc.full_text)

                    render_metadata_view(keywords, entities, methods, datasets, metrics, gaps)

                    if st.button("➕ Add to Cross-Paper Comparison Matrix", key="add_to_matrix_main"):
                        profile = PaperProfile(
                            paper_title=doc.metadata.title or filename,
                            authors=", ".join(entities.get("authors", ["Not detected"])),
                            problem_statement=tldr or "Not detected",
                            dataset=", ".join(datasets),
                            method=", ".join(methods),
                            model="BART / Scientific Baseline",
                            metrics=str(metrics),
                            results=executive_summary[:200] if executive_summary else "Not detected",
                            limitations="; ".join(gaps.get("potential_limitations", ["Not detected"]))[:200],
                            future_work="; ".join(gaps.get("potential_future_directions", ["Not detected"]))[:200],
                        )
                        st.session_state.comparison_matrix.add_paper(profile)
                        st.success(f"Added '{filename}' to Comparison Matrix!")

            except Exception as exc:
                logger.error(f"Error processing paper {filename}: {exc}", exc_info=True)
                st.error(
                    f"An error occurred while processing the document: {str(exc)}. "
                    "Please verify that the PDF contains readable text and is not corrupted."
                )

    with p_tabs[1]:
        render_paper_intelligence_view(st.session_state.get("current_profile"))

    with p_tabs[2]:
        render_gap_intelligence_view(
            report=st.session_state.get("current_gap_report"),
            profile=st.session_state.get("current_profile"),
        )

    with p_tabs[3]:
        render_qa_view(
            pipeline=st.session_state.get("current_pipeline"),
            retriever=st.session_state.get("current_retriever"),
            qa_engine=st.session_state.get("current_qa_engine"),
            document_loaded=bool(st.session_state.get("current_document")),
        )

elif nav_selection == "Literature & Cross-Paper Analysis":
    render_page_header(
        title="Literature & Cross-Paper Analysis",
        subtitle="Multi-paper synthesis: batch PDF analysis, cross-paper comparison matrices, conflict detection, and continuous topic monitoring.",
        workspace_context="Literature Hub",
    )

    l_tabs = st.tabs([
        "📚 Multi-Paper Literature Intelligence",
        "📊 Cross-Paper Comparison Matrix",
        "📡 Continuous Research Monitor",
    ])

    with l_tabs[0]:
        render_literature_intelligence_view(collection=st.session_state.get("current_collection"))

    with l_tabs[1]:
        render_comparison_view(st.session_state.comparison_matrix)

    with l_tabs[2]:
        render_research_monitor_view()

elif nav_selection == "Collaborative Workspace":
    render_page_header(
        title="Collaborative Research Workspace",
        subtitle="Shared project environment for team literature management, evidence bookmarks, hypothesis formulation, and research audit trails.",
        workspace_context="Workspace Lab",
    )
    render_workspace_view(store=ws_store)

elif nav_selection == "Experiments & MLOps":
    render_page_header(
        title="Reproducible Experiments & MLOps",
        subtitle="Sandboxed local and container execution, MLflow experiment tracking, artifact hashing, and model version governance.",
        workspace_context="MLOps Console",
    )

    e_tabs = st.tabs([
        "⚙️ Sandboxed Execution & History",
        "📊 MLflow Tracking & Model Registry",
        "🚀 Automated MLOps Infrastructure",
    ])

    with e_tabs[0]:
        render_execution_view(store=ws_store)

    with e_tabs[1]:
        render_mlops_view(store=ws_store)

    with e_tabs[2]:
        render_mlops_infrastructure_view(store=ws_store)

elif nav_selection == "Research Automation":
    render_page_header(
        title="Guided Research Automation",
        subtitle="13-stage evidence-grounded scientific progression with strict human researcher approval gates and peer-review audits.",
        workspace_context="Workflow Automation",
    )
    render_research_automation_view(store=ws_store)

elif nav_selection == "Architecture & Health":
    render_page_header(
        title="Platform Architecture & System Diagnostics",
        subtitle="Subsystem health probes, component specifications, and scientific anti-hallucination contracts.",
        workspace_context="Diagnostics",
    )

    a_tabs = st.tabs([
        "📡 System Health & Diagnostics",
        "🏛️ Engine Architecture & Specifications",
    ])

    with a_tabs[0]:
        health = check_system_health()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Overall Status", health.get("status", "UNKNOWN"))
        col2.metric("Compute Device", health.get("compute", {}).get("configured_device", "CPU").upper())
        col3.metric("Free Disk", f"{health.get('storage', {}).get('free_space_gb', 0):.1f} GB")
        col4.metric("Docker Daemon", "Active" if health.get("docker", {}).get("available") else "Not Connected")

        st.markdown("#### Diagnostic Report")
        st.json(health)

    with a_tabs[1]:
        st.markdown(
            """
            ### 🔬 Scientific Retrieval & Intelligence Architecture

            - **Two-Stage Hybrid Scientific Retrieval:** Combines `sentence-transformers/all-MiniLM-L6-v2` dense embeddings (via local FAISS-CPU) with lexical `BM25Okapi` keyword matching.
            - **Score Normalization & Linear Fusion:** Min-max normalizes disparate dense inner product and BM25 scores before weighted convex combination (default 0.6 dense / 0.4 BM25).
            - **Cross-Encoder Reranking:** Employs `cross-encoder/ms-marco-MiniLM-L6-v2` joint cross-attention to score `(query, passage)` pairs, significantly boosting top-1 precision and MRR.
            - **Full Provenance Preservation:** Every chunk retains its exact `page_number`, `section_name`, `chunk_id`, `retrieval_score`, and `reranker_score`.
            - **Anti-Hallucination Guarantee:** If retrieved excerpts do not provide sufficient evidentiary support, the system explicitly refuses to answer.
            - **Decoupled Architecture:** Streamlit acts strictly as an orchestration layer; NLP, embeddings, indexing, and QA logic reside in testable modular libraries.
            - **Production Security:** Hardened non-root containerization, request-size limits (100MB), rate limiting (50r/s), TLS 1.2/1.3 termination, and token-masked logging.
            """
        )
