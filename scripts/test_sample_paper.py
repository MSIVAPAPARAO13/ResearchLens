"""End-to-end verification script testing ResearchLens with sample research paper."""

import os

from researchlens.extraction import extract_research_profile
from researchlens.ingestion import load_pdf
from researchlens.preprocessing import create_document_chunks, split_into_sections
from researchlens.qa import LocalQuestionAnswerer
from researchlens.retrieval import (
    BM25Retriever,
    DenseRetriever,
    FAISSVectorStore,
    MockEmbeddingModel,
    MockReranker,
    ScientificRetrievalPipeline,
)
from researchlens.summarization import generate_executive_summary, generate_tldr
from researchlens.workspace.store import WorkspaceStore


def test_sample_paper_pipeline():
    pdf_path = "data/sample_research_paper.pdf"
    assert os.path.exists(pdf_path), f"File {pdf_path} not found"

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    print("=== 1. PDF INGESTION ===")
    doc = load_pdf(pdf_bytes, filename="sample_research_paper.pdf")
    print(f"Filename: {doc.filename}")
    print(f"Pages: {doc.page_count}")
    print(f"Total Characters: {len(doc.full_text)}")
    assert doc.page_count >= 3
    assert "Attention-Driven Hybrid RAG" in doc.full_text

    print("\n=== 2. SCIENTIFIC SECTION PARSING ===")
    sections = split_into_sections(doc.full_text)
    for s_name, s_txt in sections.items():
        print(f"  - Section [{s_name}]: {len(s_txt)} chars")
    assert len(sections) >= 3

    print("\n=== 3. SUMMARIZATION ENGINE ===")
    tldr = generate_tldr(doc.full_text)
    print(f"TL;DR: {tldr}")
    exec_summary = generate_executive_summary(doc.full_text)
    print(f"Executive Summary: {exec_summary[:180]}...")
    assert len(tldr) > 0
    assert len(exec_summary) > 0

    print("\n=== 4. RESEARCH PROFILE & METADATA EXTRACTION ===")
    profile = extract_research_profile(doc)
    print(f"Paper Title: {profile.title}")
    print(f"Authors: {[a.name if hasattr(a, 'name') else str(a) for a in profile.authors]}")
    print(f"Methods Identified: {[m.name if hasattr(m, 'name') else str(m) for m in profile.methods]}")
    print(f"Datasets Identified: {[d.name if hasattr(d, 'name') else str(d) for d in profile.datasets]}")
    print(f"Metrics Identified: {[m.name if hasattr(m, 'name') else str(m) for m in profile.metrics]}")
    print(f"Limitations Extracted: {[lim.description if hasattr(lim, 'description') else str(lim) for lim in profile.limitations]}")

    print("\n=== 5. CHUNKING & HYBRID RETRIEVAL PIPELINE ===")
    chunks = create_document_chunks(doc)
    print(f"Token-Bounded Chunks Generated: {len(chunks)}")
    assert len(chunks) > 0

    emb_model = MockEmbeddingModel(dimension=128)
    embeddings = emb_model.embed_documents(chunks)
    vector_store = FAISSVectorStore()
    vector_store.add_chunks(chunks, embeddings)

    dense_retriever = DenseRetriever(
        embedding_model=emb_model,
        vector_store=vector_store,
        similarity_threshold=0.1,
    )

    bm25 = BM25Retriever()
    bm25.index_chunks(chunks)
    reranker = MockReranker()

    retrieval_pipeline = ScientificRetrievalPipeline(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25,
        reranker=reranker,
    )

    query = "What datasets were used and what was the F1 score?"
    results = retrieval_pipeline.retrieve(query, top_k=3)
    print(f"Retrieved passages: {len(results)}")
    if results:
        print(f"Top Score: {results[0].score:.4f} (Page {results[0].page_number})")

    print("\n=== 6. EVIDENCE-GROUNDED QUESTION ANSWERING ===")
    qa_engine = LocalQuestionAnswerer(confidence_threshold=0.1)
    qa_response = qa_engine.answer_question(query, results)
    print(f"Query: {query}")
    print(f"Grounded Answer: {qa_response.answer}")
    print(f"Confidence Score: {qa_response.confidence_score}")
    print(f"Citations ({len(qa_response.citations)}):")
    for c in qa_response.citations:
        print(f"  - Page {c.page_number} ({c.section_name}): \"{c.quote[:100]}...\"")
    assert qa_response.answer is not None

    print("\n=== 7. WORKSPACE PERSISTENCE ===")
    from researchlens.workspace.schema import SavedPaper
    store = WorkspaceStore(base_dir="data/test_workspace")
    ws = store.create_workspace("Sample Paper Test Workspace")
    paper = SavedPaper(
        paper_id=doc.document_id,
        workspace_id=ws.workspace_id,
        title=profile.title or doc.filename,
        authors=[a.name if hasattr(a, 'name') else str(a) for a in profile.authors],
        year=profile.year or 2024,
    )
    store.save_paper(paper)
    loaded = store.list_papers(ws.workspace_id)
    print(f"Saved & Retrieved Paper: '{loaded[0].title}' (ID: {loaded[0].paper_id}) in workspace '{ws.name}'")
    assert len(loaded) > 0

    print("\nSUCCESS: All 7 pipeline stages passed cleanly with sample research paper!")


if __name__ == "__main__":
    test_sample_paper_pipeline()
