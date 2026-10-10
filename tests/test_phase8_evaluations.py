"""Comprehensive Phase 8 regression and gold-standard verification tests."""

import pytest

from evaluation.corpus_benchmark import build_10_paper_corpus
from researchlens.analysis import (
    ExternalPaper,
    LiteratureGapValidator,
    check_scientific_comparability,
    deduplicate_papers,
)
from researchlens.qa.question_answering import LocalQuestionAnswerer
from researchlens.retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    FAISSVectorStore,
    ScientificRetrievalPipeline,
    get_embedding_model,
)


@pytest.fixture(scope="module")
def corpus():
    return build_10_paper_corpus()


def test_10_paper_corpus_integrity(corpus):
    assert len(corpus) == 10
    domains = set(p.domain for p in corpus)
    assert len(domains) >= 7, "Corpus must span at least 7 distinct scientific domains"
    for paper in corpus:
        doc = paper.to_document()
        assert doc.filename.endswith(".pdf")
        assert len(doc.chunks) >= 4
        assert len(paper.qa_pairs) >= 4


def test_rag_and_citation_accuracy(corpus):
    """Test RAG QA answerability, evidence grounding, citations, and unsupported query refusal."""
    paper = corpus[0]  # SciTransformer NLP paper
    emb_model = get_embedding_model()
    embeddings = emb_model.embed_documents(paper.chunks)
    vs = FAISSVectorStore()
    vs.add_chunks(paper.chunks, embeddings)
    dense = DenseRetriever(emb_model, vs, similarity_threshold=0.0)

    bm25 = BM25Retriever()
    bm25.index_chunks(paper.chunks)

    reranker = CrossEncoderReranker()
    pipeline = ScientificRetrievalPipeline(dense, bm25, reranker, dense_weight=0.6, bm25_weight=0.4)
    qa_engine = LocalQuestionAnswerer(confidence_threshold=0.25)

    for qa in paper.qa_pairs:
        candidates = pipeline.retrieve(qa.question, strategy="hybrid_reranked", top_k=3)
        res = qa_engine.answer_question(qa.question, candidates)

        if not qa.is_supported:
            assert "could not find enough evidence" in res.answer.lower()
            assert res.confidence == 0.0
            assert len(res.sources) == 0
        else:
            assert res.confidence > 0.0
            assert len(res.sources) >= 1
            # Check citation metadata matches source chunk
            top_cit = res.sources[0]
            assert top_cit.page_number == qa.expected_page
            assert top_cit.section_name == qa.expected_section
            assert any(kw.lower() in res.answer.lower() for kw in qa.expected_answer_keywords)


def test_scientific_comparability_cases():
    """Verify strict comparability logic preventing invalid numerical assertions."""
    # Case 1: Same task, same dataset, same metric -> Comparable
    is_comp1, reason1 = check_scientific_comparability("F1", "SQuAD", "F1", "SQuAD")
    assert is_comp1 is True
    assert "Directly comparable" in reason1

    # Case 2: Same task, different dataset -> Not directly comparable
    is_comp2, reason2 = check_scientific_comparability("F1", "SQuAD", "F1", "PubMed")
    assert is_comp2 is False
    assert "Distinct evaluation datasets" in reason2

    # Case 3: Same dataset, different metric -> Not directly comparable
    is_comp3, reason3 = check_scientific_comparability("Accuracy", "ImageNet", "F1", "ImageNet")
    assert is_comp3 is False
    assert "Incompatible evaluation metrics" in reason3


def test_deduplication_comprehensive_cases():
    """Verify deduplication across DOIs, arXiv IDs, and normalized title variations."""
    p1 = ExternalPaper("p1", "Attention Is All You Need", doi="10.5555/3295222", citation_count=50)
    p2 = ExternalPaper("p2", "attention is all you need!", doi="10.5555/3295222", citation_count=80)  # DOI duplicate
    p3 = ExternalPaper("p3", "Attention Is All You Need (2017)", arxiv_id="1706.03762", citation_count=90)
    p4 = ExternalPaper("p4", "Attention is all you need.", arxiv_id="1706.03762", citation_count=10)   # arXiv duplicate
    p5 = ExternalPaper("p5", "BERT: Pre-training of Deep Bidirectional Transformers", doi="10.18653/v1/N19-1423")

    papers = [p1, p2, p3, p4, p5]
    deduped = deduplicate_papers(papers)

    assert len(deduped) == 3
    dois = {p.doi.lower() for p in deduped if p.doi}
    assert "10.5555/3295222" in dois
    assert "10.18653/v1/n19-1423" in dois


def test_gap_validation_taxonomy():
    """Verify gap validation classifications and rationales."""
    validator = LiteratureGapValidator(similarity_threshold=0.20)
    gap_query = "High GPU memory footprint during long-sequence transformer attention."

    # 1. Confirmed Open Gap
    open_papers = [
        ExternalPaper("op1", "Long Context Limitations", abstract="Quadratic GPU memory remains an unsolved bottleneck in transformers.", year=2024),
        ExternalPaper("op2", "Memory Constraints in Attention", abstract="Memory footprint continues to constrain long-sequence training.", year=2023),
    ]
    res_open = validator.validate_gap(gap_query, open_papers)
    assert res_open.status == "CONFIRMED_OPEN_GAP"

    # 2. Potentially Addressed Gap
    solved_papers = [
        ExternalPaper("sp1", "Linear Transformer Architecture", abstract="In this paper we solve quadratic GPU memory and eliminate the bottleneck.", year=2024)
    ]
    res_solved = validator.validate_gap(gap_query, solved_papers)
    assert res_solved.status == "POTENTIALLY_ADDRESSED"

    # 3. Unverified Gap
    unrel_papers = [
        ExternalPaper("up1", "Soil Nitrogen Balance", abstract="An empirical analysis of crop rotation chemistry.", year=2021)
    ]
    res_unver = validator.validate_gap(gap_query, unrel_papers)
    assert res_unver.status == "UNVERIFIED"
