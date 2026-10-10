"""Unit tests for evidence-grounded question answering and anti-hallucination constraints."""

from researchlens.qa.question_answering import (
    NO_EVIDENCE_ANSWER,
    ExtractiveQASystem,
    LocalQuestionAnswerer,
    QAResponse,
)
from researchlens.retrieval.retriever import SearchResult


def test_qa_grounded_answer_with_citations():
    answerer = LocalQuestionAnswerer(confidence_threshold=0.2)
    chunks = [
        SearchResult(
            doc_id="paper1_p04_c02",
            text="The experimental evaluation was conducted on the GLUE benchmark dataset and SQuAD v2.0.",
            score=0.91,
            page_number=4,
            section_name="dataset",
        )
    ]

    question = "What dataset was used?"
    response: QAResponse = answerer.answer_question(question, chunks)

    assert response.question == question
    assert "GLUE" in response.answer or "SQuAD" in response.answer
    assert response.confidence >= 0.8
    assert len(response.sources) == 1

    source = response.sources[0]
    assert source.page_number == 4
    assert source.section_name == "dataset"
    assert source.chunk_id == "paper1_p04_c02"
    assert source.relevance_score == 0.91
    assert "GLUE" in source.source_text


def test_qa_refusal_when_no_evidence_available():
    answerer = LocalQuestionAnswerer()
    response = answerer.answer_question("What optimizer was used?", [])
    assert response.answer == NO_EVIDENCE_ANSWER
    assert response.confidence == 0.0
    assert response.sources == []


def test_qa_anti_hallucination_on_unrelated_query():
    """Verify that asking unrelated questions returns refusal rather than guessing."""
    answerer = LocalQuestionAnswerer(confidence_threshold=0.2)
    chunks = [
        SearchResult(
            doc_id="chunk_01",
            text="We present convolutional neural networks for semantic image segmentation.",
            score=0.4,
            page_number=1,
            section_name="abstract",
        )
    ]

    # Query about completely unrelated knowledge absent from the text
    unrelated_question = "What is the capital city of France?"
    response = answerer.answer_question(unrelated_question, chunks)

    assert response.answer == NO_EVIDENCE_ANSWER
    assert response.confidence == 0.0


def test_qa_response_serialization():
    answerer = LocalQuestionAnswerer()
    chunks = [
        SearchResult(
            doc_id="c1",
            text="The model achieved an accuracy of 94.5%.",
            score=0.88,
            page_number=6,
            section_name="results",
        )
    ]
    resp = answerer.answer_question("What accuracy was achieved?", chunks)
    d = resp.to_dict()
    assert d["question"] == "What accuracy was achieved?"
    assert "accuracy" in d["answer"].lower() or "94.5" in d["answer"]
    assert len(d["sources"]) == 1
    assert d["sources"][0]["page_number"] == 6


def test_extractive_qa_backward_compatibility():
    engine = ExtractiveQASystem()
    assert isinstance(engine, LocalQuestionAnswerer)
