"""Evidence-grounded Question Answering engine and source citation models."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from researchlens.config.settings import get_settings
from researchlens.utils.logging import get_logger

logger = get_logger("qa")

NO_EVIDENCE_ANSWER = "I could not find enough evidence in the uploaded paper to answer this question."


@dataclass
class SourceCitation:
    """Explicit citation pointing directly to the source page, section, and chunk."""
    page_number: Optional[int]
    section_name: str
    chunk_id: str
    relevance_score: float
    source_text: str

    @property
    def quote(self) -> str:
        return self.source_text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "section_name": self.section_name,
            "chunk_id": self.chunk_id,
            "relevance_score": round(self.relevance_score, 4),
            "source_text": self.source_text,
        }


@dataclass
class QAResponse:
    """Structured QA output containing evidence attribution and source citations."""
    question: str
    answer: str
    sources: List[SourceCitation] = field(default_factory=list)
    confidence: float = 0.0
    retrieved_chunks: List[Any] = field(default_factory=list)

    # Backward compatibility properties
    source_section: str = "Not detected"
    page_number: Optional[int] = None
    chunk_id: Optional[str] = None
    evidence_snippet: Optional[str] = None

    def __post_init__(self):
        if self.sources:
            first = self.sources[0]
            self.source_section = first.section_name
            self.page_number = first.page_number
            self.chunk_id = first.chunk_id
            self.evidence_snippet = first.source_text[:200]

    @property
    def confidence_score(self) -> float:
        return self.confidence

    @property
    def citations(self) -> List[SourceCitation]:
        return self.sources

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "confidence": round(self.confidence, 4),
            "sources": [s.to_dict() for s in self.sources],
        }


class BaseQuestionAnswerer(ABC):
    """Abstract interface for scientific question answering."""

    @abstractmethod
    def answer_question(self, question: str, retrieved_chunks: List[Any]) -> QAResponse:
        """Answer a question grounded strictly in provided context chunks."""
        pass


class LocalQuestionAnswerer(BaseQuestionAnswerer):
    """Controlled, evidence-grounded answerer constrained strictly to retrieved paper chunks.

    Guarantees no hallucination: If the retrieved chunks do not provide relevant
    evidence for the question, it refuses to answer.
    """

    def __init__(self, confidence_threshold: Optional[float] = None):
        settings = get_settings()
        self.threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else settings.similarity_threshold
        )

    def _extract_question_keywords(self, question: str) -> List[str]:
        """Extract informative keywords from the question, excluding common interrogatives."""
        stopwords = {
            "what", "which", "where", "when", "who", "why", "how", "is", "are",
            "was", "were", "the", "a", "an", "in", "on", "of", "to", "for", "with",
            "does", "do", "did", "paper", "this", "author", "authors", "show", "used",
        }
        words = re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", question.lower())
        return [w for w in words if w not in stopwords]

    def answer_question(self, question: str, retrieved_chunks: List[Any]) -> QAResponse:
        if not question or not question.strip():
            return QAResponse(
                question=question,
                answer="Please enter a valid question about the paper.",
                confidence=0.0,
            )

        if not retrieved_chunks:
            return QAResponse(
                question=question,
                answer=NO_EVIDENCE_ANSWER,
                confidence=0.0,
            )

        # Build citations and filter candidates
        citations: List[SourceCitation] = []
        valid_evidence_chunks: List[Any] = []

        for chunk in retrieved_chunks:
            text = getattr(chunk, "text", "")
            score = getattr(chunk, "score", 0.0)
            meta = getattr(chunk, "metadata", {})
            p_num = getattr(chunk, "page_number", meta.get("page_number"))
            s_name = getattr(chunk, "section_name", meta.get("section_name", "unknown"))
            c_id = getattr(chunk, "doc_id", getattr(chunk, "chunk_id", meta.get("chunk_id", "chunk")))

            # Check if chunk passed retrieval threshold or is a reranked candidate
            has_reranker = (getattr(chunk, "reranker_score", None) is not None) or ("reranker_score" in meta)
            is_valid = (score >= self.threshold) or has_reranker

            if is_valid and text.strip():
                if has_reranker:
                    norm_score = meta.get("hybrid_score", meta.get("retrieval_score"))
                    if norm_score is not None and float(norm_score) > 0.0:
                        rel_score = float(norm_score)
                    else:
                        import math
                        rel_score = round(1.0 / (1.0 + math.exp(-max(min(float(score), 10.0), -10.0))), 4)
                else:
                    rel_score = float(score)

                valid_evidence_chunks.append(chunk)
                citations.append(
                    SourceCitation(
                        page_number=p_num,
                        section_name=s_name,
                        chunk_id=c_id,
                        relevance_score=rel_score,
                        source_text=text.strip(),
                    )
                )

        if not valid_evidence_chunks or not citations:
            return QAResponse(
                question=question,
                answer=NO_EVIDENCE_ANSWER,
                confidence=0.0,
                retrieved_chunks=retrieved_chunks,
            )

        # Evidence validation: check if the retrieved chunks contain query concepts
        keywords = self._extract_question_keywords(question)
        combined_text = " ".join([c.source_text.lower() for c in citations])

        def matches_keyword(kw: str, text: str) -> bool:
            if kw in text:
                return True
            stem = kw.rstrip("sedign")
            if len(stem) >= 4 and stem in text:
                return True
            return False

        # If question had specific keywords, at least one key concept must be present in evidence
        if keywords and not any(matches_keyword(kw, combined_text) for kw in keywords):
            return QAResponse(
                question=question,
                answer=NO_EVIDENCE_ANSWER,
                confidence=0.0,
                sources=[],
                retrieved_chunks=retrieved_chunks,
            )

        # Grounded answer synthesis: extract the top matching sentences from the evidence chunks
        matched_sentences: List[str] = []
        for citation in citations:
            sentences = re.split(r"(?<=[.!?])\s+", citation.source_text)
            for s in sentences:
                s_clean = s.strip()
                if len(s_clean) < 20:
                    continue
                s_lower = s_clean.lower()
                # Score sentence by query keyword overlap
                match_count = sum(1 for kw in keywords if matches_keyword(kw, s_lower))
                if match_count > 0:
                    matched_sentences.append((match_count, s_clean))

        matched_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [s for _, s in matched_sentences[:3]]

        if top_sentences:
            answer_text = " ".join(top_sentences)
        else:
            # Fall back to the most salient sentence from the top retrieved chunk
            primary_chunk_text = citations[0].source_text
            first_sentence = re.split(r"(?<=[.!?])\s+", primary_chunk_text)[0]
            answer_text = first_sentence.strip()

        mean_confidence = sum(c.relevance_score for c in citations) / len(citations)

        return QAResponse(
            question=question,
            answer=answer_text,
            sources=citations,
            confidence=mean_confidence,
            retrieved_chunks=retrieved_chunks,
        )


# Backward compatibility aliases
BaseQuestionAnswering = BaseQuestionAnswerer
ExtractiveQASystem = LocalQuestionAnswerer
