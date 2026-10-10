"""Cross-Encoder reranking module for second-stage candidate precision refinement."""

from abc import ABC, abstractmethod
from typing import List, Optional

import torch

from researchlens.config.settings import get_settings
from researchlens.retrieval.retriever import SearchResult
from researchlens.utils.logging import get_logger

logger = get_logger("reranker")

_RERANKER_CACHE: Optional["BaseReranker"] = None


class BaseReranker(ABC):
    """Abstract interface for query-document candidate reranking."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """Score candidate chunks using a cross-attentive model and return sorted top-k."""
        pass


class CrossEncoderReranker(BaseReranker):
    """Cross-Encoder reranker using sentence-transformers CrossEncoder.

    Lazily loads the model upon first inference. Computes joint cross-attention
    over (query, candidate_text) pairs to generate calibrated relevance scores.
    """

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.reranker_model_name
        self.device = device or ("cuda" if torch.cuda.is_available() and settings.device == "cuda" else "cpu")
        self._model = None

    @property
    def model(self):
        """Lazily initialize the CrossEncoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading CrossEncoder reranker: {self.model_name} on {self.device}")
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """Score candidate chunks against query and return top-k sorted by reranker score descending."""
        if not candidates or not query or not query.strip():
            return []

        settings = get_settings()
        k = top_k or settings.rerank_top_k

        # Construct (query, passage) pairs
        pairs = [(query, c.text) for c in candidates]

        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            logger.error(f"Error during CrossEncoder inference: {e}")
            # Fallback: preserve existing candidate order if reranker fails
            return candidates[:k]

        reranked: List[SearchResult] = []
        for i, cand in enumerate(candidates):
            r_score = float(scores[i])
            meta = dict(cand.metadata)
            meta["retrieval_score"] = cand.score
            meta["reranker_score"] = r_score

            reranked.append(
                SearchResult(
                    doc_id=cand.doc_id,
                    text=cand.text,
                    score=r_score,
                    page_number=cand.page_number,
                    section_name=cand.section_name,
                    metadata=meta,
                    reranker_score=r_score,
                )
            )

        # Sort descending by reranker score
        reranked.sort(key=lambda x: x.score, reverse=True)
        logger.info(
            f"Reranked {len(candidates)} candidates down to top {min(k, len(reranked))} items"
        )
        return reranked[:k]


class MockReranker(BaseReranker):
    """Deterministic mock reranker for unit tests without downloading neural models."""

    def __init__(self, score_offset: float = 0.5):
        self.score_offset = score_offset

    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        if not candidates or not query or not query.strip():
            return []

        settings = get_settings()
        k = top_k or settings.rerank_top_k

        q_terms = set(query.lower().split())
        reranked: List[SearchResult] = []

        for i, cand in enumerate(candidates):
            c_terms = set(cand.text.lower().split())
            overlap = len(q_terms.intersection(c_terms))
            # Deterministic mock score combining lexical overlap and original score
            mock_score = float(overlap * 1.5 + cand.score * 0.1 + (len(cand.text) % 7) * 0.01)

            meta = dict(cand.metadata)
            meta["retrieval_score"] = cand.score
            meta["reranker_score"] = mock_score

            reranked.append(
                SearchResult(
                    doc_id=cand.doc_id,
                    text=cand.text,
                    score=mock_score,
                    page_number=cand.page_number,
                    section_name=cand.section_name,
                    metadata=meta,
                    reranker_score=mock_score,
                )
            )

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:k]


def get_reranker(
    model_name: Optional[str] = None,
    use_mock: bool = False,
    device: Optional[str] = None,
) -> BaseReranker:
    """Retrieve singleton or mock reranker instance."""
    global _RERANKER_CACHE
    if use_mock:
        return MockReranker()

    target_name = model_name or get_settings().reranker_model_name
    if (
        _RERANKER_CACHE is None
        or not isinstance(_RERANKER_CACHE, CrossEncoderReranker)
        or _RERANKER_CACHE.model_name != target_name
    ):
        _RERANKER_CACHE = CrossEncoderReranker(model_name=target_name, device=device)
    return _RERANKER_CACHE
